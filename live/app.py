"""Deployable read-only dashboard and scheduled BTC/ETH research alerts."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import sqlite3
import threading
import time

from flask import Flask,abort,jsonify,send_file

from .market import BinanceMarket,SYMBOLS
from .notify import WeComNotifier
from .predictor import Predictor

ROOT=Path(__file__).resolve().parents[1]
MINUTE=60000
HALF_HOUR=1800000
HOUR=3600000


class Worker:
    def __init__(self,model_dir,state_dir,webhook='',market=None):
        from .history import FeatureHistory
        self.model_dir=Path(model_dir)
        self.state_dir=Path(state_dir);self.state_dir.mkdir(parents=True,exist_ok=True)
        self.predictor=Predictor(model_dir)
        self.history=FeatureHistory(self.state_dir/'history.sqlite3')
        self.history.import_seed(self.model_dir,self.predictor.manifest)
        self.notifier=WeComNotifier(webhook,self.state_dir/'notifications.sqlite3')
        self.market=market or BinanceMarket()
        self.lock=threading.RLock()
        self.db=self.state_dir/'worker.sqlite3'
        with closing(sqlite3.connect(self.db,isolation_level=None)) as con:
            con.execute('CREATE TABLE IF NOT EXISTS processed (slot INTEGER PRIMARY KEY, state TEXT NOT NULL)')
            con.execute('CREATE TABLE IF NOT EXISTS admissions (symbol TEXT, signal_ms INTEGER, entry_ms INTEGER, settlement_ms INTEGER, direction INTEGER, outcome TEXT, net REAL, PRIMARY KEY(symbol,signal_ms))')
        self.state={'mode':'research_alerts','symbols':{},'last_signal':None,'last_decisions':[],
                    'last_error':None,'last_prediction_ms':None,'data_health':'starting',
                    'prediction_schedule':'每半小时收盘后约5秒；60秒后模拟入场，持有30分钟',
                    'price_report_schedule':'北京时间每小时整点，24小时',
                    'order_execution_enabled':False,'forward_validated':False}
        self.last_quote=0;self.last_label_poll=0;self.last_refresh_attempt=0
        self.stop=threading.Event()

    def snapshot(self):
        with self.lock:
            value=json.loads(json.dumps(self.state))
        now=int(time.time()*1000)
        for quote in value['symbols'].values():quote['stale']=now-quote['price_time_ms']>30000
        value['model']={'id':self.predictor.manifest['model_version'],
                        'trained_at':self.predictor.manifest['training_cutoff_utc'],
                        'rule_id':self.predictor.manifest['rule_id'],'max_age_days':35}
        value['notifications']=self.notifier.status()
        return value

    def _processed(self,slot):
        with closing(sqlite3.connect(self.db,isolation_level=None)) as con:
            return con.execute('SELECT state FROM processed WHERE slot=?',(slot,)).fetchone() is not None

    def _mark(self,slot,state):
        with closing(sqlite3.connect(self.db,isolation_level=None)) as con:con.execute('INSERT OR IGNORE INTO processed VALUES (?,?)',(slot,state))

    def admit(self,signal):
        """Causal paper budget: reserve all as-yet-unsettled loss risk at alert time."""
        entry=int(signal['entry_due_ms']);day=entry//86400000
        with closing(sqlite3.connect(self.db,isolation_level=None)) as con:
            con.execute('BEGIN IMMEDIATE')
            if con.execute('SELECT 1 FROM admissions WHERE symbol=? AND signal_ms=?',(signal['symbol'],signal['signal_ms'])).fetchone():
                con.rollback();return False,'duplicate_paper_signal'
            pending=con.execute('SELECT settlement_ms FROM admissions WHERE outcome IS NULL').fetchall()
            if any(row[0]>entry for row in pending):
                con.rollback();return False,'paper_position_open'
            loss=con.execute("SELECT count(*) FROM admissions WHERE outcome='loss' AND settlement_ms>=? AND settlement_ms<?",(day*86400000,(day+1)*86400000)).fetchone()[0]*5
            if loss+len(pending)*5+5>25:
                con.rollback();return False,'paper_daily_risk_budget'
            net=con.execute('SELECT coalesce(sum(net),0) FROM admissions WHERE outcome IS NOT NULL').fetchone()[0]
            if 1000+net-len(pending)*5<5:
                con.rollback();return False,'paper_capital_insufficient'
            con.execute('INSERT INTO admissions VALUES (?,?,?,?,?,NULL,NULL)',(signal['symbol'],signal['signal_ms'],entry,signal['expiry_due_ms'],signal['direction']))
            con.commit();return True,'paper_risk_admitted'

    def settle_admission(self,symbol,signal_ms,entry,expiry):
        with closing(sqlite3.connect(self.db,isolation_level=None)) as con:
            row=con.execute('SELECT direction FROM admissions WHERE symbol=? AND signal_ms=?',(symbol,signal_ms)).fetchone()
            if not row:return
            change=(expiry-entry)*row[0]
            outcome='win' if change>0 else 'loss' if change<0 else 'refund'
            net=4.25 if outcome=='win' else -5 if outcome=='loss' else 0
            con.execute('UPDATE admissions SET outcome=?,net=? WHERE symbol=? AND signal_ms=?',(outcome,net,symbol,signal_ms))

    def tick(self,now_ms=None):
        now=int(time.time()*1000) if now_ms is None else int(now_ms)
        if now-self.last_quote>=15000:
            try:
                quotes=self.market.quotes()
                with self.lock:self.state.update(symbols=quotes,data_health='fresh',last_error=None)
                self.last_quote=now
            except Exception:
                with self.lock:self.state.update(data_health='market_unavailable',last_error='公开指数行情暂不可用')
                self.last_quote=now-5000
        # Only the first two minutes of an hour may create its scheduled quote.
        # Restarting at 12:45 will not replay a stale 12:00 announcement.
        hour=now//HOUR*HOUR
        if now-hour<120000:
            quotes=self.snapshot()['symbols']
            if len(quotes)==2 and all(not quote['stale'] for quote in quotes.values()):
                self.notifier.send_price_report(quotes,hour)
        slot=now//HALF_HOUR*HALF_HOUR
        # A signal is only useful before its modeled entry at boundary+60s.
        if 5000<=now-slot<45000 and not self._processed(slot):
            try:
                from .features import build_latest_features
                spot,premium,funding=self.market.features_input(slot)
                rows=build_latest_features(spot,premium,funding,slot)
                decisions=self.predictor.predict(rows,slot)
                # Never emit old signals after a slow endpoint response.
                actual_now=int(time.time()*1000)
                late=actual_now>=slot+55000
                for symbol in SYMBOLS:self.history.record_features(symbol,slot,rows[symbol])
                last=None
                for signal in decisions:
                    signal['model_id']=self.predictor.manifest['model_version']
                    quote=self.state['symbols'].get(signal['symbol'],{})
                    if ('index_price' in quote and abs(actual_now-quote.get('price_time_ms',0))<=30000):
                        signal['index_price']=quote['index_price']
                    if signal['side']!='NONE' and not late:
                        admitted,reason=self.admit(signal)
                        signal['paper_risk_status']=reason
                        if admitted:
                            result=self.notifier.send_signal(signal)
                            last={**signal,'notification':result}
                with self.lock:
                    self.state.update(last_decisions=decisions,last_prediction_ms=slot,
                                      last_error='本次预测完成过晚，未推送过期信号' if late else None)
                    if last:self.state['last_signal']=last
                self._mark(slot,'late_skipped' if late else 'evaluated')
            except Exception as error:
                with self.lock:self.state['last_error']='本轮预测未完成：'+type(error).__name__
        if now-self.last_label_poll>=60000:
            self.last_label_poll=now
            for pending in self.history.pending_labels(now,limit=4):
                try:
                    entry=self.market.index_open(pending['symbol'],pending['entry_ms'])
                    expiry=self.market.index_open(pending['symbol'],pending['settlement_ms'])
                    self.history.update_label(pending['symbol'],pending['signal_ms'],entry,expiry)
                    self.settle_admission(pending['symbol'],pending['signal_ms'],entry,expiry)
                except Exception:break
            # Recover a paper settlement if its history label committed before a
            # transient failure writing the separate paper portfolio database.
            with closing(sqlite3.connect(self.db,isolation_level=None)) as con:
                pending=con.execute('SELECT symbol,signal_ms,entry_ms,settlement_ms FROM admissions WHERE outcome IS NULL AND settlement_ms+5000<=? LIMIT 2',(now,)).fetchall()
            for symbol,signal,entry_ms,expiry_ms in pending:
                try:
                    self.settle_admission(symbol,signal,self.market.index_open(symbol,entry_ms),self.market.index_open(symbol,expiry_ms))
                except Exception:break
        self.notifier.retry_pending(limit=3)
        self.maybe_refresh(now)

    def maybe_refresh(self,now):
        cutoff=int(self.predictor.manifest['training_cutoff_ms'])
        if now-cutoff<30*86400000 or now-self.last_refresh_attempt<6*HOUR:return
        # Keep fitting away from the minute in which an event signal is due.
        if now%HALF_HOUR<120000:return
        self.last_refresh_attempt=now
        try:
            from .predictor import retrain_from_rows
            day=now//86400000*86400000
            target=self.state_dir/'models'/f'refresh-{day}'
            retrain_from_rows(target,self.history.training_rows(day),day,progress=lambda _:None)
            replacement=Predictor(target)
            self.predictor=replacement;self.model_dir=target
            (self.state_dir/'active-model.txt').write_text(str(target),encoding='utf-8')
            self.notifier.send_status('研究模型已按最近180天已结算样本更新。',key=f'model-refresh-{day}')
        except Exception as error:
            with self.lock:self.state['last_error']='模型更新未完成：'+type(error).__name__

    def run(self):
        while not self.stop.is_set():
            try:self.tick()
            except Exception:
                with self.lock:self.state['last_error']='后台循环异常，正在重试'
            self.stop.wait(5)


def create_app(worker=None):
    app=Flask(__name__,static_folder=None)
    app.json.ensure_ascii=False
    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Cache-Control']='no-store'
        return response
    @app.get('/health')
    def health():
        return jsonify(ok=True,mode='research_alerts',model_loaded=worker is not None,order_execution_enabled=False)
    @app.get('/api/live/status')
    def status():
        if worker is None:return jsonify(mode='static',symbols={},notifications={'enabled':False}),503
        return jsonify(worker.snapshot())
    @app.get('/')
    @app.get('/index.html')
    def index():return send_file(ROOT/'index.html')
    @app.get('/demo/<name>')
    def demo(name):
        if name not in ('trades.html','quantdinger.html'):abort(404)
        return send_file(ROOT/'demo'/name)
    @app.get('/<name>')
    def asset(name):
        if name not in ('app.js','styles.css','research.js','workbuddy.js','trade-view.js'):abort(404)
        return send_file(ROOT/name)
    @app.get('/research/<name>')
    def report(name):
        if name not in ('trade-chart-data.json','workbuddy-results.json','workbuddy-trades.csv',
                        'event-results.json','event-trades.csv','results.json','derivative-study-results.json'):abort(404)
        return send_file(ROOT/'research'/name)
    return app


def read_env(path):
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if line and not line.startswith('#') and '=' in line:
            key,value=line.split('=',1);os.environ.setdefault(key.strip(),value.strip())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host',default='127.0.0.1');parser.add_argument('--port',type=int,default=8088)
    parser.add_argument('--no-worker',action='store_true');parser.add_argument('--env-file',type=Path)
    args=parser.parse_args()
    if args.env_file:read_env(args.env_file)
    state_dir=Path(os.environ.get('CODEXQUANT_STATE_DIR',str(ROOT/'live-state')))
    model_dir=Path(os.environ.get('CODEXQUANT_MODEL_DIR',str(ROOT/'.private'/'models')))
    active=state_dir/'active-model.txt'
    if active.exists():
        candidate=Path(active.read_text(encoding='utf-8').strip()).resolve()
        if candidate.is_relative_to(state_dir.resolve()) and candidate.is_dir():model_dir=candidate
    worker=None if args.no_worker else Worker(model_dir,state_dir,os.environ.get('WECOM_WEBHOOK_URL',''))
    if worker:threading.Thread(target=worker.run,name='market-worker',daemon=True).start()
    from waitress import serve
    serve(create_app(worker),host=args.host,port=args.port,threads=4)


if __name__=='__main__':main()
