"""Deployable read-only dashboard and scheduled BTC/ETH research alerts."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime,timezone
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import threading
import time
from urllib.parse import urlsplit

from flask import Flask,abort,jsonify,send_file

from .market import BinanceMarket,SYMBOLS
from .notify import WeComNotifier
from .predictor import Predictor
from .stakes import PaperSettings,amount_cents,settlement_cents,PAYOUT_BPS

ROOT=Path(__file__).resolve().parents[1]
MINUTE=60000
HALF_HOUR=1800000
HOUR=3600000
LEGACY_RULE='both_regularized_logistic'
CONSENSUS_RULE='pair_logistic_1m_gbm_deeper_5m_0.60_unrestricted'


def strategy_mode(environ=None):
    value=(os.environ if environ is None else environ).get('CODEXQUANT_STRATEGY','legacy')
    if value not in ('legacy','consensus'):raise ValueError('Unsupported configured strategy')
    return value


def model_pointer(state_dir,mode):
    return Path(state_dir)/('active-consensus-model.txt' if mode=='consensus' else 'active-model.txt')


class Worker:
    def __init__(self,model_dir,state_dir,webhook='',market=None,paper_settings=None):
        from .history import FeatureHistory
        self.paper_settings=paper_settings or PaperSettings.from_env()
        hold_path=os.environ.get('CODEXQUANT_PUBLICATION_HOLD_FILE')
        self.publication_hold_file=Path(hold_path) if hold_path else None
        self.model_dir=Path(model_dir)
        self.state_dir=Path(state_dir);self.state_dir.mkdir(parents=True,exist_ok=True)
        self.strategy_mode=strategy_mode()
        if self.strategy_mode=='consensus':
            from .consensus import ConsensusPredictor
            self.predictor=ConsensusPredictor(model_dir)
        else:self.predictor=Predictor(model_dir)
        history_name='history-consensus.sqlite3' if self.strategy_mode=='consensus' else 'history.sqlite3'
        self.history=FeatureHistory(self.state_dir/history_name)
        self.history.import_seed(self.model_dir,self.predictor.manifest)
        self.notifier=WeComNotifier(webhook,self.state_dir/'notifications.sqlite3')
        self.market=market or BinanceMarket()
        self.lock=threading.RLock()
        self.db=self.state_dir/'worker.sqlite3'
        self._ensure_tables()
        self.state={'mode':'research_alerts','symbols':{},'last_signal':None,'last_decisions':[],
                    'last_error':None,'prediction_error':None,'settlement_error':None,'last_prediction_ms':None,'data_health':'starting',
                    'scan_interval_seconds':60,
                    'prediction_schedule':'每分钟收盘后约5秒检查机会；下一分钟模拟入场，持有30分钟',
                    'price_report_schedule':'北京时间每小时整点，24小时',
                    'research_report_schedule':'北京时间每小时整点汇报扫描和前向模拟；到期分钟扫描完成、核实指数后复盘',
                    'order_execution_enabled':False,'forward_validated':False}
        self.last_quote=0;self.last_label_poll=0;self.last_refresh_attempt=0
        self.refresh_thread=None
        self.maintenance_thread=None
        self.history_status_cache=None;self.history_status_at=0
        self.stop=threading.Event()

    def _ensure_tables(self):
        with closing(sqlite3.connect(self.db,isolation_level=None)) as con:
            con.executescript('''
                CREATE TABLE IF NOT EXISTS processed (slot INTEGER PRIMARY KEY,state TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS admissions (symbol TEXT,signal_ms INTEGER,entry_ms INTEGER,
                    settlement_ms INTEGER,direction INTEGER,outcome TEXT,net REAL,PRIMARY KEY(symbol,signal_ms));
                CREATE TABLE IF NOT EXISTS scan_decisions (symbol TEXT,signal_ms INTEGER,reason TEXT,
                    candidate INTEGER,admitted INTEGER,risk_status TEXT,PRIMARY KEY(symbol,signal_ms));
                CREATE TABLE IF NOT EXISTS settlement_outbox (symbol TEXT,signal_ms INTEGER,
                    payload TEXT NOT NULL,queued INTEGER NOT NULL DEFAULT 0,PRIMARY KEY(symbol,signal_ms));
                CREATE INDEX IF NOT EXISTS scan_decision_time ON scan_decisions(signal_ms);
                CREATE INDEX IF NOT EXISTS admission_settlement ON admissions(settlement_ms,outcome);
                CREATE INDEX IF NOT EXISTS outbox_time ON settlement_outbox(signal_ms DESC);
                CREATE INDEX IF NOT EXISTS outbox_pending ON settlement_outbox(queued,signal_ms);
            ''')
            # Keep the original seven-column table intact. Terms live beside
            # each admission, so old rows and queued notifications retain 5U.
            con.execute('BEGIN IMMEDIATE')
            con.execute('''CREATE TABLE IF NOT EXISTS admission_terms (
                symbol TEXT NOT NULL,signal_ms INTEGER NOT NULL,
                stake_cents INTEGER NOT NULL DEFAULT 500,
                payout_bps INTEGER NOT NULL DEFAULT 8500,
                PRIMARY KEY(symbol,signal_ms))''')
            con.execute('''INSERT OR IGNORE INTO admission_terms(symbol,signal_ms,stake_cents,payout_bps)
                SELECT symbol,signal_ms,500,8500 FROM admissions''')
            con.execute('''CREATE TABLE IF NOT EXISTS admission_strategy (
                symbol TEXT NOT NULL,signal_ms INTEGER NOT NULL,rule_id TEXT NOT NULL,
                model_id TEXT NOT NULL,PRIMARY KEY(symbol,signal_ms))''')
            # Records predating this migration belong to the old rule, even
            # when their stake was already changed to 30U.
            con.execute('''INSERT OR IGNORE INTO admission_strategy(symbol,signal_ms,rule_id,model_id)
                SELECT symbol,signal_ms,?,'legacy_untracked' FROM admissions''',(LEGACY_RULE,))
            con.commit()

    def strategy_description(self):
        predictor=getattr(self,'predictor',None)
        manifest=getattr(predictor,'manifest',{})
        if not isinstance(manifest,dict):manifest={}
        rule=manifest.get('rule_id',CONSENSUS_RULE if getattr(self,'strategy_mode','legacy')=='consensus' else LEGACY_RULE)
        consensus=rule==CONSENSUS_RULE
        return {'rule_id':rule,'model_id':manifest.get('model_version','legacy_untracked'),
                'mode':'consensus' if consensus else 'legacy',
                'label':'Logistic + GBM 双模型同向确认' if consensus else '原单模型反转策略',
                'threshold':.60 if consensus else .56,
                'members':['logistic_1m','gbm_deeper_5m'] if consensus else ['regularized_logistic'],
                'training_sampling_minutes':1 if consensus else 30,
                'train_days':180,'refit_days':30,'forward_validated':False,
                'description':'两模型方向相同且各自方向分数至少0.60；每分钟扫描' if consensus else '单模型分数与极端偏离反转过滤'}

    def paper_config(self):
        if not hasattr(self,'paper_settings'):
            self.paper_settings=PaperSettings.from_env()
        return self.paper_settings

    def publication_held(self):
        path=getattr(self,'publication_hold_file',None)
        return path is not None and path.exists()

    def snapshot(self):
        with self.lock:
            value=json.loads(json.dumps(self.state))
        now=int(time.time()*1000)
        for quote in value['symbols'].values():quote['stale']=now-quote['price_time_ms']>30000
        predictor=self.predictor
        value['model']={'id':predictor.manifest['model_version'],
                        'trained_at':predictor.manifest['training_cutoff_utc'],
                        'rule_id':predictor.manifest['rule_id'],'max_age_days':35}
        value['strategy']=self.strategy_description()
        value['notifications']=self.notifier.status()
        value['scan_stats']=self.scan_stats(now)
        value['paper_performance']=self.paper_performance()
        value['publication_hold']=self.publication_held()
        historical=self.historical_research()
        value['historical_research']={'available':historical is not None,**(historical or {})}
        value['recent_settlements']=self.recent_settlements()
        with self.lock:
            if self.history_status_cache is None or now-self.history_status_at>=MINUTE:
                self.history_status_cache=self.history.status();self.history_status_at=now
            value['feature_history']=self.history_status_cache
        return value

    def scan_stats(self,now):
        start=now-HOUR
        with closing(sqlite3.connect(self.db,isolation_level=None)) as con:
            states=dict(con.execute('SELECT state,count(*) FROM processed WHERE slot>=? AND slot<? GROUP BY state',(start,now)))
            decisions=con.execute('SELECT reason,candidate,admitted,risk_status FROM scan_decisions WHERE signal_ms>=? AND signal_ms<?',(start,now)).fetchall()
        reasons={};blocked={}
        for reason,candidate,admitted,risk in decisions:
            reasons[reason]=reasons.get(reason,0)+1
            if candidate and not admitted and risk:blocked[risk]=blocked.get(risk,0)+1
        return {'window_start_ms':start,'window_end_ms':now,'evaluated_slots':states.get('evaluated',0),
                'late_slots':states.get('late_skipped',0),'decisions':len(decisions),
                'candidates':sum(row[1] for row in decisions),'admitted':sum(row[2] for row in decisions),
                'reasons':reasons,'risk_blocks':blocked}

    def paper_performance(self):
        with closing(sqlite3.connect(self.db,isolation_level=None)) as con:
            rows=con.execute('''SELECT a.outcome,a.net,coalesce(t.stake_cents,500),
                coalesce(s.rule_id,?),coalesce(s.model_id,'legacy_untracked')
                FROM admissions a LEFT JOIN admission_terms t USING(symbol,signal_ms)
                LEFT JOIN admission_strategy s USING(symbol,signal_ms)''',(LEGACY_RULE,)).fetchall()
        counts={};net_cents=0;amounts={}
        for outcome,net,stake,rule,model in rows:
            counts[outcome]=counts.get(outcome,0)+1
            net_cents+=amount_cents(net or 0)
            amounts[stake]=amounts.get(stake,0)+1
        config=self.paper_config().public();net=net_cents/100
        wins=counts.get('win',0);losses=counts.get('loss',0);refunds=counts.get('refund',0)
        current=self.strategy_description()
        def cohort(selected):
            outcomes={};total_net=0
            for outcome,net,stake,rule,model in selected:
                outcomes[outcome]=outcomes.get(outcome,0)+1;total_net+=amount_cents(net or 0)
            w,l,r=(outcomes.get(key,0) for key in ('win','loss','refund'))
            return {'total':len(selected),'settled':w+l+r,'pending':outcomes.get(None,0),
                    'wins':w,'losses':l,'refunds':r,'win_rate':w/(w+l) if w+l else None,
                    'net_usdt':total_net/100}
        rules=sorted({row[3] for row in rows})
        current_rows=[row for row in rows if row[3]==current['rule_id']]
        return {'total':sum(counts.values()),'settled':wins+losses+refunds,'pending':counts.get(None,0),
                'wins':wins,'losses':losses,'refunds':refunds,'win_rate':wins/(wins+losses) if wins+losses else None,
                'net_usdt':net,'balance_usdt':config['initial_capital_usdt']+net,
                **config,'mixed_stakes':len(amounts)>1,
                'stake_history':[{'stake_usdt':stake/100,'trades':count} for stake,count in sorted(amounts.items())],
                'mixed_strategies':len(rules)>1,'all_history_is_current_strategy':all(rule==current['rule_id'] for rule in rules),
                'all_history':cohort(rows),
                'current_strategy':{**current,**cohort(current_rows),'source':'current_strategy_forward_paper_only'},
                'strategy_history':[{'rule_id':rule,**cohort([row for row in rows if row[3]==rule])} for rule in rules],
                'source':'forward_paper_simulation'}

    def recent_settlements(self):
        with closing(sqlite3.connect(self.db,isolation_level=None)) as con:
            rows=con.execute('SELECT payload FROM settlement_outbox ORDER BY signal_ms DESC,symbol LIMIT 10').fetchall()
        return [json.loads(row[0]) for row in rows]

    def historical_research(self):
        """Small, explicit historical disclosure; absence never blocks scanning."""
        now=time.monotonic()
        if now-getattr(self,'historical_research_loaded_at',float('-inf'))<60:
            return getattr(self,'historical_research_cache',None)
        value=None
        try:
            path=ROOT/'research'/'consensus-public.json'
            three_year=path.is_file()
            if not three_year:path=ROOT/'research'/'recent-public.json'
            if path.stat().st_size>16*1024*1024:raise ValueError('Oversized research payload')
            data=json.loads(path.read_text(encoding='utf-8'))
            a=data['assumptions'];strategy=data['strategy']
            stats=data['stats'] if three_year else data['published']['recent_six_months']['stats']
            period=data['period'] if three_year else next(p for p in data['periods'] if p['id']=='recent_six_months')
            if (data['schema_version']!=1 or a['default_stake']!=30 or a['initial_capital']!=10000
                    or a['net_payout']!=.85 or a['fee']!=0
                    or strategy['is_live_strategy'] is not three_year or strategy['posthoc'] is not True
                    or strategy['forward_validated'] is not False):
                raise ValueError('Unsupported recent research provenance')
            if three_year and (strategy['rule_id']!=CONSENSUS_RULE or
                    period.get('start_utc')!='2023-11-12' or period.get('end_exclusive_utc')!='2026-11-12'):
                raise ValueError('Unsupported three-year rule or interval')
            rate=stats['win_rate'];net=stats['net_usdt']
            if (not isinstance(stats['signals'],int) or stats['signals']<=0
                    or not math.isfinite(rate) or not 0<=rate<=1 or not math.isfinite(net)):
                raise ValueError('Invalid recent research statistics')
            base=os.environ.get('CODEXQUANT_PUBLIC_BASE_URL','http://38.76.182.201').rstrip('/')
            url=urlsplit(base)
            if (url.scheme not in ('http','https') or not url.hostname or url.username or url.password
                    or url.query or url.fragment or any(c in base for c in '[]()\r\n')):
                raise ValueError('Invalid public research URL')
            value={'strategy':strategy['label'],'rule_id':strategy['rule_id'],
                   'start_utc':period['start_utc'],'end_exclusive_utc':period['end_exclusive_utc'],
                   'signals':stats['signals'],'win_rate':rate,'net_usdt':net,
                   'stake_usdt':a['default_stake'],'initial_capital_usdt':a['initial_capital'],
                   'posthoc':True,'is_live_strategy':three_year and self.strategy_description()['rule_id']==CONSENSUS_RULE,
                   'forward_validated':False,'scope':'three_year' if three_year else 'recent_six_months',
                   'url':base+('/demo/consensus.html' if three_year else '/demo/recent.html')}
            if three_year:
                value.update(max_drawdown_usdt=stats['max_drawdown_usdt'],
                             yearly=data.get('yearly',[]),stress=data.get('stress',[]))
            value['matches_current_strategy']=value['is_live_strategy']
        except (OSError,ValueError,KeyError,TypeError,StopIteration):
            value=None
        self.historical_research_loaded_at=now
        self.historical_research_cache=value
        return value

    def _record_decisions(self,decisions):
        with closing(sqlite3.connect(self.db,isolation_level=None)) as con:
            con.executemany('INSERT OR IGNORE INTO scan_decisions VALUES (?,?,?,?,?,?)',[
                (row['symbol'],row['signal_ms'],row.get('reason','signal'),int(row['side']!='NONE'),
                 int(row.get('paper_risk_status')=='paper_risk_admitted'),row.get('paper_risk_status')) for row in decisions])

    def _processed(self,slot):
        with closing(sqlite3.connect(self.db,isolation_level=None)) as con:
            return con.execute('SELECT state FROM processed WHERE slot=?',(slot,)).fetchone() is not None

    def _mark(self,slot,state):
        with closing(sqlite3.connect(self.db,isolation_level=None)) as con:con.execute('INSERT OR IGNORE INTO processed VALUES (?,?)',(slot,state))

    def admit(self,signal):
        """Causal paper budget: reserve all as-yet-unsettled loss risk at alert time."""
        entry=int(signal['entry_due_ms']);day=entry//86400000
        if self.publication_held():return False,'research_publication_hold'
        settings=self.paper_config();stake=settings.stake_cents
        with closing(sqlite3.connect(self.db,isolation_level=None)) as con:
            con.execute('BEGIN IMMEDIATE')
            if con.execute('SELECT 1 FROM admissions WHERE symbol=? AND signal_ms=?',(signal['symbol'],signal['signal_ms'])).fetchone():
                con.rollback();return False,'duplicate_paper_signal'
            pending=con.execute('''SELECT a.settlement_ms,coalesce(t.stake_cents,500)
                FROM admissions a LEFT JOIN admission_terms t USING(symbol,signal_ms)
                WHERE a.outcome IS NULL''').fetchall()
            if any(row[0]>entry for row in pending):
                con.rollback();return False,'paper_position_open'
            reserved=sum(row[1] for row in pending)
            loss=con.execute('''SELECT coalesce(sum(coalesce(t.stake_cents,500)),0)
                FROM admissions a LEFT JOIN admission_terms t USING(symbol,signal_ms)
                WHERE a.outcome='loss' AND a.settlement_ms>=? AND a.settlement_ms<?''',
                (day*86400000,(day+1)*86400000)).fetchone()[0]
            if loss+reserved+stake>settings.daily_limit_cents:
                con.rollback();return False,'paper_daily_risk_budget'
            net=sum(amount_cents(row[0]) for row in con.execute('SELECT net FROM admissions WHERE outcome IS NOT NULL'))
            if settings.initial_capital_cents+net-reserved<stake:
                con.rollback();return False,'paper_capital_insufficient'
            con.execute('INSERT INTO admissions VALUES (?,?,?,?,?,NULL,NULL)',(signal['symbol'],signal['signal_ms'],entry,signal['expiry_due_ms'],signal['direction']))
            con.execute('INSERT INTO admission_terms(symbol,signal_ms,stake_cents,payout_bps) VALUES(?,?,?,?)',
                        (signal['symbol'],signal['signal_ms'],stake,PAYOUT_BPS))
            strategy=self.strategy_description()
            rule=signal.get('rule_id',strategy['rule_id']);model=signal.get('model_id',strategy['model_id'])
            if rule!=strategy['rule_id'] or not isinstance(model,str) or not model:
                raise ValueError('Admission strategy does not match its predictor')
            con.execute('INSERT INTO admission_strategy(symbol,signal_ms,rule_id,model_id) VALUES(?,?,?,?)',
                        (signal['symbol'],signal['signal_ms'],rule,model))
            signal.update(stake_usdt=stake/100,payout=.85,win_net_usdt=settlement_cents(stake,'win')/100,
                          loss_net_usdt=-stake/100,daily_loss_limit_usdt=settings.daily_limit_cents/100,
                          rule_id=rule,model_id=model)
            con.commit();return True,'paper_risk_admitted'

    def settle_admission(self,symbol,signal_ms,entry,expiry):
        if any(isinstance(value,bool) or not math.isfinite(float(value)) or float(value)<=0 for value in (entry,expiry)):
            raise ValueError('Settlement requires positive finite index observations')
        entry,expiry=float(entry),float(expiry)
        with closing(sqlite3.connect(self.db,isolation_level=None)) as con:
            con.execute('BEGIN IMMEDIATE')
            row=con.execute('''SELECT a.direction,a.entry_ms,a.settlement_ms,a.outcome,a.net,
                coalesce(t.stake_cents,500),coalesce(t.payout_bps,8500),
                coalesce(s.rule_id,?),coalesce(s.model_id,'legacy_untracked')
                FROM admissions a LEFT JOIN admission_terms t USING(symbol,signal_ms)
                LEFT JOIN admission_strategy s USING(symbol,signal_ms)
                WHERE a.symbol=? AND a.signal_ms=?''',(LEGACY_RULE,symbol,signal_ms)).fetchone()
            if not row:return
            change=(expiry-entry)*row[0]
            outcome='win' if change>0 else 'loss' if change<0 else 'refund'
            net=settlement_cents(row[5],outcome,row[6])/100
            if row[3] is not None and (row[3]!=outcome or row[4]!=net):
                raise ValueError('Conflicting paper settlement; original result preserved')
            payload={'symbol':symbol,'signal_ms':signal_ms,'entry_ms':row[1],'settlement_ms':row[2],
                     'direction':row[0],'entry_price':entry,'exit_price':expiry,'outcome':outcome,'net_usdt':net,
                     'stake_usdt':row[5]/100,'payout':row[6]/10000,'rule_id':row[7],'model_id':row[8]}
            old=con.execute('SELECT payload FROM settlement_outbox WHERE symbol=? AND signal_ms=?',(symbol,signal_ms)).fetchone()
            if old:
                previous=json.loads(old[0])
                normalized={**previous,'stake_usdt':previous.get('stake_usdt',5),'payout':previous.get('payout',.85),
                            'rule_id':previous.get('rule_id',LEGACY_RULE),'model_id':previous.get('model_id','legacy_untracked')}
                if normalized!=payload:raise ValueError('Conflicting paper execution prices')
            con.execute('UPDATE admissions SET outcome=?,net=? WHERE symbol=? AND signal_ms=?',(outcome,net,symbol,signal_ms))
            con.execute('INSERT OR IGNORE INTO settlement_outbox(symbol,signal_ms,payload) VALUES(?,?,?)',
                        (symbol,signal_ms,json.dumps(payload,ensure_ascii=False,allow_nan=False)))
            con.commit()

    def _settle_due(self,now):
        # Admissions take priority over the larger training population. This also
        # backfills outcomes committed by the old version before reports existed.
        with closing(sqlite3.connect(self.db,isolation_level=None)) as con:
            pending=con.execute('''SELECT a.symbol,a.signal_ms,a.entry_ms,a.settlement_ms FROM admissions a
                LEFT JOIN settlement_outbox o ON a.symbol=o.symbol AND a.signal_ms=o.signal_ms
                WHERE (a.outcome IS NULL OR o.symbol IS NULL) AND a.settlement_ms+5000<=?
                ORDER BY a.settlement_ms LIMIT 4''',(now,)).fetchall()
        errors=[]
        for symbol,signal,entry_ms,expiry_ms in pending:
            try:
                entry=self.market.index_open(symbol,entry_ms);expiry=self.market.index_open(symbol,expiry_ms)
                self.settle_admission(symbol,signal,entry,expiry)
            except Exception as error:errors.append(type(error).__name__)
        for pending in self.history.pending_labels(now,limit=8):
            try:
                entry=self.market.index_open(pending['symbol'],pending['entry_ms'])
                expiry=self.market.index_open(pending['symbol'],pending['settlement_ms'])
                self.history.update_label(pending['symbol'],pending['signal_ms'],entry,expiry)
            except Exception as error:
                errors.append(type(error).__name__);break
        with self.lock:self.state['settlement_error']='模拟结算/标签等待行情：'+errors[0] if errors else None

    def _flush_settlements(self):
        with closing(sqlite3.connect(self.db,isolation_level=None)) as con:
            pending=con.execute('SELECT symbol,signal_ms,payload FROM settlement_outbox WHERE queued=0 ORDER BY signal_ms LIMIT 4').fetchall()
        for symbol,signal,payload in pending:
            result=self.notifier.send_settlement(json.loads(payload))
            if result.get('state')!='disabled':
                with closing(sqlite3.connect(self.db,isolation_level=None)) as con:
                    con.execute('UPDATE settlement_outbox SET queued=1 WHERE symbol=? AND signal_ms=?',(symbol,signal))

    def tick(self,now_ms=None):
        now=int(time.time()*1000) if now_ms is None else int(now_ms)
        if now-self.last_quote>=15000:
            try:
                quotes=self.market.quotes()
                with self.lock:self.state.update(symbols=quotes,data_health='fresh')
                self.last_quote=now
            except Exception:
                with self.lock:self.state.update(data_health='market_unavailable',last_error='公开指数行情暂不可用')
                self.last_quote=now-5000
        slot=now//MINUTE*MINUTE
        # A signal is only useful before its modeled entry at boundary+60s.
        if 5000<=now-slot<45000 and not self._processed(slot):
            try:
                from .features import build_latest_features
                spot,premium,funding=self.market.features_input(slot)
                rows=build_latest_features(spot,premium,funding,slot)
                predictor=self.predictor
                decisions=predictor.predict(rows,slot)
                # Never emit old signals after a slow endpoint response.
                actual_now=int(time.time()*1000)
                late=actual_now>=slot+55000
                for symbol in SYMBOLS:self.history.record_features(symbol,slot,rows[symbol])
                last=None
                for signal in decisions:
                    signal['model_id']=predictor.manifest['model_version']
                    signal['rule_id']=predictor.manifest.get('rule_id',self.strategy_description()['rule_id'])
                    quote=self.state['symbols'].get(signal['symbol'],{})
                    if ('index_price' in quote and abs(actual_now-quote.get('price_time_ms',0))<=30000):
                        signal['index_price']=quote['index_price']
                    if signal['side']!='NONE' and not late:
                        admitted,reason=self.admit(signal)
                        signal['paper_risk_status']=reason
                        if admitted:
                            result=self.notifier.send_signal(signal)
                            last={**signal,'notification':result}
                    elif signal['side']!='NONE':signal['paper_risk_status']='late_signal_skipped'
                self._record_decisions(decisions)
                with self.lock:
                    self.state.update(last_decisions=decisions,last_prediction_ms=slot,
                                      prediction_error='本次预测完成过晚，未推送过期信号' if late else None,
                                      last_error='本次预测完成过晚，未推送过期信号' if late else None)
                    if last:self.state['last_signal']=last
                self._mark(slot,'late_skipped' if late else 'evaluated')
            except Exception as error:
                with self.lock:
                    self.state['prediction_error']='本轮预测未完成：'+type(error).__name__
                    self.state['last_error']=self.state['prediction_error']
        self.start_maintenance(now)
        self.maybe_refresh(now)

    def start_maintenance(self,now):
        # One slow index/webhook request must never occupy the scan loop.
        if getattr(self,'maintenance_thread',None) is not None and self.maintenance_thread.is_alive():return
        self.maintenance_thread=threading.Thread(target=self._maintenance,args=(now,),name='report-maintenance',daemon=True)
        self.maintenance_thread.start()

    def _maintenance(self,now):
        try:
            self._maintenance_work(now)
            with self.lock:self.state['maintenance_error']=None
        except Exception as error:
            with self.lock:self.state['maintenance_error']='播报维护异常：'+type(error).__name__

    def _maintenance_work(self,now):
        # Report before historical label catch-up, which may be slow after an
        # outage. Expiration prevents reporting stale scheduled hourly messages.
        hour=now//HOUR*HOUR
        if now-hour<120000:
            with self.lock:
                quotes=json.loads(json.dumps(self.state['symbols']))
                prediction_error=self.state.get('prediction_error')
            if len(quotes)==2 and all(abs(now-quote['price_time_ms'])<=30000 for quote in quotes.values()):
                self.notifier.send_price_report(quotes,hour)
            self.notifier.send_research_report({'scan_stats':self.scan_stats(hour),
                'paper_performance':self.paper_performance(),'prediction_error':prediction_error,
                'historical_research':self.historical_research()},hour)
        self.notifier.retry_pending(limit=3)
        slot=now//MINUTE*MINUTE
        # Preserve the simulator's expiry < signal rule during admission. Once
        # this minute's scan is durably complete, its expiring result can settle
        # so it is known to the following minute's scan. A maintenance job from
        # an earlier minute carries its original cutoff and cannot see ahead.
        label_cutoff=now if self._processed(slot) else slot-1
        if label_cutoff//MINUTE>self.last_label_poll//MINUTE:
            self.last_label_poll=label_cutoff
            self._settle_due(label_cutoff)
        self._flush_settlements()

    def maybe_refresh(self,now):
        cutoff=int(self.predictor.manifest['training_cutoff_ms'])
        if now-cutoff<30*86400000 or now-self.last_refresh_attempt<6*HOUR:return
        if getattr(self,'refresh_thread',None) is not None and self.refresh_thread.is_alive():return
        self.last_refresh_attempt=now
        # Fitting runs independently now that every minute is a prediction minute.
        self.refresh_thread=threading.Thread(target=self._refresh_model,args=(now,),name='model-refresh',daemon=True)
        self.refresh_thread.start()

    def _refresh_model(self,now):
        try:
            mode=getattr(self,'strategy_mode','legacy')
            if mode=='consensus':
                from .consensus import retrain_from_rows,ConsensusPredictor
                predictor_class=ConsensusPredictor;sampling=1
            else:
                from .predictor import retrain_from_rows
                predictor_class=Predictor;sampling=30
            day=now//86400000*86400000
            target=self.state_dir/('models-consensus' if mode=='consensus' else 'models')/f'refresh-{day}'
            retrain_from_rows(target,self.history.training_rows(day,sample_interval_minutes=sampling),day,progress=lambda _:None)
            replacement=predictor_class(target)
            destination=model_pointer(self.state_dir,mode);pointer=destination.with_suffix('.tmp')
            pointer.write_text(str(target),encoding='utf-8')
            pointer.replace(destination)
            with self.lock:
                self.predictor=replacement;self.model_dir=target
                self.state['model_refresh_error']=None
            label='双模型共识' if mode=='consensus' else '原单模型'
            self.notifier.send_status(f'{label}研究模型已按最近180天已结算样本更新。',key=f'model-refresh-{mode}-{day}')
        except Exception as error:
            with self.lock:
                self.state['model_refresh_error']='模型更新未完成：'+type(error).__name__
                self.state['last_error']=self.state['model_refresh_error']

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
        if name not in ('trades.html','quantdinger.html','research.html','recent.html','consensus.html'):abort(404)
        return send_file(ROOT/'demo'/name)
    @app.get('/<name>')
    def asset(name):
        if name not in ('app.js','styles.css','research.js','workbuddy.js','trade-view.js',
                        'research-view.js','research-view.css','recent-view.js','recent-view.css',
                        'consensus-view.js','consensus-view.css'):abort(404)
        return send_file(ROOT/name)
    @app.get('/research/<name>')
    def report(name):
        if name not in ('trade-chart-data.json','workbuddy-results.json','workbuddy-trades.csv',
                        'event-results.json','event-trades.csv','results.json','derivative-study-results.json',
                        'minute-scan-results.json','high-win-public.json','high-win-public-trades.csv',
                        'recent-public.json','recent-public-trades.csv','consensus-public.json','consensus-public-trades.csv'):abort(404)
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
    active=model_pointer(state_dir,strategy_mode())
    if active.exists():
        candidate=Path(active.read_text(encoding='utf-8').strip()).resolve()
        if candidate.is_relative_to(state_dir.resolve()) and candidate.is_dir():model_dir=candidate
    worker=None if args.no_worker else Worker(model_dir,state_dir,os.environ.get('WECOM_WEBHOOK_URL',''))
    if worker:threading.Thread(target=worker.run,name='market-worker',daemon=True).start()
    from waitress import serve
    serve(create_app(worker),host=args.host,port=args.port,threads=4)


if __name__=='__main__':main()
