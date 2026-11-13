"""Publish an explicit, compact projection of the completed local research.

No training, API requests, notification or live-model change occurs here.
Research provenance files remain immutable; only public projections are built.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RULE = 'gbm_deeper_5m_0.60_exhaustion'
PERIODS = (
    ('validation', '长期验证', '2025-05-07', '2026-08-09'),
    ('historical_confirmation', '近期检查', '2026-08-09', '2026-11-12'),
    ('continuous_post_selection', '连续后续期间', '2025-05-07', '2026-11-12'),
)
STATS = ('signals','wins','losses','refunds','decisive_trades','win_rate','net_units','max_drawdown_units')


def milliseconds(value):
    return int(datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()*1000)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(trades):
    wins = sum(t['outcome']=='win' for t in trades)
    losses = sum(t['outcome']=='loss' for t in trades)
    refunds = len(trades)-wins-losses
    net = Decimal(0); peak = Decimal(0); drawdown = Decimal(0)
    for trade in sorted(trades,key=lambda t:(t['settlement_ms'],t['symbol'])):
        net += Decimal(str(trade['net_units']))
        peak = max(peak,net); drawdown = max(drawdown,peak-net)
    return {'signals':len(trades),'wins':wins,'losses':losses,'refunds':refunds,
            'decisive_trades':wins+losses,'win_rate':wins/(wins+losses) if wins+losses else None,
            'net_units':float(net),'max_drawdown_units':float(drawdown)}


def read_verified_trades(path, expected_sha, first, stop, unavailable_dates):
    if sha(path)!=expected_sha:
        raise ValueError('Audited ledger hash differs from the calendar report')
    rows = list(csv.DictReader(io.StringIO(Path(path).read_text(encoding='utf-8-sig'))))
    trades=[]; seen=set(); previous_expiry=None
    forbidden=[milliseconds(day) for day in unavailable_dates]
    for row in rows:
        symbol=row['symbol']; side=int(row['direction'])
        signal,entry_ms,expiry=(int(row[k]) for k in ('signal_ms','entry_ms','settlement_ms'))
        key=(symbol,signal)
        if symbol not in ('BTCUSDT','ETHUSDT') or side not in (-1,1) or key in seen:
            raise ValueError('Invalid or duplicate trade identity')
        seen.add(key)
        if (signal%60000 or entry_ms!=signal+60000 or expiry!=entry_ms+1800000
                or signal<first or expiry>=stop or int(row['horizon_minutes'])!=30):
            raise ValueError('Trade outside the frozen event clock or evaluation window')
        if any(signal<day+86400000 and expiry>=day for day in forbidden):
            raise ValueError('Trade touches an unavailable date')
        if not int(row['train_latest_settlement_ms'])<int(row['model_fit_cutoff_ms'])<=signal:
            raise ValueError('Noncausal model provenance')
        if previous_expiry is not None and entry_ms<previous_expiry:
            raise ValueError('Ledger violates shared one-position capital')
        previous_expiry=expiry
        opened,closed=(Decimal(row[k]) for k in ('entry','exit'))
        if not opened.is_finite() or not closed.is_finite() or min(opened,closed)<=0:
            raise ValueError('Invalid index prices')
        change=(closed-opened)*side
        outcome='win' if change>0 else 'loss' if change<0 else 'refund'
        net=Decimal('4.25') if change>0 else Decimal('-5') if change<0 else Decimal(0)
        if row['outcome']!=outcome or Decimal(row['net_units'])!=net or Decimal(row['stake'])!=5:
            raise ValueError('Outcome or 85% zero-fee payoff does not reconcile')
        trades.append({'id':len(trades)+1,'symbol':symbol,'direction':side,
            'signal_ms':signal,'entry_ms':entry_ms,'settlement_ms':expiry,
            'entry':float(opened),'exit':float(closed),'entry_exact':row['entry'],'exit_exact':row['exit'],
            'outcome':outcome,'net_units':float(net)})
    return trades


def build(root=ROOT):
    root=Path(root); folder=root/'research'
    calendar=json.loads((folder/'calendar-audit-results.json').read_text(encoding='utf-8'))
    register=json.loads((folder/'high-win-research-register.json').read_text(encoding='utf-8'))
    if (register['completed_rounds']!=8 or register['candidate_configurations']!=137
            or register['matched_control_configurations']!=24 or register['historical_gate_passed_rounds']!=0):
        raise ValueError('Research status changed; explicitly review the public projection')
    source_rows={r['period']:r for r in calendar['rows'] if r['rule_id']==RULE}
    source=source_rows['continuous_post_selection']
    ledger=(root/source['ledger'].replace('\\','/')).resolve()
    if not ledger.is_relative_to(folder.resolve()):
        raise ValueError('Ledger must belong to the research directory')
    trades=read_verified_trades(ledger,source['ledger_sha256'],milliseconds('2025-05-07'),
                               milliseconds('2026-11-12'),calendar['known_unavailable_dates_utc'])
    periods=[]
    for key,label,start,end in PERIODS:
        first,stop=milliseconds(start),milliseconds(end)
        measured=stats([t for t in trades if t['signal_ms']>=first and t['settlement_ms']<stop])
        audited=source_rows[key]['after_calendar_gate']
        if any(measured[k]!=audited[k] for k in STATS):
            raise ValueError('Public period statistics differ from the audited ledger')
        periods.append({'id':key,'label':label,'start_utc':start,'end_exclusive_utc':end,
                        'start_ms':first,'end_exclusive_ms':stop,**measured})
    if len(trades)!=1262:
        raise ValueError('Unexpected corrected GBM cohort')
    study_rows=[]
    for study in register['studies']:
        path=folder/study['file']
        if sha(path)!=study['sha256']:
            raise ValueError('A registered research report changed')
        study_rows.append({'label':study['label'],'candidates':study['candidate_configurations'],
                          'controls':study['matched_control_configurations'],
                          'historical_gates_passed':bool(study['historical_gates_passed'])})
    interval=source['after_calendar_gate']['weekly_block_interval']
    payload={'schema_version':1,'published_data_version':'2026-11-13-research-v1',
        'research':{'rounds':8,'candidate_configurations':137,'matched_controls':24,
                    'qualified_strategies':0,'forward_validated':False,'studies':study_rows},
        'comparison':{'rule_id':RULE,'label':'GBM 极端偏离反转 · 探索性对照',
                      'is_live_strategy':False,'calendar_corrected':True,'periods':periods,
                      'weekly_interval':{k:interval[k] for k in ('block_days','win_rate_lower_95','win_rate_upper_95','net_lower_95','net_upper_95')},
                      'by_symbol':{symbol:stats([t for t in trades if t['symbol']==symbol]) for symbol in ('BTCUSDT','ETHUSDT')}},
        'assumptions':{'symbols':['BTCUSDT','ETHUSDT'],'horizon_minutes':30,'entry_delay_minutes':1,
                       'net_payout':.85,'fee':0,'stake':5,'initial_capital':1000,'daily_loss_limit':25,
                       'max_open_positions':1,'break_even_win_rate':1/1.85,
                       'known_unavailable_dates_utc':calendar['known_unavailable_dates_utc']},
        'limitations':['该 GBM 是历史研究对照，并非服务器当前运行策略；未通过稳定 58% 的完整研究门槛。',
            '137 项配置及 24 项对照包含重复组合；历史已反复研究，不能当作未查看的新测试集。',
            '7 日块重采样的收益区间包含亏损；历史盈利不等于已经证实可持续盈利。',
            '分钟指数开盘价代替真实合约行权与到期报价；价格偏差会改变二元胜负。',
            '新增持仓数据采用记录后延迟 10 分钟的研究假设，不能证明历史 API 的真实可见时间。'],
        'trades':trades}
    public_json=folder/'high-win-public.json'
    public_json.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')
    public_csv=folder/'high-win-public-trades.csv'
    with public_csv.open('w',encoding='utf-8',newline='') as stream:
        fields=['id','symbol','direction','signal_ms','entry_ms','settlement_ms','entry_exact','exit_exact','outcome','net_units']
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(trades)
    print(json.dumps({'trades':len(trades),'json_bytes':public_json.stat().st_size,'csv_bytes':public_csv.stat().st_size,
                      'win_rate':periods[-1]['win_rate'],'net_units':periods[-1]['net_units']}))
    return payload


if __name__=='__main__':
    build()
