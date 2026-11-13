"""Publish a bounded, reproducible projection of the frozen three-year study."""
from __future__ import annotations
import csv
from datetime import date
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import pandas as pd
from research.event_engine import KNOWN_UNAVAILABLE_DATES,utc_ms
from research.consensus_three_year_research import RULE_ID,normalize_opportunities
from research.public_stake_engine import simulate_opportunities
from scripts.build_recent_public import project_opportunities,verify_replay,canonical_hash

FOLDER='research/consensus-three-year-2026-11-13'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(root=ROOT):
    root=Path(root);folder=root/FOLDER
    report=json.loads((folder/'results.json').read_text(encoding='utf-8'))
    plan=report['plan']
    if (plan['rule']['id']!=RULE_ID or plan['candidate_count']!=1 or plan['selection_performed']
            or plan['stake_usdt']!=30 or plan['initial_capital_usdt']!=10000 or report['forward_validated']):
        raise ValueError('Three-year study contract changed')
    if digest(folder/'plan.json')!=report['plan_sha256']:raise ValueError('Frozen plan checksum changed')
    for name,expected in report['artifact_sha256'].items():
        if Path(name).name!=name or digest(folder/name)!=expected:raise ValueError('Research artifact changed: '+name)
    first,end=utc_ms(date.fromisoformat(plan['start_utc'])),utc_ms(date.fromisoformat(plan['end_exclusive_utc']))
    frame=pd.read_csv(folder/'opportunities.csv',float_precision='round_trip')
    frame=normalize_opportunities(frame,first,end)
    raw=project_opportunities(frame,frame.direction.to_numpy())
    for item,source in zip(raw,frame.to_dict('records')):
        item.update({name:float(source[name]) for name in ('p_logistic_1m','p_gbm_deeper_5m')})
        item.update({name:int(source[name]) for name in ('model_fit_cutoff_ms','train_latest_settlement_ms')})
    ledger,stats=verify_replay(frame,frame.direction.to_numpy(),raw,first,end,30,10000)
    compare=('signals','wins','losses','refunds','win_rate','net_usdt','max_drawdown_usdt','ending_balance')
    if any(stats[k]!=report['total'][k] for k in compare):raise ValueError('Published baseline differs from frozen report')
    original=pd.read_csv(folder/'trades.csv',float_precision='round_trip').to_dict('records')
    keys=('symbol','signal_ms','entry_ms','settlement_ms','direction','entry','exit','outcome','stake','net_units','balance_after_settlement')
    if len(ledger)!=len(original) or any(any(a[k]!=b[k] for k in keys) for a,b in zip(ledger,original)):
        raise ValueError('Frozen report trade sequence changed')
    stress=[]
    for witness in report['stress']:
        bps=witness['adverse_bps']
        _,test=verify_replay(frame,frame.direction.to_numpy(),raw,first,end,30,10000,bps)
        if any(test[k]!=witness['stats'][k] for k in compare):raise ValueError('Price stress differs from frozen report')
        stress.append({'adverse_bps':bps,'stats':test,'weekly_interval':witness['weekly_interval']})
    amounts=[]
    for amount in (5,5.01,30,249.99,250):
        _,test=verify_replay(frame,frame.direction.to_numpy(),raw,first,end,amount,10000)
        amounts.append({'stake':amount,**{k:test[k] for k in compare}})
    period={'id':'three_years','label':'完整三年','start_utc':plan['start_utc'],
            'end_exclusive_utc':plan['end_exclusive_utc'],'start_ms':first,'end_exclusive_ms':end}
    annual=[{'year':int(row['label']),**row} for row in report['calendar_years']]
    monthly=[{'month':row['label'],**row} for row in report['monthly']]
    assumptions={'default_stake':30,'initial_capital':10000,'min_stake':5,'max_stake':250,'stake_step':.01,
                 'daily_risk_mode':'stake_multiple','daily_risk_multiplier':5,'daily_risk_limit':150,
                 'net_payout':.85,'fee':0,'horizon_minutes':30,'entry_delay_minutes':1,'max_open_positions':1,
                 'payout_rounding':'half_up_cent','unsettled_principal_reserved':True,
                 'known_unavailable_dates_utc':list(KNOWN_UNAVAILABLE_DATES),'continuous_account':True}
    public={'schema_version':1,'published_data_version':'2026-11-13-consensus-three-years-30u-v1',
            'strategy':{'rule_id':RULE_ID,'label':'Logistic + GBM 双模型同向确认',
                        'description':'Logistic 1分钟训练模型与 GBM 5分钟训练模型同方向，双方方向分数各自至少 0.60；每分钟扫描，分数不等于胜率。',
                        'is_live_strategy':True,'posthoc':True,'forward_validated':False,
                        'activation_status':'selected_for_forward_research'},
            'assumptions':assumptions,'period':period,'periods':[period],'stats':stats,'yearly':annual,
            'anniversary_years':report['anniversary_years'],'stress':stress,'opportunities':raw,
            'published':{'three_years':{'stats':stats,'trades':ledger,'monthly':monthly,
                        'by_symbol':report['by_symbol'],'weekly_interval':report['weekly_interval'],'stress':stress}},
            'diagnostics':report['diagnostics'],
            'limitations':['策略最初由近期历史结果择优发现；三年扩展没有重新挑选，但仍不是独立前瞻验证。',
                '三年胜率未达到58%，2024自然年为负收益，2bp不利开仓报价情景同样亏损。',
                '已选定用于前向研究提醒，实际启用状态见服务器；没有自动下单。',
                '本页10,000U为历史研究起始本金；服务器旧账户与策略批次统计独立留存。',
                '金额设置只影响当前浏览器的回测，不更改群机器人金额；日风险预算始终为单笔五倍。',
                '本金与持仓在三年内连续，不因跨年重置；分年表现按到期时刻归属。',
                '指数分钟开盘价代理实际事件合约报价；价格微小偏差也可能改变二元胜负。',
                '特征缺失窗口与已知不可交易日被剔除，未插值；历史可交易日历可能不完整。'],
            'provenance':{'report_sha256':digest(folder/'results.json'),'opportunities_sha256':canonical_hash(raw),
                'numeric_replay_checks':amounts,'source_sha256':{name:digest(root/name) for name in
                  ('scripts/build_consensus_public.py','research/public_stake_engine.py','research/stake_sizing_engine.py')},
                'training_or_download_performed':False}}
    path=root/'research/consensus-public.json'
    content=json.dumps(public,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
    if len(content.encode())>16*1024*1024:raise ValueError('Public payload exceeds server limit')
    path.write_text(content,encoding='utf-8')
    with (root/'research/consensus-public-trades.csv').open('w',encoding='utf-8',newline='') as out:
        fields=['id','opportunity_id','symbol','direction','signal_ms','entry_ms','settlement_ms',
                'entry_exact','exit_exact','p_logistic_1m','p_gbm_deeper_5m','outcome','stake','net_units',
                'balance_after_settlement','model_fit_cutoff_ms','train_latest_settlement_ms']
        writer=csv.DictWriter(out,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(ledger)
    print(json.dumps({'opportunities':len(raw),'trades':len(ledger),'bytes':path.stat().st_size,
                      'win_rate':stats['win_rate'],'net_usdt':stats['net_usdt']}))
    return public


if __name__=='__main__':build()
