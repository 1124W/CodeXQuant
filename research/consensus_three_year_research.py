"""Frozen two-model consensus over three full years; no reselection or publication.

python -m research.consensus_three_year_research --plan-only
python -m research.consensus_three_year_research
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .consensus_search_rules import candidates, directions
from .consensus_search_research import calendar_mask
from .event_engine import KNOWN_UNAVAILABLE_DATES, utc_ms
from .event_strategy_research import moving_week_interval
from .high_win_research import file_hash, write_json
from .model_search_models import specs
from .public_stake_engine import ledger_stats
from .stake_sizing_engine import simulate_stakes

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research'/'consensus-three-year-2026-11-13'
START=date(2023,11,12)
END=date(2026,11,12)
JOIN=date(2024,5,7)
RULE_ID='pair_logistic_1m_gbm_deeper_5m_0.60_unrestricted'
RULE=next(rule for rule in candidates() if rule['id']==RULE_ID)
POLICY={'id':'fixed_30u','kind':'fixed','amount':30}
SOURCES=('research/consensus_search_rules.py','research/consensus_search_research.py',
         'research/model_search_models.py','research/model_search_data.py',
         'research/stake_sizing_engine.py','research/public_stake_engine.py',
         'research/event_engine.py','research/event_strategy_research.py','live/features.py')


def make_plan():
    return {'schema_version':1,'start_utc':str(START),'end_exclusive_utc':str(END),
            'reason_for_cutoff':'same verified data boundary as the published recent comparison',
            'rule':RULE,'candidate_count':1,'selection_performed':False,
            'models':{name:specs()[name] for name in RULE['members']},
            'train_days':180,'refit_days':30,'preserve_original_fold_clock':True,
            'early_prediction_extension_end_exclusive_utc':str(JOIN),
            'stake_usdt':30,'initial_capital_usdt':10000,'daily_risk_limit_usdt':150,
            'net_payout':.85,'fee':0,'max_open_positions':1,'entry_delay_minutes':1,
            'horizon_minutes':30,'known_unavailable_dates_utc':list(KNOWN_UNAVAILABLE_DATES),
            'stress_adverse_entry_bps':[0,.5,1,2],
            'main_account':'continuous across all three years; no yearly balance or risk reset',
            'partition_policy':'attribute continuous ledger by settlement time; carry opening equity',
            'recent_reference':'independent 10000U restart only to reconcile the already published comparison',
            'historical_selection_bias_remains':True,'pristine_holdout':False,
            'forward_validated':False,'local_only':True,'publish_or_deploy':False,
            'frozen_source_sha256':{name:file_hash(ROOT/name) for name in SOURCES}}


def freeze_plan(folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/'plan.json';plan=make_plan()
    if path.exists():
        saved=json.loads(path.read_text(encoding='utf-8'))
        if {k:v for k,v in saved.items() if k!='frozen_at_utc'}!=plan:
            raise ValueError('The frozen plan changed; do not overwrite it')
        return saved
    plan['frozen_at_utc']=datetime.now(timezone.utc).isoformat()
    write_json(path,plan)
    return plan


def normalize_opportunities(frame, first, end):
    """Recompute the one fixed rule, retaining model-clock witnesses."""
    required={'symbol','signal_ms','entry_ms','settlement_ms','entry','exit',
              'model_fit_cutoff_ms','train_latest_settlement_ms',
              'p_logistic_1m','p_gbm_deeper_5m'}
    if required-set(frame):raise ValueError('Missing verified consensus observations')
    if frame.duplicated(['symbol','signal_ms']).any():raise ValueError('Duplicate source observations')
    score=frame[['p_logistic_1m','p_gbm_deeper_5m']].to_numpy(dtype=float)
    if not np.isfinite(score).all() or np.any(score<0) or np.any(score>1):
        raise ValueError('Invalid model score')
    clock=frame[['signal_ms','model_fit_cutoff_ms','train_latest_settlement_ms']].to_numpy(dtype=float)
    if (not np.isfinite(clock).all() or np.any(clock!=np.floor(clock))
            or np.any(clock[:,2]>=clock[:,1]) or np.any(clock[:,1]>clock[:,0])):
        raise ValueError('Noncausal fit witness')
    sign=directions(frame,RULE);sign[~calendar_mask(frame)]=0
    if 'direction' in frame and not np.array_equal(frame.direction.to_numpy(),sign):
        raise ValueError('Cached opportunity direction differs from the frozen rule')
    keep=(sign!=0)&(frame.signal_ms>=first)&(frame.signal_ms<end)
    result=frame.loc[keep,sorted(required)].copy();result['direction']=sign[keep]
    result['rule_id']=RULE_ID
    return result.sort_values(['signal_ms','symbol']).reset_index(drop=True)


def combine_opportunities(early,existing):
    border=utc_ms(JOIN)
    if (early.signal_ms>=border).any() or (existing.signal_ms<border).any():
        raise ValueError('Early and existing prediction source ranges overlap')
    combined=pd.concat([early,existing],ignore_index=True)
    return normalize_opportunities(combined,utc_ms(START),utc_ms(END))


def replay(frame,first=START,end=END,bps=0):
    return simulate_stakes(frame,frame.direction.to_numpy(),utc_ms(first),utc_ms(end),
                           POLICY,adverse_bps=bps,initial_bankroll=10000)


def contribution_stats(ledger,initial,first_ms):
    stats=ledger_stats(ledger,initial,first_ms)
    streak=longest=0
    for trade in sorted(ledger,key=lambda t:(t['settlement_ms'],t['symbol'])):
        streak=streak+1 if trade['outcome']=='loss' else 0
        longest=max(longest,streak)
    stats['longest_losing_streak']=longest
    stats['losing_streak_refund_policy']='reset'
    return stats


def partition(ledger,first,end,initial=10000):
    """Carry prior settled equity; never simulate a fresh portfolio here."""
    low,high=utc_ms(first),utc_ms(end)
    before=sum(round(t['net_units']*100) for t in ledger if t['settlement_ms']<low)
    opening=(round(initial*100)+before)/100
    if opening<=0:raise ValueError('Partition opening equity is exhausted')
    subset=[t for t in ledger if low<=t['settlement_ms']<high]
    return {'start_utc':str(first),'end_exclusive_utc':str(end),
            'opening_balance':opening,'source':'partition_of_continuous_ledger',
            **contribution_stats(subset,opening,low)}


def calendar_partitions(ledger,unit):
    cursor=START;rows=[]
    while cursor<END:
        if unit=='year':following=date(cursor.year+1,1,1)
        elif unit=='month':following=date(cursor.year+1,1,1) if cursor.month==12 else date(cursor.year,cursor.month+1,1)
        else:raise ValueError('Unsupported calendar partition')
        stop=min(following,END)
        rows.append({'label':str(cursor.year) if unit=='year' else cursor.strftime('%Y-%m'),
                     **partition(ledger,cursor,stop)})
        cursor=stop
    return rows


def reconcile_recent(frame):
    published=json.loads((ROOT/'research/recent-public.json').read_text(encoding='utf-8'))
    if published['strategy']['rule_id']!=RULE_ID:raise ValueError('Published rule changed')
    rows=[]
    for window in published['periods']:
        first,end=map(date.fromisoformat,(window['start_utc'],window['end_exclusive_utc']))
        ledger,stats=replay(frame,first,end)
        reference=published['published'][window['id']]
        keys=('symbol','signal_ms','entry_ms','settlement_ms','entry','exit','direction',
              'stake','outcome','net_units','balance_after_settlement')
        if (len(ledger)!=len(reference['trades']) or
                any(any(a[key]!=b[key] for key in keys) for a,b in zip(ledger,reference['trades']))):
            raise ValueError('Extended sources changed a published recent trade')
        for key in ('signals','wins','losses','refunds','net_usdt','win_rate','max_drawdown_usdt'):
            if stats[key]!=reference['stats'][key]:raise ValueError('Published statistic changed: '+key)
        rows.append({'period':window['id'],'per_trade_match':True,
                     **{key:stats[key] for key in ('signals','win_rate','net_usdt')}})
    return rows


def metrics(stats):
    count=stats['decisive_trades']
    return {**stats,'return_on_initial':stats['net_usdt']/stats['initial_bankroll'],
            'expectancy_per_decisive_trade':stats['net_usdt']/count if count else None,
            'break_even_win_rate':1/1.85}


def line(item):
    rate='无样本' if item['win_rate'] is None else f"{item['win_rate']:.2%}"
    return f"{item['signals']}笔，胜率{rate}，净收益{item['net_usdt']:+.2f}U，最大回撤{item['max_drawdown_usdt']:.2f}U"


def summary(report):
    lines=['Logistic + GBM 双模型同向确认：完整三年固定规则回测',
           f'{START} 00:00 至 {END} 00:00 UTC（右端不含）',
           '固定同向、各自方向分数至少0.60；无额外市场状态过滤。没有重新挑选或调整参数。',
           '每笔30U，初始10,000U，85%净赔付，零额外手续费，每日已亏损及待结算风险150U，共享最多一笔持仓。',
           '信号后一分钟开仓，30分钟后结算；只使用当时已完成行情与已结算的训练标签。',
           '主账本三年连续，跨年不重置资金或持仓。',
           '', '三年总计：'+line(report['total']),
           f"期末资金：{report['total']['ending_balance']:.2f}U；最长连亏：{report['total']['longest_losing_streak']}笔。",
           '', '连续账本分三个完整年度：']
    for row in report['anniversary_years']:
        lines.append(f"{row['start_utc']} → {row['end_exclusive_utc']}：{line(row)}")
    lines+=['','按币种拆分已准入交易的贡献（未各分配新本金）：']
    for symbol,row in report['by_symbol'].items():lines.append(symbol+'：'+line(row))
    lines+=['','不利开仓报价压力（重新判胜负及执行风控）：']
    for row in report['stress']:lines.append(f"{row['adverse_bps']:g}bp：{line(row['stats'])}")
    interval=report['weekly_interval']
    if interval['net_lower_95'] is not None:
        lines += ['',f"7日块重采样95%区间：胜率{interval['win_rate_lower_95']:.2%}—{interval['win_rate_upper_95']:.2%}；净收益{interval['net_lower_95']:+.2f}—{interval['net_upper_95']:+.2f}U。"]
    lines+=['','诊断检查（全部披露，不根据结果换规则）：']
    lines.extend(f'{name}: {value}' for name,value in report['diagnostics'].items())
    lines+=['','已逐笔核对原半年338笔与年初443笔，结果保持一致。',
            '本轮更早数据拓展只是历史稳健性检查；策略最初由2026年表现择优发现，三年结果仍不是独立前瞻验证。',
            '开仓与结算用币安指数分钟开盘价代理，不能证明真实事件合约报价、历史可交易日历或实际成交完全一致。',
            '无效特征分钟和缺失溢价窗口被排除，未插值填充；覆盖明细见results.json中的provenance。',
            '本轮只保存本地研究报告与逐笔账本，没有修改GitHub、服务器、线上模型或企业微信推送。']
    return '\n'.join(lines)+'\n'


def run(folder=OUT):
    began=time.perf_counter();folder=Path(folder);plan=freeze_plan(folder)
    if (folder/'results.json').exists():raise ValueError('Completed research already exists; preserve it')
    from .consensus_three_year_existing import load_existing_opportunities
    from .consensus_three_year_extension import load_verified_extension
    print('LOAD verified historical sources; frozen single rule',flush=True)
    early,early_proof=load_verified_extension()
    # The extension exposes complete minute predictions; require joint availability
    # before looking for either coin's eligible model agreement.
    if early.duplicated(['symbol','signal_ms']).any():raise ValueError('Duplicate early predictions')
    joint=early.groupby('signal_ms').symbol.transform('nunique')==2
    early=normalize_opportunities(early.loc[joint].copy(),utc_ms(START),utc_ms(JOIN))
    existing,existing_proof=load_existing_opportunities()
    frame=combine_opportunities(early,existing)
    reconciliation=reconcile_recent(frame)
    print(f'READY {len(frame)} eligible opportunities; recent ledgers reconcile',flush=True)
    ledger,total=replay(frame);total=metrics(total)
    annual=[partition(ledger,date(y,11,12),date(y+1,11,12)) for y in (2023,2024,2025)]
    months=calendar_partitions(ledger,'month');years=calendar_partitions(ledger,'year')
    by_symbol={symbol:contribution_stats([t for t in ledger if t['symbol']==symbol],10000,utc_ms(START))
               for symbol in ('BTCUSDT','ETHUSDT')}
    interval=moving_week_interval(ledger,utc_ms(START),utc_ms(END))
    stress=[]
    for bps in (.5,1,2):
        stressed,stats=replay(frame,bps=bps)
        name=f'adverse-{bps:g}bps-trades.csv';pd.DataFrame(stressed).to_csv(folder/name,index=False)
        stress.append({'adverse_bps':bps,'stats':metrics(stats),'ledger':name,'ledger_sha256':file_hash(folder/name),
                       'weekly_interval':moving_week_interval(stressed,utc_ms(START),utc_ms(END))})
        print(f'STRESS {bps:g}bp '+line(stats),flush=True)
    checks={'total_58pct':(total['win_rate'] or 0)>=.58,'total_positive':total['net_usdt']>0,
            'each_full_year_58pct':all((row['win_rate'] or 0)>=.58 for row in annual),
            'each_full_year_positive':all(row['net_usdt']>0 for row in annual),
            'each_asset_positive':all(row['net_usdt']>0 for row in by_symbol.values()),
            'weekly_lower_above_break_even':(interval['win_rate_lower_95'] or 0)>1/1.85,
            'weekly_net_lower_positive':(interval['net_lower_95'] or 0)>0,
            'adverse_1bp_positive':stress[1]['stats']['net_usdt']>0,
            'adverse_2bp_positive':stress[2]['stats']['net_usdt']>0}
    assert sum(row['signals'] for row in annual)==len(ledger)
    assert sum(round(row['net_usdt']*100) for row in months)==round(total['net_usdt']*100)
    if {name:file_hash(ROOT/name) for name in SOURCES}!=plan['frozen_source_sha256']:
        raise ValueError('Frozen implementation changed while running')
    for name,rows in [('trades.csv',ledger),('monthly.csv',months),('calendar-years.csv',years),
                      ('full-years.csv',annual),('equity.csv',total['equity_curve'])]:
        pd.DataFrame(rows).to_csv(folder/name,index=False,encoding='utf-8-sig')
    frame.to_csv(folder/'opportunities.csv',index=False,encoding='utf-8-sig')
    report={'plan':plan,'plan_sha256':file_hash(folder/'plan.json'),'total':total,
            'anniversary_years':annual,'calendar_years':years,'monthly':months,'by_symbol':by_symbol,
            'weekly_interval':interval,'stress':stress,'diagnostics':checks,
            'recent_reference_reconciliation':reconciliation,
            'provenance':{'early':early_proof,'existing':existing_proof},
            'new_source_sha256':{p.name:file_hash(p) for p in (ROOT/'research').glob('consensus_three_year_*.py')},
            'artifact_sha256':{p.name:file_hash(p) for p in folder.glob('*.csv')},
            'local_only':True,'publish_or_deploy':False,'forward_validated':False,
            'generated_at_utc':datetime.now(timezone.utc).isoformat(),'runtime_seconds':round(time.perf_counter()-began,2)}
    write_json(folder/'results.json',report)
    (folder/'summary.txt').write_text(summary(report),encoding='utf-8')
    print('DONE '+line(total),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only',action='store_true');parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    if args.plan_only:
        freeze_plan(args.output_dir);print('PLAN FROZEN '+str(args.output_dir/'plan.json'))
    else:run(args.output_dir)
