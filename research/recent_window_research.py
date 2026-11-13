"""Local recent-window evaluation of existing rules, with chronological selection.

python -m research.recent_window_research
Uses audited cached rolling predictions only. No downloads, training, publishing,
notifications, live-model changes or orders. Prior history has already been used.
"""
from __future__ import annotations

import argparse
import calendar
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .consensus_search_research import calendar_mask, save_ledger
from .consensus_search_rules import candidates as consensus_candidates, directions as consensus_directions
from .event_engine import KNOWN_UNAVAILABLE_DATES, event_stats, utc_ms
from .high_win_research import BASE_CAPITAL, detail, file_hash, write_json
from .model_search_data import END as VERIFIED_END
from .model_search_research import choose_discovery, simulate_frame
from .model_search_rules import candidates as model_candidates, directions as model_directions

ROOT = Path(__file__).resolve().parents[1]
COMPARATORS = (
    'logistic_halfhour_0.56_exhaustion',
    'logistic_halfhour_0.64_exhaustion',
    'gbm_deeper_5m_0.60_exhaustion',
    'pair_logistic_halfhour_gbm_deeper_5m_0.60_unrestricted',
)
LABELS = {'year_to_date':'年初以来', 'recent_six_months':'近半年',
          'discovery':'年初筛选段', 'recent_first_half':'近半年前三个月',
          'recent_second_half':'近半年后三个月'}


def shift_months(day, months):
    number = day.year*12+day.month-1+months
    year, month = divmod(number, 12)
    return date(year, month+1, min(day.day, calendar.monthrange(year, month+1)[1]))


def periods(end=VERIFIED_END):
    if type(end) is not date or end > VERIFIED_END:
        raise ValueError('End must be a UTC date within the verified cache')
    first = date(end.year, 1, 1)
    recent = shift_months(end, -6)
    middle = shift_months(end, -3)
    if first < date(2026, 1, 1) or (recent-first).days < 90:
        raise ValueError('Require at least 90 days of same-year discovery before the recent six months')
    return {'discovery':(first,recent), 'year_to_date':(first,end),
            'recent_six_months':(recent,end),
            'recent_first_half':(recent,middle), 'recent_second_half':(middle,end)}


def candidate_rules():
    return model_candidates()+consensus_candidates()


def direction_for(frame, rule):
    fn = consensus_directions if rule['family']=='consensus' else model_directions
    signs = fn(frame,rule)
    signs[~calendar_mask(frame)] = 0
    return signs


def simulate_window(frame, rule, first, end, *, stress=None, direction=None):
    signs = direction_for(frame,rule) if direction is None else direction
    return simulate_frame(frame,signs,utc_ms(first),utc_ms(end),stress=stress,rule=rule)


def settlement_partition(ledger, first, end):
    """Disjoint diagnostics of one continuous simulation, retaining boundary trades."""
    low, high = utc_ms(first), utc_ms(end)
    subset = [t for t in ledger if low<=t['settlement_ms']<high]
    return event_stats(subset,low,high,payout=.85,fee=0.,bootstrap=False)


def acceptance(selected, half, partitions, stress):
    interval = half['weekly_block_interval']
    tests = {'selection_exists': selected is not None,
             'recent_min_200':half['decisive_trades']>=200,
             'recent_each_asset_min_50':all(s['decisive_trades']>=50 for s in half['by_symbol'].values()),
             'recent_58pct':(half['win_rate'] or 0)>=.58,
             'recent_positive':half['net_units']>0,
             'weekly_lower_above_break_even':(interval['win_rate_lower_95'] or 0)>1/1.85,
             'weekly_net_lower_positive':(interval['net_lower_95'] or 0)>0,
             'adverse_entry_2bps_positive':stress['net_units']>0}
    for name, stats in partitions.items():
        tests[name+'_min_75'] = stats['decisive_trades']>=75
        tests[name+'_58pct'] = (stats['win_rate'] or 0)>=.58
        tests[name+'_positive'] = stats['net_units']>0
    return tests


def metric_line(stats):
    rate='无交易' if stats['win_rate'] is None else f"{stats['win_rate']:.2%}"
    return f"{stats['signals']}笔 / 胜率{rate} / 净收益{stats['net_units']:+.2f} USDT / 最大回撤{stats['max_drawdown_units']:.2f} USDT"


def build_summary(report):
    p=report['plan']; selected=report['selection']['selected_rule']
    lines=['BTC/ETH 30分钟事件：年初以来与近半年本地研究',
           '所有日期为UTC，起点含、结束时刻不含。',
           '近半年 = 六个日历月；截至已校验数据边界，未虚构最新日期。']
    for name in ('year_to_date','recent_six_months','discovery'):
        window=p['periods'][name]
        lines.append(f"{LABELS[name]}：{window['start_utc']} 至 {window['end_exclusive_utc']}")
    lines += [f"76项已有固定规则（60项模型/直接反转，16项共识），没有新增参数网格。",
              '每笔5 USDT、初始1000、85%净赔付、零手续费、共享一笔持仓、每日亏损及待结算风险限额25。',
              '先按年初筛选段选择，再冻结；近半年结果和年初全年结果不会反过来改变选择。',
              '年初以来包含筛选期，且与近半年重叠，不能把两者视为两份独立验证。',
              '滚动模型仍使用此前180天已结算标签，每30天更新；回测展示起点不会截断必要训练历史。',
              '每个主区间独立从1000起步、无持仓重新模拟；两个三月段按半年连续账本的结算时刻归属，不重置风控。',
              '', '年初筛选段冻结候选：'+(selected['id'] if selected else '无合格候选；未用后续赢家替换。')]
    for key, value in report['details'].items():
        lines += ['',f"{key}"+('（本轮冻结候选）' if selected and selected['id']==key else '（预先指定对照）')]
        for name in ('discovery','year_to_date','recent_six_months'):
            lines.append(f"  {LABELS[name]}：{metric_line(value[name])}")
        for name,stats in value['recent_partitions'].items():
            lines.append(f"  {LABELS[name]}：{metric_line(stats)}")
        lines.append('  2bp不利开仓（重新判胜负及风控）：'+metric_line(value['recent_adverse_2bps']))
        block=value['recent_six_months']['weekly_block_interval']
        if block['net_lower_95'] is not None:
            lines.append(f"  半年7日块收益95%区间：{block['net_lower_95']:+.2f} 至 {block['net_upper_95']:+.2f} USDT")
    lines += ['', '后验描述榜（至少200笔，按近半年胜率排列；不是冻结入选策略）：']
    for key in report['descriptive_top_rule_ids']:
        row=next(r for r in report['matrix'] if r['rule']['id']==key)
        lines.append(f"  {key}：{metric_line(row['recent_six_months'])}；年初以来：{metric_line(row['year_to_date'])}")
    lines += ['',f"冻结候选通过本轮全部历史门槛：{report['historical_gates_passed']}",
              '这仍是复用历史的探索研究，不是全新留出集，也不是已证明稳定58%或未来保证盈利。',
              '分钟指数价代理真实合约开仓与到期报价；已剔除已知不可交易日期，但无法证明日历完整。',
              '按月及币种表现见results.json；76项全部结果见comparison.csv。',
              '本轮仅在本地生成新文件；没有修改旧报告、线上模型、GitHub、服务器或发送企业微信消息。']
    return '\n'.join(lines)+'\n'


def run(end=VERIFIED_END,output_dir=None):
    from .recent_window_data import load_recent_predictions
    began=time.perf_counter(); windows=periods(end); rules=candidate_rules()
    out=Path(output_dir or ROOT/'research'/'recent-window')
    out.mkdir(parents=True,exist_ok=True)
    # Preserve each completed or partial run, rather than replacing past evidence.
    if any(out.iterdir()):
        raise ValueError('Output directory already contains a run; choose a new --output-dir')
    source_names=['recent_window_research.py','recent_window_data.py','model_search_research.py',
                  'model_search_rules.py','consensus_search_rules.py','consensus_search_research.py',
                  'high_win_research.py','event_engine.py','event_strategy_research.py']
    plan={'version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),
          'periods':{name:{'start_utc':str(first),'end_exclusive_utc':str(stop)} for name,(first,stop) in windows.items()},
          'cache_end_exclusive_utc':str(VERIFIED_END),'candidate_count':len(rules),'rules':rules,
          'comparator_ids':list(COMPARATORS),'capital':dict(BASE_CAPITAL),
          'selection':{'period':'discovery','min_decisive':200,'min_each_asset':50,'min_win_rate':.58,
                       'net_positive':True,'ranking':'Bonferroni Wilson lower, then lexical rule ID'},
          'validation':{'period':'recent_six_months','min_decisive':200,'min_each_asset':50,
                        'each_three_month_min':75,'each_three_month_win_rate':.58,
                        'weekly_net_lower_positive':True,'adverse_2bps_positive':True},
          'known_unavailable_dates_utc':list(KNOWN_UNAVAILABLE_DATES),
          'train_days':180,'refit_days':30,'no_new_training':True,
          'primary_windows_restart_capital':True,'quarter_diagnostics':'partition continuous half-year by settlement time',
          'year_to_date_contains_discovery':True,'windows_overlap':True,
          'historical_data_previously_researched':True,'pristine_holdout':False,'forward_validated':False,
          'local_only':True,'publish_or_deploy':False,
          'source_sha256':{name:file_hash(ROOT/'research'/name) for name in source_names}}
    write_json(out/'plan.json',plan)
    print('PLAN FROZEN '+str(out/'plan.json'),flush=True)
    first=windows['year_to_date'][0]
    frame,provenance=load_recent_predictions(first,end)
    print(f'READY {len(frame):,} minute/asset rows; no fitting',flush=True)
    lo,stop=windows['discovery']; discovery=frame[(frame.signal_ms>=utc_ms(lo))&(frame.settlement_ms<utc_ms(stop))].copy()
    records=[]
    for index,rule in enumerate(rules,1):
        _,stats=simulate_window(discovery,rule,lo,stop)
        records.append({'rule':rule,'discovery':stats})
        if index%10==0 or index==len(rules):print(f'DISCOVERY {index}/{len(rules)}',flush=True)
    chosen,ranking=choose_discovery(records,min_trades=200,min_per_symbol=50)
    selection={'selected_rule':chosen,'ranking':ranking,'selection_cutoff_ms':utc_ms(stop),
               'locked_at_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':file_hash(out/'plan.json')}
    write_json(out/'selection.json',selection)
    print('LOCKED '+(chosen['id'] if chosen else 'NO ELIGIBLE CANDIDATE'),flush=True)
    del discovery
    detailed_ids=set(COMPARATORS)|({chosen['id']} if chosen else set())
    matrix=[]; details={}; csv_rows=[]
    for index,record in enumerate(records,1):
        rule=record['rule']; measured=dict(record); signs=direction_for(frame,rule); ledgers={}
        for period in ('year_to_date','recent_six_months'):
            low,high=windows[period]
            ledger,stats=simulate_window(frame,rule,low,high,direction=signs)
            measured[period]=stats; ledgers[period]=ledger
        partitions={name:settlement_partition(ledgers['recent_six_months'],*windows[name])
                    for name in ('recent_first_half','recent_second_half')}
        measured.update(partitions);matrix.append(measured)
        if rule['id'] in detailed_ids:
            item={}
            # Retain complete audit ledgers for the locked candidate and fixed comparators.
            disc,_=simulate_window(frame,rule,*windows['discovery'],direction=signs)
            ledgers['discovery']=disc
            for name,ledger in ledgers.items():
                low,high=windows[name];item[name]=detail(ledger,utc_ms(low),utc_ms(high))
                filename=f"{rule['id']}-{name}-trades.csv"
                save_ledger(out/filename,ledger)
                item[name]['ledger']=filename;item[name]['ledger_sha256']=file_hash(out/filename)
            low,high=windows['recent_six_months']
            stressed,stats=simulate_window(frame,rule,low,high,stress={'adverse_bps':2.},direction=signs)
            item['recent_adverse_2bps']=stats
            filename=f"{rule['id']}-recent-adverse2bps-trades.csv"
            save_ledger(out/filename,stressed);item['recent_adverse_2bps']['ledger']=filename
            item['recent_adverse_2bps']['ledger_sha256']=file_hash(out/filename)
            item['recent_partitions']=partitions
            details[rule['id']]=item
            print(rule['id']+' HALF '+metric_line(item['recent_six_months']),flush=True)
        for name in windows:
            csv_rows.append({'rule_id':rule['id'],'period':name,**plan['periods'][name],**measured[name]})
        if index%10==0 or index==len(rules):print(f'RECENT {index}/{len(rules)}',flush=True)
    eligible=[r for r in matrix if r['recent_six_months']['decisive_trades']>=200]
    top=sorted(eligible,key=lambda r:(-(r['recent_six_months']['win_rate'] or 0),r['rule']['id']))[:5]
    checks={}
    if chosen:
        row=details[chosen['id']]
        checks=acceptance(chosen,row['recent_six_months'],row['recent_partitions'],row['recent_adverse_2bps'])
    else:checks={'selection_exists':False}
    report={'plan':plan,'selection':selection,'provenance':provenance,'matrix':matrix,'details':details,
            'descriptive_top_rule_ids':[r['rule']['id'] for r in top],
            'validation_checks':checks,'historical_gates_passed':all(checks.values()),
            'forward_validated':False,'local_only':True,'publish_or_deploy':False,
            'generated_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':round(time.perf_counter()-began,2)}
    write_json(out/'results.json',report)
    pd.DataFrame(csv_rows).to_csv(out/'comparison.csv',index=False,encoding='utf-8-sig')
    (out/'summary.txt').write_text(build_summary(report),encoding='utf-8')
    print('DONE '+json.dumps({'seconds':report['elapsed_seconds'],'selected':chosen['id'] if chosen else None,
          'historical_gates_passed':report['historical_gates_passed'],'report':str(out/'summary.txt')}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--end',type=date.fromisoformat,default=VERIFIED_END)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args();run(args.end,args.output_dir)
