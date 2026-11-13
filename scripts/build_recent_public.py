"""Build the public 30-USDT recent-window research projection and replay inputs.

Original reports and cached predictions are read-only. Published projections
contain no model files, paths, environment, credentials or live settings.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from research.event_engine import KNOWN_UNAVAILABLE_DATES, utc_ms
from research.event_strategy_research import moving_week_interval
from research.high_win_research import file_hash
from research.public_stake_engine import ledger_stats, simulate_opportunities, validate_opportunities
from research.recent_candidate_audit import verify_base
from research.recent_window_data import load_recent_predictions
from research.recent_window_research import direction_for
from research.stake_sizing_engine import simulate_stakes

RULE_ID='pair_logistic_1m_gbm_deeper_5m_0.60_unrestricted'
BASE_FOLDER='research/recent-window-2026-11-12'
DIAGNOSTIC_FOLDER='research/recent-window-2026-11-12-diagnostics'
PERIODS=(('year_to_date','年初以来','2026-01-01','2026-11-12'),
         ('recent_six_months','近半年','2026-05-12','2026-11-12'))
REFERENCE_FIELDS=('signals','wins','losses','refunds','decisive_trades','win_rate',
                  'net_usdt','ending_balance','max_drawdown_usdt','max_drawdown_pct',
                  'max_drawdown_initial_pct','minimum_balance','longest_losing_streak',
                  'raw_opportunities','skipped_overlap','skipped_daily_limit','skipped_insufficient')


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def project_opportunities(frame, signs):
    selected=frame.loc[signs!=0].copy()
    selected['direction']=signs[signs!=0]
    selected=selected.sort_values(['signal_ms','symbol'])
    rows=[]
    for item in selected.to_dict('records'):
        opened,closed=float(item['entry']),float(item['exit'])
        sign=int(item['direction']);change=(closed-opened)*sign
        rows.append({'id':len(rows)+1,'symbol':str(item['symbol']),'direction':sign,
                     **{k:int(item[k]) for k in ('signal_ms','entry_ms','settlement_ms')},
                     'entry':opened,'exit':closed,'entry_exact':repr(opened),'exit_exact':repr(closed),
                     'outcome':'win' if change>0 else 'loss' if change<0 else 'refund'})
    validate_opportunities(rows)
    return rows


def partitions(trades,first,end,initial):
    by_symbol={symbol:ledger_stats([t for t in trades if t['symbol']==symbol],initial,first)
               for symbol in ('BTCUSDT','ETHUSDT')}
    months=[];cursor=datetime.fromtimestamp(first/1000,timezone.utc).replace(day=1)
    while int(cursor.timestamp()*1000)<end:
        month=cursor.strftime('%Y-%m')
        subset=[t for t in trades if datetime.fromtimestamp(t['settlement_ms']/1000,timezone.utc).strftime('%Y-%m')==month]
        months.append({'month':month,**ledger_stats(subset,initial,first)})
        cursor=cursor.replace(year=cursor.year+1,month=1) if cursor.month==12 else cursor.replace(month=cursor.month+1)
    return months,by_symbol


def verify_replay(frame,signs,opportunities,first,end,stake,initial,bps=0):
    projected,stats=simulate_opportunities(opportunities,first,end,stake=stake,
                                           initial_capital=initial,adverse_bps=bps)
    reference,ref_stats=simulate_stakes(frame,signs,first,end,
        {'id':'publication-reference','kind':'fixed','amount':stake},adverse_bps=bps,initial_bankroll=initial)
    if any(stats[key]!=ref_stats[key] for key in REFERENCE_FIELDS):
        raise ValueError('Public replay statistics differ from the audited sizing engine')
    keys=('symbol','signal_ms','entry_ms','settlement_ms','direction','entry','exit','outcome',
          'stake','net_units','equity_at_signal','available_at_signal','daily_budget_usdt','balance_after_settlement')
    if len(reference)!=len(projected) or any(any(a[k]!=b[k] for k in keys) for a,b in zip(projected,reference)):
        raise ValueError('Public opportunity replay changed an admitted trade')
    return projected,stats


def build(*,initial_capital=10000,default_stake=30,root=ROOT):
    root=Path(root)
    original,matrix=verify_base(root/BASE_FOLDER/'results.json')
    if RULE_ID not in original['descriptive_top_rule_ids']:
        raise ValueError('The disclosed research candidate changed')
    diag_path=root/DIAGNOSTIC_FOLDER/'results.json'
    diagnostic=json.loads(diag_path.read_text(encoding='utf-8'))
    if (diagnostic['plan']['base_report_sha256']!=file_hash(root/BASE_FOLDER/'results.json')
            or len(diagnostic['verified_baselines'])!=10 or len(diagnostic['plan']['descriptive_rule_ids'])!=5):
        raise ValueError('The 40-scenario audit is not tied to the original study')
    for name,expected in diagnostic['plan']['source_sha256'].items():
        if Path(name).name!=name or file_hash(root/'research'/name)!=expected:
            raise ValueError('A diagnostic implementation source changed')
    period_rows=[{'id':key,'label':label,'start_utc':start,'end_exclusive_utc':end,
                 'start_ms':utc_ms(date.fromisoformat(start)),'end_exclusive_ms':utc_ms(date.fromisoformat(end))}
                for key,label,start,end in PERIODS]
    frame,proof=load_recent_predictions(date(2026,1,1),date(2026,11,12))
    rule=matrix[RULE_ID]['rule'];signs=direction_for(frame,rule)
    opportunities=project_opportunities(frame,signs)
    # The old 5-USDT study remains an immutable witness for signal population.
    for p in period_rows:
        ledger,stats=verify_replay(frame,signs,opportunities,p['start_ms'],p['end_exclusive_ms'],5,1000)
        audited=matrix[RULE_ID][p['id']]
        for key in ('signals','wins','losses','refunds','decisive_trades','win_rate'):
            if stats[key]!=audited[key]:raise ValueError('Original recent-window witness no longer reconciles')
        if stats['net_usdt']!=audited['net_units'] or stats['max_drawdown_usdt']!=audited['max_drawdown_units']:
            raise ValueError('Original five-USDT economic witness changed')
    published={};csv_rows=[]
    for p in period_rows:
        first,end=p['start_ms'],p['end_exclusive_ms']
        trades,stats=verify_replay(frame,signs,opportunities,first,end,default_stake,initial_capital)
        monthly,by_symbol=partitions(trades,first,end,initial_capital)
        stresses=[]
        for bps in (.5,1.,2.):
            stressed,stress_stats=verify_replay(frame,signs,opportunities,first,end,default_stake,initial_capital,bps)
            stresses.append({'adverse_bps':bps,'stats':stress_stats,
                             'weekly_interval':moving_week_interval(stressed,first,end)})
        published[p['id']]={'stats':stats,'trades':trades,'monthly':monthly,'by_symbol':by_symbol,
                            'weekly_interval':moving_week_interval(trades,first,end),
                            'stress':stresses,'stake_usdt':default_stake,'initial_capital':initial_capital}
        csv_rows.extend({'period':p['id'],**t} for t in trades)
    # Independent high/low and fractional-cent-reward compatibility checks.
    checks=[]
    for stake in (5,5.01,30,249.99,250):
        for p in period_rows:
            ledger,stats=verify_replay(frame,signs,opportunities,p['start_ms'],p['end_exclusive_ms'],stake,initial_capital)
            checks.append({'stake':stake,'period':p['id'],'signals':stats['signals'],
                           'net_usdt':stats['net_usdt'],'checked_with_sizing_engine':True})
    selected=original['selection']['selected_rule']
    failed=next(r for r in original['matrix'] if r['rule']['id']==selected['id'])
    assumptions={'default_stake':default_stake,'min_stake':5,'max_stake':250,'stake_step':.01,
        'initial_capital':initial_capital,'daily_risk_mode':'stake_multiple','daily_risk_multiplier':5,
        'daily_risk_limit':5*default_stake,'net_payout':.85,'fee':0,'horizon_minutes':30,
        'entry_delay_minutes':1,'max_open_positions':1,'payout_rounding':'half_up_cent',
        'known_unavailable_dates_utc':list(KNOWN_UNAVAILABLE_DATES),
        'initial_capital_resets_per_period':True,'unsettled_principal_reserved':True}
    payload={'schema_version':1,'published_data_version':'2026-11-13-recent-30u-v1',
        'assumptions':assumptions,'periods':period_rows,
        'strategy':{'rule_id':RULE_ID,'label':'Logistic + GBM 双模型同向确认',
                    'description':'两个模型方向一致，且各自方向分数至少 0.60；分数不等于胜率。',
                    'is_live_strategy':False,'posthoc':True,'forward_validated':False},
        'research':{'candidate_configurations':76,'diagnostic_scenarios':40,
                    'original_selected_rule_id':selected['id'],'original_selected_failed':True,
                    'original_selected_recent_win_rate':failed['recent_six_months']['win_rate'],
                    'original_selected_reference_stake':5,
                    'qualified_stable_58_strategies':0,'periods_overlap':True,
                    'comparison_is_after_viewing_results':True},
        'opportunities':opportunities,'published':published,
        'limitations':['该共识策略是比较历史结果后发现的探索候选，不是未经查看的独立验证，也不是当前线上信号模型。',
            '原年初筛选段冻结策略未通过后半年验证；不能用后来表现好的候选替换原选择并称为验证成功。',
            '年初以来与近半年相互重叠；历史高胜率不能证明未来稳定达到 58%。',
            '使用分钟指数开盘价代理真实事件合约行权与到期价格，报价偏差会改变二元胜负。',
            '网页金额只改变本浏览器的研究测算；机器人金额以服务器配置为准，页面不会下单。',
            '调整金额会重新运行信号准入、资金和每日风险；每日风险预算为单笔金额的五倍。',
            '盈亏按 USDT 分四舍五入；只有早于当前信号的已结算结果才能用于资金与日风险判断。'],
        'provenance':{'base_report_sha256':file_hash(root/BASE_FOLDER/'results.json'),
                      'diagnostic_report_sha256':file_hash(diag_path),
                      'feature_store_identity':proof['feature_store_identity'],
                      'opportunities_sha256':canonical_hash(opportunities),
                      'source_sha256':{name:file_hash(root/name) for name in
                          ('scripts/build_recent_public.py','research/public_stake_engine.py','research/stake_sizing_engine.py')},
                      'numeric_replay_checks':checks,'training_or_download_performed':False}}
    out=root/'research'/'recent-public.json'
    out.write_text(json.dumps(payload,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n',encoding='utf-8')
    csv_path=root/'research'/'recent-public-trades.csv'
    with csv_path.open('w',encoding='utf-8',newline='') as stream:
        fields=['period','id','opportunity_id','symbol','direction','signal_ms','entry_ms','settlement_ms',
                'entry_exact','exit_exact','outcome','stake','net_units','equity_at_signal',
                'available_at_signal','daily_budget_usdt','balance_after_settlement']
        money_fields=('stake','net_units','equity_at_signal','available_at_signal',
                      'daily_budget_usdt','balance_after_settlement')
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader()
        for row in csv_rows:
            writer.writerow({**row,**{key:f'{row[key]:.2f}' for key in money_fields}})
    print(json.dumps({'opportunities':len(opportunities),'public_json_bytes':out.stat().st_size,
                      'csv_rows':len(csv_rows),'default_stake':default_stake,'initial_capital':initial_capital,
                      'periods':{k:{m:v['stats'][m] for m in ('signals','win_rate','net_usdt','max_drawdown_usdt')}
                                 for k,v in published.items()}}))
    return payload


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-capital',type=float,default=10000)
    parser.add_argument('--default-stake',type=float,default=30)
    args=parser.parse_args();build(initial_capital=args.initial_capital,default_stake=args.default_stake)
