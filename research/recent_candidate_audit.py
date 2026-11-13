"""Post-hoc diagnostics of exactly five already disclosed recent-window rules.

This expands disclosure, not selection. It never replaces the frozen candidate,
changes parameters, trains models, downloads data or publishes any artifact.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
from statistics import NormalDist
import time

import pandas as pd

from .consensus_search_research import save_ledger
from .event_engine import event_stats, utc_ms
from .event_strategy_research import wilson_lower
from .high_win_research import detail, file_hash, write_json
from .recent_window_data import load_recent_predictions
from .recent_window_research import candidate_rules, direction_for, metric_line, simulate_window

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT/'research'/'recent-window-2026-11-12'/'results.json'
DEFAULT_OUT = ROOT/'research'/'recent-window-2026-11-12-diagnostics'
WINDOWS = ('year_to_date','recent_six_months')
SCENARIOS = (('baseline',0.),('adverse0p5bps',.5),('adverse1bps',1.),('adverse2bps',2.))
CORE = ('signals','wins','losses','refunds','decisive_trades','win_rate','net_units',
        'max_drawdown_units','profit_factor','expectancy_per_trade')


def verify_base(path):
    report=json.loads(path.read_text(encoding='utf-8'))
    folder=path.parent
    if report['plan'] != json.loads((folder/'plan.json').read_text(encoding='utf-8')):
        raise ValueError('Embedded original plan differs from its frozen file')
    if (report['selection'] != json.loads((folder/'selection.json').read_text(encoding='utf-8'))
            or report['selection']['plan_sha256'] != file_hash(folder/'plan.json')):
        raise ValueError('Original selection is not tied to the frozen plan')
    for name,expected in report['plan']['source_sha256'].items():
        if Path(name).name != name or file_hash(ROOT/'research'/name) != expected:
            raise ValueError('Original research source changed: '+name)
    canonical=candidate_rules()
    if report['plan']['candidate_count'] != 76 or report['plan']['rules'] != canonical:
        raise ValueError('The original catalog is not the fixed 76-rule comparison')
    ids=report['descriptive_top_rule_ids']
    if len(ids) != 5 or len(set(ids)) != 5:
        raise ValueError('Require exactly the five disclosed post-hoc descriptions')
    rows={row['rule']['id']:row for row in report['matrix']}
    if len(rows) != 76 or any(key not in rows for key in ids):
        raise ValueError('Original result matrix population differs')
    candidates=[row for row in rows.values() if row['recent_six_months']['decisive_trades']>=200]
    expected=[row['rule']['id'] for row in sorted(candidates,
        key=lambda row:(-(row['recent_six_months']['win_rate'] or 0),row['rule']['id']))[:5]]
    if ids != expected:
        raise ValueError('Disclosed post-hoc list differs from the original ranking')
    return report,rows


def summary(report):
    p=report['plan']; lines=[
        '近期描述榜五项固定策略：补充诊断（本地）',
        '这5项由已看过的近半年表现选出，属于后验描述，不是新的独立验证。',
        '不替换原先冻结候选 '+p['frozen_selected_rule_id']+'，不更改策略参数。',
        '年初以来2026-01-01至2026-11-12；近半年2026-05-12至2026-11-12。UTC，右端不含。',
        '每笔5 USDT、1000初始资金、85%净赔付、零额外手续费、共享最多1笔持仓、每日风险预算25 USDT。',
        '每个区间/价格压力情景独立重跑信号准入、二元胜负、每日风险和资金；不从原收益扣固定费用。',
        '0.5/1/2bp=0.005%/0.01%/0.02%的不利开仓价偏差；不改到期价。',
        '原矩阵的10项基准结果与新账本逐笔重算一致，原始报告与源码哈希校验通过。',
        '']
    for key in p['descriptive_rule_ids']:
        lines.append(key)
        for window,label in (('year_to_date','年初以来'),('recent_six_months','近半年')):
            for scenario,bps in SCENARIOS:
                item=report['results'][key][window][scenario]
                lines.append(f"  {label} / 不利开仓{bps:g}bp：{metric_line(item)}")
            base=report['results'][key][window]['baseline']; block=base['weekly_block_interval']
            lines.append(f"  {label} 基准7日块收益95%区间：{block['net_lower_95']:+.2f} 至 {block['net_upper_95']:+.2f} USDT；胜率区间：{block['win_rate_lower_95']:.2%} 至 {block['win_rate_upper_95']:.2%}")
            lines.append(f"  {label} 基准76项Bonferroni-Wilson下界：{base['bonferroni_wilson_lower_95_76']:.2%}；正收益月份 {base['positive_months']}/{base['active_months']}")
        lines.append('')
    lines += [
        '解释限制：',
        '1. 两个区间重叠；年初以来包含原筛选时段，不能作为两份独立证据。',
        '2. Wilson界依赖独立二项近似；交易序列及BTC/ETH相关，76项校正不覆盖所有历次搜索、时间窗口和压力比较。',
        '3. 7日块区间保留局部相关性，但不是多轮择优后的无偏置信保证；收益下界为负时仍不能认定稳定盈利。',
        '4. 使用分钟指数开盘价代理真实合约行权/到期报价，日历和可成交性仍不完整。',
        '5. 未重训、联网、部署、推送GitHub、改动线上模型或发送通知。',
        '明细见results.json，汇总comparison.csv，月度/币种分别见monthly.csv/by-symbol.csv；每个情景均保存账本及哈希。']
    return '\n'.join(lines)+'\n'


def run(base=DEFAULT_BASE, output_dir=DEFAULT_OUT):
    began=time.perf_counter(); base=Path(base); out=Path(output_dir)
    original, matrix=verify_base(base)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Preserve prior evidence: choose a fresh output directory')
    ids=list(original['descriptive_top_rule_ids'])
    selected=original['selection']['selected_rule']
    source_names=['recent_candidate_audit.py',*original['plan']['source_sha256']]
    plan={'version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'base_report':str(base.relative_to(ROOT)).replace('\\','/'),'base_report_sha256':file_hash(base),
        'base_plan_sha256':file_hash(base.parent/'plan.json'),
        'base_selection_sha256':file_hash(base.parent/'selection.json'),
        'descriptive_rule_ids':ids,'descriptive_rules':[matrix[key]['rule'] for key in ids],
        'frozen_selected_rule_id':selected['id'] if selected else '(none)',
        'posthoc_descriptive_only':True,'replaces_frozen_selection':False,
        'new_independent_validation':False,'no_new_parameters':True,
        'periods':{key:original['plan']['periods'][key] for key in WINDOWS},
        'scenarios':[{'name':name,'adverse_bps':bps,'fee':0.} for name,bps in SCENARIOS],
        'capital':original['plan']['capital'],'bonferroni_family_count':76,
        'bonferroni_two_sided_alpha':.05,'independent_binomial_assumption_not_established':True,
        'prior_rounds_and_window_multiplicity_not_covered':True,
        'each_window_and_scenario_restarts_capital':True,
        'local_only':True,'publish_or_deploy':False,'forward_validated':False,
        'source_sha256':{name:file_hash(ROOT/'research'/name) for name in source_names}}
    write_json(out/'plan.json',plan)
    print('DIAGNOSTIC PLAN FROZEN: exactly original five post-hoc descriptions',flush=True)
    first=date.fromisoformat(plan['periods']['year_to_date']['start_utc'])
    end=date.fromisoformat(plan['periods']['year_to_date']['end_exclusive_utc'])
    frame,provenance=load_recent_predictions(first,end)
    if provenance['feature_store_identity']!=original['provenance']['feature_store_identity']:
        raise ValueError('Diagnostic data identity differs from the original comparison')
    z=NormalDist().inv_cdf(1-.05/(2*76))
    results={}; table=[]; months=[]; symbols=[]; verified_baselines=[]
    for key in ids:
        rule=matrix[key]['rule']; signs=direction_for(frame,rule); results[key]={}
        for window in WINDOWS:
            dates=plan['periods'][window];low=date.fromisoformat(dates['start_utc']);high=date.fromisoformat(dates['end_exclusive_utc'])
            results[key][window]={}
            for scenario,bps in SCENARIOS:
                ledger,stats=simulate_window(frame,rule,low,high,direction=signs,stress={'adverse_bps':bps,'fee':0.})
                recomputed=event_stats(ledger,utc_ms(low),utc_ms(high),payout=.85,fee=0.,bootstrap=False)
                if any(recomputed[name]!=stats[name] for name in CORE):
                    raise ValueError('Ledger and simulator statistics differ')
                if scenario=='baseline':
                    if stats != matrix[key][window]:
                        raise ValueError('Base result no longer matches original matrix: '+key+' / '+window)
                    verified_baselines.append({'rule_id':key,'window':window,'matrix_matches':True})
                value=detail(ledger,utc_ms(low),utc_ms(high))
                value['adverse_bps']=bps;value['fee']=0.
                value['raw_opportunities']=stats['raw_opportunities']
                value['bonferroni_wilson_lower_95_76']=wilson_lower(value['wins'],value['decisive_trades'],z)
                filename=f'{key}-{window}-{scenario}-trades.csv'
                save_ledger(out/filename,ledger)
                value['ledger']=filename;value['ledger_sha256']=file_hash(out/filename)
                results[key][window][scenario]=value
                common={'rule_id':key,'window':window,'scenario':scenario,'adverse_bps':bps,**dates}
                table.append({**common,**{name:value[name] for name in CORE},
                    'weekly_net_lower_95':value['weekly_block_interval']['net_lower_95'],
                    'weekly_net_upper_95':value['weekly_block_interval']['net_upper_95'],
                    'bonferroni_wilson_lower_95_76':value['bonferroni_wilson_lower_95_76'],
                    'positive_months':value['positive_months'],'active_months':value['active_months']})
                months.extend({**common,**row} for row in value['by_month'])
                symbols.extend({**common,'symbol':symbol,**row} for symbol,row in value['by_symbol'].items())
            print(key+' '+window+' 2bp '+metric_line(results[key][window]['adverse2bps']),flush=True)
    if file_hash(base)!=plan['base_report_sha256']:
        raise RuntimeError('Original report changed during diagnostics')
    for name,expected in plan['source_sha256'].items():
        if file_hash(ROOT/'research'/name)!=expected:raise RuntimeError('Source changed during diagnostics: '+name)
    report={'plan':plan,'provenance':provenance,'verified_baselines':verified_baselines,'results':results,
            'generated_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':round(time.perf_counter()-began,2),
            'local_only':True,'posthoc_descriptive_only':True,'forward_validated':False,'publish_or_deploy':False}
    write_json(out/'results.json',report)
    for name,rows in (('comparison.csv',table),('monthly.csv',months),('by-symbol.csv',symbols)):
        pd.DataFrame(rows).to_csv(out/name,index=False,encoding='utf-8-sig')
    (out/'summary.txt').write_text(summary(report),encoding='utf-8')
    print('DONE '+json.dumps({'seconds':report['elapsed_seconds'],'baseline_checks':len(verified_baselines),'scenarios':len(table),'output':str(out)}),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--output-dir',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args();run(args.base,args.output_dir)
