"""Build a local human-readable research register from completed study reports.

No candidate is selected here. The register preserves failed experiments and
separates calendar corrections from the original frozen evaluation results.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research'
STUDIES = (
    ('high-win-results.json', '原模型信号过滤', 16),
    ('model-search-results.json', '模型与训练采样比较', 60),
    ('consensus-search-results.json', '多个模型共同确认', 16),
    ('adaptive-threshold-results.json', '逐周历史胜率筛选', 12),
    ('margin-search-results.json', '考虑2bp价格余量的三分类', 8),
    ('reversal-specialist-results.json', '极端偏离反转专用模型', 1),
    ('metrics-search-results.json', '持仓量与多空结构特征配对研究', 8),
    ('metrics-threshold-results.json', '持仓特征模型的高分数门槛检验', 16),
)


def read(name):
    return json.loads((OUT/name).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def percent(value):
    return '无交易' if value is None else f'{100*value:.2f}%'


def stats_line(stats):
    return (f"{stats['signals']}笔 / 胜率{percent(stats['win_rate'])} / "
            f"净收益{stats['net_units']:+.2f} USDT / 最大回撤{stats['max_drawdown_units']:.2f} USDT")


def point_gate(row):
    v, c = row.get('validation',{}), row.get('historical_confirmation',{})
    return (v.get('decisive_trades',0)>=400 and c.get('decisive_trades',0)>=100
            and (v.get('win_rate') or 0)>=.58 and (c.get('win_rate') or 0)>=.58
            and v.get('net_units',0)>0 and c.get('net_units',0)>0)


def build():
    registry = []; missing = []
    for filename,label,count in STUDIES:
        if not (OUT/filename).exists():
            missing.append({'file':filename,'label':label,'planned_count':count})
            continue
        result = read(filename)
        actual = len(result['all_candidates'])
        if actual != count:
            raise ValueError('Unexpected study candidate count: '+filename)
        registry.append({'file':filename,'label':label,'candidate_configurations':actual,
                         'matched_control_configurations':len(result.get('matched_controls',[])),
                         'selected_rule':result['selection']['selected_rule'],
                         'selected_statistics':result['selected'],
                         'historical_gates_passed':result.get('historical_research_gates_passed',result.get('research_gates_passed',False)),
                         'point_gates_passed_candidates':sum(point_gate(r) for r in result['all_candidates']),
                         'checks':result['checks'],'sha256':sha(OUT/filename)})
    total = sum(r['candidate_configurations'] for r in registry)
    control_total = sum(r['matched_control_configurations'] for r in registry)
    passed = [r for r in registry if r['historical_gates_passed']]
    test_evidence = None
    test_log = OUT/'high-win-local-tests.log'
    if test_log.exists():
        log = test_log.read_text(encoding='utf-8',errors='replace')
        counts = re.findall(r'Ran (\d+) tests in ([\d.]+)s',log)
        if counts:
            test_evidence = {'file':test_log.name,'count':int(counts[-1][0]),
                             'seconds':float(counts[-1][1]),'passed':log.rstrip().endswith('OK'),
                             'sha256':sha(test_log)}
    lines = [
        'BTC/ETH 30分钟事件合约：本地高胜率研究进展',
        f'生成时间（UTC）：{datetime.now(timezone.utc).isoformat()}',
        '',
        f'已完成 {len(registry)} 轮、{total} 项候选配置，另有 {control_total} 项不参与筛选的配对对照。含重复基线和相关模型，不能当作相互独立的实验。',
        ('已有候选通过本轮历史门槛，但仍须新的前瞻数据验证。' if passed else
         '目前没有按冻结筛选流程通过全部历史验收门槛的策略。不能宣称已找到稳定58%或保证盈利。'),
        '所有结果保存在本地；没有上传GitHub、连接/更改你的部署服务器、替换在线模型或发送企业微信消息。',
        '',
        '统一经济口径：净赔付85%，无额外手续费，盈亏平衡胜率54.0541%；',
        '每笔固定5 USDT，初始本金1000 USDT，两币共享本金，最多1个持仓，日累计亏损及待结算风险上限25 USDT。',
        '已完成分钟特征 -> 延迟1分钟入场 -> 持有30分钟；指数分钟开盘价仅为真实合约报价的替代值。',
        '3年特征缓存：2023-11-09至2026-11-12（右端不含）；滚动训练180天，每30天更新。',
        '一般筛选期：2024-05-07至2025-05-07；自适应研究因统一暖机从2024-11-04开始。',
        '长期验证期：2025-05-07至2026-08-09；近期检查：2026-08-09至2026-11-12。全部为UTC、右端不含。',
        '所有历史均已多次研究，不能再称为从未查看的测试集。',
        (f"相关本地测试：{test_evidence['count']}项，全部通过。" if test_evidence and test_evidence['passed'] else
         '测试状态请查看对应研究文件及本地测试日志。'),
        '',
        '完成的研究：',
    ]
    for row in registry:
        lines.extend([f"- {row['label']}：{row['candidate_configurations']}项。",f"  详细结果：{OUT/row['file']}"])
        if row['matched_control_configurations']:
            lines.append(f"  另有{row['matched_control_configurations']}个对照，和增强模型使用完全相同的训练及预测样本。")
        selected = row['selected_rule']
        if not selected:
            lines.append('  筛选期没有符合预设条件的候选，未事后挑选验证期赢家。')
        else:
            lines.append('  冻结候选：'+selected['id'])
            for key,label in [('discovery','筛选期'),('validation','长期验证'),('historical_confirmation','近期检查')]:
                if key in row['selected_statistics']:
                    lines.append(f'  {label}：'+stats_line(row['selected_statistics'][key]))
        lines.append(f"  按该轮原始口径，同时满足两段58%与400/100笔下限的候选数：{row['point_gates_passed_candidates']}。")
    corrections = []
    if (OUT/'calendar-audit-results.json').exists():
        calendar = read('calendar-audit-results.json')
        corrections = calendar['rows']
        lines += ['', '交易日历修正（此前两轮主结果遗漏了已公告的不可交易日期）：',
                  '排除2026-01-17、18、19、24、25（UTC），在准入持仓前剔除，并重新模拟。',
                  '原始研究文件保留；后续自适应、价格余量、专用模型均在主流程应用日历。']
        for rule in ('logistic_halfhour_0.64_exhaustion','gbm_deeper_5m_0.60_exhaustion'):
            records = [r for r in corrections if r['rule_id']==rule]
            label = '原冻结候选' if rule.startswith('logistic') else 'GBM探索性对照（不可据后来结果替换冻结候选）'
            lines.append(label+'：')
            for row in records:
                lines.append('  '+row['period']+'：'+stats_line(row['after_calendar_gate']))
    if (OUT/'consensus-search-results.json').exists():
        report = read('consensus-search-results.json')
        calendar = next(s for s in report['stresses'] if s['name']=='known_unavailable_calendar')
        lines += ['共识候选应用同一日历后：'+stats_line(calendar['test'])]
    if (OUT/'metrics-search-results.json').exists():
        metrics = read('metrics-search-results.json')
        lines += ['', '新增持仓结构研究的特定限制：',
                  '使用币安官方公开持仓量、多空账户/仓位比、主动买卖比档案，逐个校验官方CHECKSUM。',
                  '固定新增8个特征，只比较相同样本上的60特征与68特征模型；不因控制组后来表现而更换候选。',
                  '记录时间后延迟10分钟才使用，随后有效期不足5分钟；精确OI历史端点缺失时不插值。',
                  '该时间规则是研究假设。档案的发布/修订时间不能证明原API当时的可见时间或数据未曾修订。',
                  '新增特征收益改善的独立历史判定：'+str(metrics['incremental_historical_evidence_passed'])+'；不代表前瞻验证已完成。']
    additional_tests = None
    extension_test_log = OUT/'metrics-threshold-tests.log'
    if extension_test_log.exists():
        log = extension_test_log.read_text(encoding='utf-8',errors='replace')
        matches = re.findall(r'Ran (\d+) tests in ([\d.]+)s',log)
        if matches:
            additional_tests = {'file':extension_test_log.name,'module':'tests.test_metrics_threshold',
                                'count':int(matches[-1][0]),'passed':log.rstrip().endswith('OK'),
                                'sha256':sha(extension_test_log)}
    if (OUT/'metrics-threshold-results.json').exists():
        extension = read('metrics-threshold-results.json')
        example = next(r for r in extension['all_candidates']
                       if r['rule']['id']=='metrics_threshold_logistic_aug_0.68_exhaustion')
        lines += ['', '提高分数门槛的反例（仅披露，不作为入选策略）：',
                  'Logistic增强模型、0.68门槛、极端偏离反转：',
                  '  筛选期：'+stats_line(example['discovery']),
                  '  连续后续期间：'+stats_line(example['continuous_post_selection']),
                  '筛选期虽达到59.79%，后续却降至53.17%；不能只保留好看的时间段。']
        if additional_tests and additional_tests['passed']:
            lines.append(f"高分数门槛研究另有{additional_tests['count']}项专门测试通过，日志：{extension_test_log}")
    lines += ['', '判定原则与下一步证据：',
              '1. 筛选期至少500笔且每币种100笔，58%且净收益为正；随后冻结选择，不根据后续赢家换策略。',
              '2. 两段后续历史各需至少400/100笔、均达58%和正收益；连续期间以7日块重采样检查收益与胜率区间。',
              '3. 少数几十笔的60%—70%不够证明稳定性；周块区间跨越零收益时，正收益点估计仍不确定。',
              '4. 2bp不利入场价格（0.02%）会重新计算输赢、持仓和每日风控；不能只从净利减一个固定成本。',
              '5. 多轮挑选会过拟合。每轮内部的校正不覆盖全部历次研究，不能把不断试出的历史高胜率称为前瞻证据。',
              '6. 下一阶段应冻结少量候选，积累新的分钟信号、实际可成交行权价与到期价，按预定时间及样本量检验。',
              '   不因中途胜率好看而提前宣布成功，不将本文中的原始模型评分视为真实胜率。']
    if missing:
        lines += ['', '仍待完成：']+[r['label'] for r in missing]
    lines += ['', '在项目目录的PowerShell中重新生成汇总：',
              r"& 'C:\Users\23977\AppData\Local\Programs\Python\Python312\python.exe' -m research.build_high_win_summary",
              '各研究对应 research/*_research.py；程序使用本地已校验缓存，生成结果和逐笔账本。']
    text = '\n'.join(lines)+'\n'
    (OUT/'high-win-local-summary.txt').write_text(text,encoding='utf-8-sig')
    register = {'generated_at_utc':datetime.now(timezone.utc).isoformat(),
                'completed_rounds':len(registry),'candidate_configurations':total,
                'matched_control_configurations':control_total,'total_evaluated_configurations':total+control_total,
                'independent_experiments':False,'historical_gate_passed_rounds':len(passed),
                'pristine_holdout':False,'forward_validated':False,'local_only':True,
                'studies':registry,'pending':missing,
                'test_evidence':test_evidence,
                'additional_test_evidence':additional_tests,
                'calendar_audit_sha256':sha(OUT/'calendar-audit-results.json') if corrections else None,
                'summary_sha256':sha(OUT/'high-win-local-summary.txt')}
    (OUT/'high-win-research-register.json').write_text(json.dumps(register,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'completed_rounds':len(registry),'candidate_configurations':total,
                      'matched_control_configurations':control_total,
                      'passed_rounds':len(passed),'pending':missing},ensure_ascii=False))
    return register


if __name__=='__main__':
    build()
