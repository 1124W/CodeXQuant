const reportEl = id => document.getElementById(id);
const percent = value => Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";

function renderResearch(report) {
  const dates = [report.start_utc, report.cutoff_utc, report.end_exclusive_utc].map(value => {
    if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) throw new Error("Invalid report date");
    const timestamp = Date.parse(`${value}T00:00:00Z`);
    if (!Number.isFinite(timestamp) || new Date(timestamp).toISOString().slice(0, 10) !== value) throw new Error("Invalid report date");
    return timestamp;
  });
  if (!(dates[0] < dates[1] && dates[1] < dates[2]) || dates[2] > Date.now()) throw new Error("Invalid report range");
  const legacy = report.schema_version === undefined || report.schema_version === 1;
  if (!legacy && report.schema_version !== 2) throw new Error("Unsupported report version");
  const fee = legacy ? 0 : report.fee_per_trade;
  const expectedBreakEven = (1 + fee) / (1 + report.payout);
  if (!Number.isFinite(report.payout) || report.payout <= 0 || !Number.isFinite(fee) || fee < 0 || fee >= report.payout ||
      !Number.isFinite(report.break_even_win_rate) || Math.abs(report.break_even_win_rate - expectedBreakEven) > .0001) throw new Error("Invalid report costs");
  const selected = report.chosen;
  if (selected) {
    for (const sample of [selected.train, selected.test]) {
      if (!sample || !Number.isInteger(sample.signals) || sample.signals < 0 ||
          !Number.isFinite(sample.net_units) || !Number.isFinite(sample.max_drawdown_units) || sample.max_drawdown_units < 0 ||
          (sample.signals > 0 && (!Number.isFinite(sample.win_rate) || sample.win_rate < 0 || sample.win_rate > 1 ||
           !Number.isFinite(sample.win_rate_lower_95) || sample.win_rate_lower_95 < 0 || sample.win_rate_lower_95 > sample.win_rate))) throw new Error("Invalid report metrics");
    }
  }
  const stale = Date.now() - dates[2] > 3 * 86400000;
  const checks = report.validation_checks;
  const passed = !legacy && selected && report.publishable === true &&
    ["selected_on_training_only", "holdout_min_50_signals", "holdout_lower_95_above_break_even", "holdout_positive_net"].every(key => checks?.[key] === true) &&
    selected.train.signals >= 100 && selected.test.signals >= 50 && selected.test.net_units > 0 && selected.test.win_rate_lower_95 > expectedBreakEven;
  reportEl("researchTitle").textContent = legacy ? "历史报告待按新结算边界重新验证" :
    !selected ? "训练样本不足，暂无候选规则" : passed ? "历史统计门槛通过，仍需前向验证" : "当前规则未通过历史统计门槛";
  reportEl("researchBadge").textContent = legacy ? "旧版报告" : stale ? "历史快照" : passed ? "待前向验证" : "研究中";
  reportEl("trainLabel").textContent = `${(dates[1] - dates[0]) / 86400000} 天 · 训练期`;
  reportEl("testLabel").textContent = `${(dates[2] - dates[1]) / 86400000} 天 · 留出检验期`;
  reportEl("breakEvenRate").textContent = percent(report.break_even_win_rate);
  reportEl("costDetail").textContent = `赢 +${report.payout} / 输 −1 · 每笔成本 ${fee} 单位`;
  for (const split of ["train", "test"]) {
    const sample = selected?.[split];
    reportEl(`${split}Rate`).textContent = percent(sample?.win_rate);
    reportEl(`${split}Detail`).textContent = sample ? `${sample.signals} 次 · ${sample.net_units > 0 ? "+" : ""}${sample.net_units} 单位` : "无符合训练样本门槛的规则";
  }
  const test = selected?.test;
  reportEl("confidenceInterval").textContent = test?.signals ?
    `${percent(test.win_rate_lower_95)} – ${percent(test.win_rate_upper_95)}${legacy ? "（旧版仅有下限）" : "（Wilson）"}` : "无样本";
  reportEl("testDrawdown").textContent = test ? `${test.max_drawdown_units} 单位` : "—";
  reportEl("reportCoverage").textContent = `${report.start_utc} 至 ${report.end_exclusive_utc}（UTC，结束日期不含）`;
  reportEl("researchNote").textContent = [
    legacy ? "此报告由旧版引擎生成，未应用已修复的结算边界；数字仅供历史参考，请重新运行研究脚本。" : "参数仅按训练表现选择；统计门槛通过不代表允许实盘。",
    stale ? "报告距今超过 3 天，请留意数据时效。" : "",
    "BTC/ETH 及连续交易存在相关性，胜率区间可能偏乐观；同一留出期反复调参会使检验失效。未模拟真实成交延迟与点差。"
  ].filter(Boolean).join(" ");
}

fetch("research/results.json", { cache: "no-store", signal: AbortSignal.timeout(10000) })
  .then(response => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); })
  .then(renderResearch)
  .catch(error => {
    console.warn("Research report unavailable", error);
    reportEl("researchTitle").textContent = "研究报告暂不可用";
    reportEl("researchBadge").textContent = "加载失败";
    reportEl("researchNote").textContent = "无法读取有效的研究报告。请通过本地 HTTP 服务打开页面，或稍后重试。";
  });

const eventUnits = value => `${value > 0 ? "+" : ""}${value.toFixed(2)}`;
const eventDate = value => {
  if (typeof value !== "string" || !/^\d{4}-\d{2}-\d{2}$/.test(value)) throw new Error("Invalid event date");
  const timestamp = Date.parse(`${value}T00:00:00Z`);
  if (!Number.isFinite(timestamp) || new Date(timestamp).toISOString().slice(0, 10) !== value) throw new Error("Invalid event date");
  return timestamp;
};
const eventChecks = ["walk_forward_min_samples", "walk_forward_target_met", "walk_forward_positive_net",
  "walk_forward_confidence_above_break_even", "walk_forward_block_confidence_above_break_even", "walk_forward_block_net_positive",
  "walk_forward_stability", "holdout_min_samples", "holdout_target_met", "holdout_positive_net",
  "holdout_confidence_above_break_even", "holdout_block_confidence_above_break_even", "holdout_block_net_positive", "selected_on_training_only"];
let currentEventReport = null;

function validateEventSample(sample, payout, fee, stake) {
  if (!sample || !["signals", "wins", "losses", "refunds", "decisive_trades"].every(key => Number.isInteger(sample[key]) && sample[key] >= 0) ||
      sample.signals !== sample.wins + sample.losses + sample.refunds || sample.decisive_trades !== sample.wins + sample.losses ||
      !Number.isFinite(sample.net_units) || !Number.isFinite(sample.max_drawdown_units) || sample.max_drawdown_units < 0 ||
      Math.abs(sample.net_units - stake * (sample.wins * payout - sample.losses - sample.signals * fee)) > .00001) throw new Error("Inconsistent event counts or returns");
  if (!sample.decisive_trades) {
    if (sample.win_rate !== null) throw new Error("Empty sample has a win rate");
    return;
  }
  if (!Number.isFinite(sample.win_rate) || Math.abs(sample.win_rate - sample.wins / sample.decisive_trades) > 1e-10) throw new Error("Inconsistent event win rate");
  for (const [lower, upper] of [["win_rate_lower_95", "win_rate_upper_95"], ["block_bootstrap_lower_95", "block_bootstrap_upper_95"]]) {
    if (sample[lower] === null && sample[upper] === null) continue;
    if (!Number.isFinite(sample[lower]) || !Number.isFinite(sample[upper]) || sample[lower] < 0 || sample[upper] > 1 || sample[lower] > sample[upper]) throw new Error("Invalid event interval");
  }
}

function validateEventReport(report) {
  if (report?.schema_version !== 3 || report.live_approved !== false) throw new Error("Unsupported event report");
  const start = eventDate(report.start_utc), cutoff = eventDate(report.cutoff_utc), end = eventDate(report.end_exclusive_utc);
  const config = report.config;
  if (!(start < cutoff && cutoff < end && end <= Date.now()) || !config ||
      !Number.isFinite(report.payout) || report.payout <= 0 || !Number.isFinite(report.fee_per_trade) || report.fee_per_trade < 0 || report.fee_per_trade >= report.payout ||
      !Number.isFinite(report.target_win_rate) || report.target_win_rate <= 0 || report.target_win_rate >= 1 ||
      !Number.isFinite(config.stake) || config.stake <= 0 || !Number.isFinite(config.initial_bankroll) || config.initial_bankroll < config.stake ||
      !Number.isInteger(config.min_test_trades) || config.min_test_trades < 1 ||
      !Number.isFinite(report.break_even_win_rate) || Math.abs(report.break_even_win_rate - (1 + report.fee_per_trade) / (1 + report.payout)) > 1e-10 ||
      !Array.isArray(report.walk_forward?.folds) || !report.final_holdout || !Array.isArray(report.equity_curve) || !Array.isArray(report.trade_ledger)) throw new Error("Invalid event configuration");
  const verify = sample => validateEventSample(sample, report.payout, report.fee_per_trade, config.stake);
  verify(report.walk_forward.aggregate); verify(report.final_holdout.test);
  let previousEnd = eventDate(report.walk_forward.start_utc);
  for (const fold of report.walk_forward.folds) {
    verify(fold.test);
    const trainStart = eventDate(fold.train_start), trainEnd = eventDate(fold.train_end), testStart = eventDate(fold.test_start), testEnd = eventDate(fold.test_end);
    if (!(start <= trainStart && trainStart < trainEnd && trainEnd === testStart && testStart === previousEnd && testEnd > testStart && testEnd <= cutoff)) throw new Error("Invalid fold boundaries");
    if (!fold.chosen && fold.test.signals) throw new Error("No selected rule for fold trades");
    previousEnd = testEnd;
  }
  if (previousEnd !== cutoff || eventDate(report.final_holdout.start_utc) !== cutoff || eventDate(report.final_holdout.end_exclusive_utc) !== end) throw new Error("Invalid holdout boundaries");
  for (const split of [report.walk_forward, report.final_holdout]) {
    if (!split.by_symbol || !["BTCUSDT", "ETHUSDT"].every(symbol => split.by_symbol[symbol])) throw new Error("Missing symbol statistics");
    for (const sample of Object.values(split.by_symbol)) verify(sample);
  }
  const ledgerGroups = { walk_forward: [], final_holdout: [] };
  for (const trade of report.trade_ledger) {
    if (!ledgerGroups[trade.split] || !["BTCUSDT", "ETHUSDT"].includes(trade.symbol) || !["win", "loss", "refund"].includes(trade.outcome) ||
        ![trade.signal_ms, trade.entry_ms, trade.settlement_ms].every(Number.isInteger) || !(trade.signal_ms <= trade.entry_ms && trade.entry_ms < trade.settlement_ms) ||
        !Number.isFinite(trade.net_units) || trade.signal_ms < start || trade.settlement_ms >= end ||
        (trade.split === "walk_forward" ? trade.settlement_ms >= cutoff : trade.signal_ms < cutoff)) throw new Error("Invalid event trade ledger");
    ledgerGroups[trade.split].push(trade);
  }
  for (const [key, sample] of [["walk_forward", report.walk_forward.aggregate], ["final_holdout", report.final_holdout.test]]) {
    const trades = ledgerGroups[key];
    if (trades.length !== sample.signals || Math.abs(trades.reduce((sum, trade) => sum + trade.net_units, 0) - sample.net_units) > .00001 ||
        ["win", "loss", "refund"].some((outcome, i) => trades.filter(trade => trade.outcome === outcome).length !== sample[["wins", "losses", "refunds"][i]])) throw new Error("Ledger disagrees with event summary");
  }
  for (const sample of report.cost_scenarios ?? []) {
    if (!Number.isFinite(sample.payout) || sample.payout <= 0 || !Number.isFinite(sample.fee_per_trade) || sample.fee_per_trade < 0 || sample.fee_per_trade >= sample.payout) throw new Error("Invalid stress costs");
    validateEventSample(sample.walk_forward, sample.payout, sample.fee_per_trade, config.stake);
    validateEventSample(sample.final_holdout, sample.payout, sample.fee_per_trade, config.stake);
  }
  const lastTime = { walk_forward: 0, final_holdout: 0 };
  for (const point of report.equity_curve) {
    if (!(point.split in lastTime) || !Number.isInteger(point.timestamp_ms) || point.timestamp_ms <= lastTime[point.split] ||
        ![point.balance_units, point.net_units, point.drawdown_units].every(Number.isFinite) || point.drawdown_units < 0 ||
        Math.abs(point.balance_units - config.initial_bankroll - point.net_units) > .00001) throw new Error("Invalid event equity");
    lastTime[point.split] = point.timestamp_ms;
  }
  return report;
}

function eventCells(target, rows) {
  const body = reportEl(target); body.replaceChildren();
  for (const cells of rows) {
    const row = document.createElement("tr");
    for (const cell of cells) {
      const td = document.createElement("td");
      td.textContent = typeof cell === "object" ? cell.text : cell;
      if (typeof cell === "object") td.className = cell.className;
      row.appendChild(td);
    }
    body.appendChild(row);
  }
}

function drawEventEquity(report) {
  const canvas = reportEl("eventEquity"), bounds = canvas.getBoundingClientRect(), ratio = devicePixelRatio || 1;
  if (!bounds.width || !bounds.height) return;
  canvas.width = bounds.width * ratio; canvas.height = bounds.height * ratio;
  const ctx = canvas.getContext("2d"); ctx.scale(ratio, ratio);
  const width = bounds.width, height = bounds.height, bankroll = report.config.initial_bankroll;
  const all = report.equity_curve, values = [bankroll, ...all.map(point => point.balance_units)];
  const low = Math.min(...values), high = Math.max(...values), spread = high - low || 1;
  const y = value => 22 + (high - value) / spread * (height - 50);
  ctx.font = "10px ui-monospace,monospace"; ctx.fillStyle = "#8991a3";
  for (let i = 0; i <= 3; i++) {
    const value = high - spread * i / 3, level = y(value);
    ctx.strokeStyle = "rgba(255,255,255,.07)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(68, level); ctx.lineTo(width - 15, level); ctx.stroke(); ctx.fillText(value.toFixed(2), 10, level + 3);
  }
  for (const [key, color] of [["walk_forward", "#b8f348"], ["final_holdout", "#57d8ff"]]) {
    const start = eventDate(key === "walk_forward" ? report.walk_forward.start_utc : report.cutoff_utc), end = eventDate(key === "walk_forward" ? report.cutoff_utc : report.end_exclusive_utc);
    const series = [{ timestamp_ms: start, balance_units: bankroll }, ...all.filter(point => point.split === key)];
    if (series.length === 1) continue;
    const x = time => 68 + (time - start) / (end - start) * (width - 84);
    ctx.beginPath(); series.forEach((point, i) => i ? ctx.lineTo(x(point.timestamp_ms), y(point.balance_units)) : ctx.moveTo(x(point.timestamp_ms), y(point.balance_units)));
    ctx.lineWidth = 1.8; ctx.strokeStyle = color; ctx.stroke();
  }
  ctx.fillText("各自检验期开始", 68, height - 8); ctx.textAlign = "right"; ctx.fillText("检验期结束", width - 15, height - 8);
}

function renderEventResearch(raw) {
  const report = validateEventReport(raw), walk = report.walk_forward.aggregate, final = report.final_holdout.test;
  currentEventReport = report;
  const target = report.target_win_rate, min = report.config.min_test_trades, checks = report.validation_checks ?? {};
  const credible = sample => sample.decisive_trades >= min && sample.win_rate >= target && sample.net_units > 0 &&
    sample.win_rate_lower_95 > sample.effective_break_even_win_rate && sample.block_bootstrap_lower_95 > sample.effective_break_even_win_rate && sample.block_bootstrap_net_lower_95 > 0;
  const passed = report.publishable === true && eventChecks.every(key => checks[key] === true) && credible(walk) && credible(final);
  reportEl("eventTitle").textContent = passed ? "历史门槛通过，等待新增前向样本" : !final.signals ? "历史留出期空仓，目标尚未验证" : "当前策略尚未通过盈利验证";
  reportEl("eventBadge").textContent = passed ? "历史检验通过" : !final.signals ? "空仓 / 无样本" : "目标未验证";
  reportEl("eventSummary").textContent = `目标为 ${percent(target)}，每次固定投入 ${report.config.stake} 单位。${report.walk_forward.folds.length} 轮滚动检验与最后 ${report.config.holdout_days} 天历史留出期分开统计；历史结果不表示未来胜率。`;
  reportEl("eventTarget").textContent = percent(target);
  reportEl("eventBreakEven").textContent = percent(report.break_even_win_rate);
  reportEl("eventCost").textContent = `赢 +${report.payout} / 输 −1 / 平局退本金 · 研究成本 ${report.fee_per_trade} / 单位`;
  for (const [key, sample] of [["Walk", walk], ["Holdout", final]]) {
    reportEl(`event${key}Rate`).textContent = sample.decisive_trades ? percent(sample.win_rate) : "无样本";
    reportEl(`event${key}Rate`).className = !sample.signals ? "" : sample.net_units > 0 && sample.win_rate >= target ? "positive" : "negative";
    reportEl(`event${key}Detail`).textContent = sample.signals ? `${sample.signals} 笔 · ${sample.wins} 胜 / ${sample.losses} 负 / ${sample.refunds} 退款 · 净收益 ${eventUnits(sample.net_units)} 单位` : "训练期没有满足条件的规则，本期保持空仓。";
  }
  reportEl("eventWalkConfidence").textContent = walk.decisive_trades ? `${percent(walk.block_bootstrap_lower_95)} – ${percent(walk.block_bootstrap_upper_95)}` : "无已决交易，无法估计";
  reportEl("eventWalkDrawdown").textContent = `${walk.max_drawdown_units.toFixed(2)} 单位`;
  reportEl("eventConfidence").textContent = final.decisive_trades ? `${percent(final.block_bootstrap_lower_95)} – ${percent(final.block_bootstrap_upper_95)}` : "无已决交易，无法估计";
  reportEl("eventDrawdown").textContent = `${final.max_drawdown_units.toFixed(2)} 单位`;
  reportEl("eventCoverage").textContent = `${report.start_utc} 至 ${report.end_exclusive_utc} · 留出自 ${report.cutoff_utc}`;
  reportEl("eventModel").textContent = `Binance 1 分钟指数数据 · 完整 30m K 线产生信号 · 延迟 ${report.config.latency_minutes} 分钟入场 · ${report.config.horizon_minutes} 分钟到期。分钟开盘价为近似值，不能还原下一秒实际开仓与结算。`;
  const gates = [
    [checks.walk_forward_target_met, `滚动样本外胜率 ${percent(walk.win_rate)} / 目标 ${percent(target)}`],
    [checks.holdout_target_met, `历史留出胜率 ${percent(final.win_rate)} / 目标 ${percent(target)}`],
    [checks.walk_forward_min_samples && checks.holdout_min_samples, `每段至少 ${min} 笔已决交易：目前 ${walk.decisive_trades} / ${final.decisive_trades}`],
    [checks.walk_forward_positive_net && checks.holdout_positive_net, "两段扣除研究成本后均盈利"],
    [checks.walk_forward_confidence_above_break_even && checks.holdout_confidence_above_break_even, "两段胜率 95% 下限均高于盈亏平衡"],
    [checks.walk_forward_block_confidence_above_break_even && checks.holdout_block_confidence_above_break_even && checks.walk_forward_block_net_positive && checks.holdout_block_net_positive, "按日重采样后胜率和净收益仍有统计优势"],
    [checks.walk_forward_stability, `至少 3 轮有交易，且 2/3 盈利：${report.walk_forward.positive_folds} / ${report.walk_forward.active_folds} 轮`],
    [checks.selected_on_training_only, "每轮只根据此前训练期选择规则"]
  ];
  const gateList = reportEl("eventGateList"); gateList.replaceChildren();
  for (const [ok, label] of gates) { const li = document.createElement("li"); li.className = ok === true ? "passed" : "pending"; li.textContent = label; gateList.appendChild(li); }
  const ruleName = chosen => !chosen ? "空仓" : ({ mean_reversion: "震荡反转", trend_pullback: "趋势回调", trend_breakout: "趋势突破" }[chosen.parameters.family] ?? "固定规则");
  const counts = sample => `${sample.wins} / ${sample.losses} / ${sample.refunds}`;
  const netCell = sample => ({ text: `${eventUnits(sample.net_units)} 单位`, className: sample.net_units > 0 ? "positive" : sample.net_units < 0 ? "negative" : "" });
  eventCells("eventFoldRows", report.walk_forward.folds.map(fold => [`${fold.test_start} → ${fold.test_end}`, ruleName(fold.chosen), counts(fold.test), percent(fold.test.win_rate), netCell(fold.test)]));
  eventCells("eventSymbolRows", ["BTCUSDT", "ETHUSDT"].flatMap(symbol => [[`${symbol.slice(0, 3)} · 滚动`, report.walk_forward.by_symbol[symbol]], [`${symbol.slice(0, 3)} · 留出`, report.final_holdout.by_symbol[symbol]]].map(([label, sample]) => [label, counts(sample), percent(sample.win_rate), netCell(sample)])));
  const names = { base: "基准", lower_payout_075: "赔付降至 0.75", lower_payout_070: "赔付降至 0.70", higher_cost: "成本提高", adverse_strike_2bps: "开仓价不利移动 2bp" };
  eventCells("eventScenarioRows", (report.cost_scenarios ?? []).map(scenario => [names[scenario.name] ?? "压力情景", `${scenario.payout} / ${scenario.fee_per_trade}`, netCell(scenario.walk_forward), netCell(scenario.final_holdout)]));
  reportEl("eventEquityNote").textContent = `两段分别以 ${report.config.initial_bankroll} 单位本金开始；每笔 ${report.config.stake} 单位，不加倍追损。${!final.signals ? "历史留出期无交易，未绘制该段曲线。" : "横轴为各自检验期进度。"}`;
  reportEl("eventNote").textContent = `胜率以胜 /（胜 + 负）计算，退款另列；交易赔付固定为回测假设，实际赔率应逐笔记录。${report.validation_context?.pristine_unobserved_holdout === false ? "部分日期曾出现在旧研究中，历史留出期不属于全新的独立证据。" : "同一时间段反复研究不能作为新的独立证据。"}当前报告仅支持研究，不自动下单。${typeof report.provenance?.run_id === "string" ? ` 研究编号：${report.provenance.run_id}` : ""}`;
  reportEl("eventDownloads").hidden = false;
  drawEventEquity(report);
}

window.addEventListener("resize", () => { if (currentEventReport) drawEventEquity(currentEventReport); });
fetch("research/event-results.json", { cache: "no-store", signal: AbortSignal.timeout(10000) })
  .then(response => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); })
  .then(renderEventResearch)
  .catch(error => {
    console.warn("Event research report unavailable", error);
    currentEventReport = null;
    reportEl("eventTitle").textContent = "事件合约回测报告暂不可用";
    reportEl("eventBadge").textContent = "未验证";
    reportEl("eventNote").textContent = "尚未读取到有效回测。请运行 research/backtest.py 生成报告；失败或缺失的报告不会显示盈利结论。";
    reportEl("eventDownloads").hidden = true;
  });
