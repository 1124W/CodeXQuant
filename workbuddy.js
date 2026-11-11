/* Shared, dependency-free rendering for the WorkBuddy event research report. */
(() => {
  "use strict";
  const root = document.getElementById("workbuddyResearch");
  if (!root) return;
  const el = id => document.getElementById(id);
  const put = (id, value) => { const node = el(id); if (node) node.textContent = value; };
  const finite = value => typeof value === "number" && Number.isFinite(value);
  const fmt = value => finite(value) ? value.toLocaleString("zh-CN", {minimumFractionDigits: 2, maximumFractionDigits: 2}) : "—";
  const pct = value => finite(value) ? fmt(value * 100) + "%" : "无样本";
  const signed = value => finite(value) ? (value > 0 ? "+" : "") + fmt(value) : "—";
  const count = value => Number.isInteger(value) ? value.toLocaleString("zh-CN") : "—";
  const interval = (low, high) => finite(low) && finite(high) ? pct(low) + " – " + pct(high) : "无样本";
  const day = value => typeof value === "string" ? value.slice(0, 10) : "—";
  let report = null;

  function summaryMatches(stats, trades, label) {
    if (!stats || !Array.isArray(trades)) throw new Error(label + "缺少统计或流水");
    let wins = 0, losses = 0, refunds = 0, net = 0;
    for (const trade of trades) {
      if (trade.outcome === "win") wins++;
      else if (trade.outcome === "loss") losses++;
      else if (trade.outcome === "refund") refunds++;
      else throw new Error("流水含未知结算结果");
      if (!finite(trade.net_units)) throw new Error("流水盈亏无效");
      net += trade.net_units;
    }
    if (![stats.wins, stats.losses, stats.refunds, stats.signals, stats.decisive_trades].every(v => Number.isInteger(v) && v >= 0)
        || stats.wins !== wins || stats.losses !== losses || stats.refunds !== refunds
        || stats.signals !== trades.length || stats.decisive_trades !== wins + losses
        || !finite(stats.net_units) || Math.abs(net - stats.net_units) > 1e-5
        || (wins + losses ? !finite(stats.win_rate) || Math.abs(stats.win_rate - wins / (wins + losses)) > 1e-10 : stats.win_rate !== null)) {
      throw new Error(label + "的汇总与逐笔流水不一致");
    }
  }

  function recentStart(r) {
    const given = r.latest_three_months_start_utc || r.latest_three_months?.start_utc;
    if (given) return Date.parse(given);
    const result = new Date(r.end_exclusive_utc + "T00:00:00Z");
    result.setUTCMonth(result.getUTCMonth() - 3);
    return result.getTime();
  }

  function validate(r) {
    if (!r || r.engine !== "quantdinger-workbuddy-v1" || !r.config || !Array.isArray(r.trade_ledger)) throw new Error("不是有效的 WorkBuddy 三年报告");
    const c = r.config;
    if (c.horizon_minutes !== 30 || !finite(c.payout) || Math.abs(c.payout - .85) > 1e-12
        || !Array.isArray(r.symbols) || [...r.symbols].sort().join(",") !== "BTCUSDT,ETHUSDT") {
      throw new Error("报告必须仅包含 BTC/ETH 的 30 分钟事件与 85% 净赔付");
    }
    if (!finite(c.stake) || c.stake <= 0 || !finite(c.fee) || c.fee < 0 || c.fee >= c.payout) throw new Error("仓位或研究成本无效");
    const start = Date.parse(r.oos_start_utc || r.test_start_utc), end = Date.parse(r.end_exclusive_utc);
    if (!finite(start) || !finite(end) || start >= end) throw new Error("样本外日期无效");
    let previous = -Infinity;
    for (const trade of r.trade_ledger) {
      if (!r.symbols.includes(trade.symbol) || ![1, -1].includes(trade.direction)
          || ![trade.signal_ms, trade.entry_ms, trade.settlement_ms].every(Number.isInteger)
          || trade.signal_ms < start || trade.settlement_ms >= end || trade.signal_ms >= trade.entry_ms
          || trade.entry_ms + 30 * 60000 !== trade.settlement_ms || trade.settlement_ms < previous
          || !finite(trade.entry) || !finite(trade.exit) || trade.entry <= 0 || trade.exit <= 0) throw new Error("流水时间、品种或到期价不合规则");
      if (trade.horizon_minutes !== undefined && trade.horizon_minutes !== 30) throw new Error("流水包含非 30 分钟事件");
      const change = (trade.exit - trade.entry) * trade.direction;
      const outcome = change > 0 ? "win" : change < 0 ? "loss" : "refund";
      const expected = c.stake * ((outcome === "win" ? c.payout : outcome === "loss" ? -1 : 0) - c.fee);
      if (trade.outcome !== outcome || !finite(trade.net_units) || Math.abs(trade.net_units - expected) > 1e-7) throw new Error("流水赔付或成本不一致");
      previous = trade.settlement_ms;
    }
    summaryMatches(r.test, r.trade_ledger, "滚动样本外");
    const latest = r.latest_three_months?.stats || r.latest_three_months;
    if (latest) summaryMatches(latest, r.trade_ledger.filter(t => t.signal_ms >= recentStart(r)), "最近三个月");
    for (const symbol of r.symbols) if (r.by_symbol?.[symbol]) summaryMatches(r.by_symbol[symbol], r.trade_ledger.filter(t => t.symbol === symbol), symbol);
    for (const fold of r.folds || []) {
      const latest = fold.train_latest_settlement_ms ?? fold.latest_training_settlement_ms;
      if (finite(latest) && latest >= Date.parse(fold.test_start_utc)) throw new Error("训练标签越过预测边界");
    }
    return r;
  }

  function color(id, value) {
    const node = el(id);
    if (!node) return;
    node.classList.remove("wb-positive", "wb-negative");
    if (finite(value)) node.classList.add(value >= 0 ? "wb-positive" : "wb-negative");
  }

  function table(id, values, renderRow) {
    const target = el(id);
    if (!target) return;
    target.replaceChildren();
    for (const item of values) {
      const tr = document.createElement("tr");
      for (const value of renderRow(item)) {
        const td = document.createElement("td");
        td.textContent = value;
        tr.appendChild(td);
      }
      target.appendChild(tr);
    }
  }

  function status(r) {
    const s = r.test, target = r.target_win_rate || .58;
    const enough = s.decisive_trades >= (r.config.min_test_trades || 100);
    const reached = enough && s.win_rate >= target && s.net_units > 0;
    put("wbStatus", reached ? "历史样本外达到数值目标" : s.net_units > 0 ? "历史样本外盈利，胜率目标仍需检验" : "回测已完成，尚未达到盈利目标");
    put("wbOutcome", "滚动样本外 " + pct(s.win_rate) + "，净收益 " + signed(s.net_units) + " USDT。" +
      (reached ? "统计置信度和成本压力仍需结合下方结果判断。" : "58% 胜率与正收益分别核验，不因赔付修正而降低胜率目标。"));
    const badge = el("wbStatus");
    if (badge) badge.className = "wb-badge " + (reached ? "wb-positive" : "wb-pending");
  }

  function render(r) {
    validate(r);
    report = r;
    const s = r.test, c = r.config, recent = r.latest_three_months?.stats || r.latest_three_months;
    status(r);
    put("wbCoverage", day(r.data_start_utc) + " → " + day(r.end_exclusive_utc));
    put("wbOosCoverage", day(r.oos_start_utc || r.test_start_utc) + " → " + day(r.end_exclusive_utc));
    put("wbWarmup", "前 " + (c.train_days || 240) + " 天用于初始训练，不计入样本外收益；随后每 " + (c.test_days || 30) + " 天重训。");
    put("wbWinRate", pct(s.win_rate));
    put("wbWinDetail", count(s.wins) + " 胜 / " + count(s.losses) + " 负 / " + count(s.refunds) + " 退款");
    put("wbNet", signed(s.net_units)); color("wbNet", s.net_units);
    put("wbTrades", count(s.signals));
    put("wbDrawdown", fmt(s.max_drawdown_units));
    put("wbRecentRate", recent ? pct(recent.win_rate) : "待生成");
    put("wbRecentDetail", recent ? count(recent.signals) + " 笔 · " + signed(recent.net_units) + " USDT" : "最近三个月单独汇总");
    put("wbRecentNet", recent ? signed(recent.net_units) : "—");
    put("wbRecentDates", day(new Date(recentStart(r)).toISOString()) + " → " + day(r.end_exclusive_utc));
    put("wbBreakEven", pct(s.effective_break_even_win_rate ?? (1 + c.fee) / (1 + c.payout)));
    put("wbPayout", pct(c.payout)); put("wbFriction", c.fee === 0 ? "0.00%（基准无额外手续费）" : pct(c.fee) + "（研究压力情景）");
    put("wbStake", fmt(c.stake)); put("wbCapital", fmt(c.initial_bankroll ?? c.bankroll));
    put("wbDailyRisk", fmt(c.daily_loss_limit));
    put("wbMargin", fmt(r.strategy?.margin ?? .12));
    put("wbTrendBTC", String(r.strategy?.min_abs_ret_bps?.BTCUSDT ?? 20));
    put("wbTrendETH", String(r.strategy?.min_abs_ret_bps?.ETHUSDT ?? 28));
    put("wbWilson", interval(s.win_rate_lower_95, s.win_rate_upper_95));
    put("wbBootstrap", interval(s.block_bootstrap_lower_95, s.block_bootstrap_upper_95));
    put("wbNetInterval", finite(s.block_bootstrap_net_lower_95) ? signed(s.block_bootstrap_net_lower_95) + " 至 " + signed(s.block_bootstrap_net_upper_95) + " USDT" : "无样本");
    table("wbSymbolRows", Object.entries(r.by_symbol || {}), ([symbol, a]) => [symbol.replace("USDT", ""), count(a.signals), count(a.wins) + " / " + count(a.losses), pct(a.win_rate), signed(a.net_units), fmt(a.max_drawdown_units)]);
    table("wbYearRows", r.yearly || [], a => [a.year, count(a.signals), pct(a.win_rate), signed(a.net_units), fmt(a.max_drawdown_units)]);
    table("wbMonthRows", r.monthly || [], a => [a.month, count(a.signals), pct(a.win_rate), signed(a.net_units), fmt(a.max_drawdown_units)]);
    const scenarioNames = {base: c.fee === 0 ? "85% 基准 · 无额外手续费" : "85% 研究摩擦情景", payout_080: "净赔付降至 80%", payout_075: "净赔付降至 75%", payout_070: "净赔付降至 70%", friction_001: "额外摩擦 1% 压力", friction_002: "额外摩擦 2% 压力", adverse_strike_2bps: "入场价不利 2bp", no_friction: "无额外手续费对照"};
    table("wbStressRows", r.cost_scenarios || [], a => { const v = a.test || a.stats || a; return [scenarioNames[a.name] || a.name, pct(a.payout), pct(a.fee_per_stake ?? a.fee), fmt(a.adverse_bps) + " bp", count(v.signals), pct(v.win_rate), signed(v.net_units)]; });
    table("wbFoldRows", r.folds || [], a => [String(a.symbol || "").replace("USDT", ""), day(a.train_start_utc), day(a.test_start_utc), day(a.test_end_exclusive_utc || a.test_end_utc), count(a.n_train), count(a.n_test)]);
    table("wbProfileRows", r.profile_comparison || [], a => [a.profile === "research" ? "原仓库研究配置" : "原仓库线上配置", a.feature_count, a.train_days + " 天", day(a.oos_start_utc), count(a.test.signals), pct(a.test.win_rate), signed(a.test.net_units)]);
    if (el("wbProfileComparison")) el("wbProfileComparison").hidden = !(r.profile_comparison?.length);
    const checkNames = {only_btc_eth_30m: "仅 BTC/ETH 30 分钟事件", training_labels_settled_before_fit: "训练标签在预测前已结算", min_test_samples: "样本数量充足", target_win_rate: "胜率达到 58%", positive_net: "净收益为正", wilson_above_break_even: "Wilson 下限高于保本线", block_win_rate_above_break_even: "按日重采样胜率下限高于保本线", block_net_positive: "按日重采样净收益下限为正", no_future_training_labels: "训练标签在预测前已结算", complete_data: "分钟数据连续完整"};
    table("wbCheckRows", Object.entries(r.validation_checks || {}), ([key, passed]) => [checkNames[key] || key, passed === true ? "通过" : "未通过"]);
    const p = r.provenance || {}, upstream = r.upstream || p.upstream || {}, reference = r.reference || p.reference || {};
    put("wbProvenance", "运行 ID " + (p.run_id || "—") + " · QuantDinger " + String(upstream.commit || p.upstream_commit || "固定版本").slice(0, 12) + " · events-workBuddy " + String(reference.commit || p.reference_commit || "固定规则").slice(0, 12) + " · " + (p.generated_at_utc || ""));
    put("wbReportSummary", (c.model_profile === "research" ? "原仓库研究配置" : "原仓库线上配置") + " · GBM 逆势过滤 · " + (r.model_metrics?.BTCUSDT?.feature_count || "固定") + " 个特征 · 30 分钟到期 · 净赔付 " + pct(c.payout));
    if (el("wbContent")) el("wbContent").hidden = false;
    if (el("wbLoading")) el("wbLoading").hidden = true;
    if (el("wbError")) el("wbError").hidden = true;
    for (const id of ["wbDownloadJson", "wbDownloadCsv"]) if (el(id)) el(id).disabled = false;
    drawEquity(r);
  }

  function drawEquity(r) {
    const canvas = el("wbEquity");
    if (!canvas) return;
    const box = canvas.getBoundingClientRect();
    if (!box.width || !box.height) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2), width = box.width, height = box.height;
    canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr);
    const initial = r.config.initial_bankroll ?? r.config.bankroll, start = Date.parse(r.oos_start_utc || r.test_start_utc), end = Date.parse(r.end_exclusive_utc);
    const curve = r.equity_curve || [];
    const points = [{timestamp_ms: start, balance_units: initial}, ...curve];
    const final = points[points.length - 1].balance_units;
    points.push({timestamp_ms: end, balance_units: final});
    let low = initial, high = initial;
    for (const point of points) { low = Math.min(low, point.balance_units); high = Math.max(high, point.balance_units); }
    const padding = Math.max((high - low) * .12, 1); low -= padding; high += padding;
    const left = 64, right = 18, top = 15, bottom = 30;
    const x = t => left + (t - start) / (end - start) * (width - left - right);
    const y = value => top + (high - value) / (high - low) * (height - top - bottom);
    const dark = root.getAttribute("data-wb-theme") === "dark";
    ctx.clearRect(0, 0, width, height); ctx.font = '10px "Segoe UI","Microsoft YaHei",sans-serif'; ctx.textBaseline = "middle";
    for (let i = 0; i < 5; i++) {
      const value = low + (high - low) * i / 4, yy = y(value);
      ctx.beginPath(); ctx.strokeStyle = dark ? "rgba(255,255,255,.08)" : "#e6ecef"; ctx.moveTo(left, yy); ctx.lineTo(width - right, yy); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillStyle = dark ? "#8991a3" : "#647480"; ctx.fillText(value.toFixed(1), left - 8, yy);
    }
    ctx.setLineDash([4, 4]); ctx.strokeStyle = dark ? "#69788c" : "#aab8c1"; ctx.beginPath(); ctx.moveTo(left, y(initial)); ctx.lineTo(width - right, y(initial)); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = r.test.net_units >= 0 ? (dark ? "#b8f348" : "#0d765b") : (dark ? "#ff6b78" : "#b43e4a"); ctx.lineWidth = 1.7; ctx.beginPath();
    for (let i = 0; i < points.length; i++) { const point = points[i]; if (i === 0) ctx.moveTo(x(point.timestamp_ms), y(point.balance_units)); else { ctx.lineTo(x(point.timestamp_ms), y(points[i - 1].balance_units)); ctx.lineTo(x(point.timestamp_ms), y(point.balance_units)); } }
    ctx.stroke(); ctx.fillStyle = dark ? "#8991a3" : "#647480"; ctx.textAlign = "left"; ctx.fillText(day(r.oos_start_utc || r.test_start_utc), left, height - 11); ctx.textAlign = "right"; ctx.fillText(day(r.end_exclusive_utc), width - right, height - 11);
    put("wbEquityRange", fmt(initial) + " → " + fmt(final) + " USDT");
  }

  async function load() {
    const refresh = el("wbRefresh"); if (refresh) refresh.disabled = true;
    const urls = [];
    if (root.getAttribute("data-wb-api") === "true") urls.push(new URL("/api/event-contracts/report", location.href));
    urls.push(new URL(root.getAttribute("data-wb-report") || "research/workbuddy-results.json", location.href));
    let error;
    for (const url of urls) {
      try {
        const response = await fetch(url, {cache: "no-store", signal: AbortSignal.timeout(15000)});
        if (!response.ok) throw new Error("报告返回 HTTP " + response.status);
        const payload = await response.json();
        if (payload.code !== undefined && payload.code !== 1) throw new Error(payload.msg || "报告接口不可用");
        render(payload.code === 1 ? payload.data : payload);
        error = null; break;
      } catch (failure) { error = failure; }
    }
    if (error) {
      put("wbStatus", "报告尚未就绪");
      put("wbOutcome", "三年回测完成后自动展示真实结果。");
      put("wbError", "暂时无法展示研究报告：" + error.message);
      if (el("wbError")) el("wbError").hidden = false;
      if (el("wbLoading")) el("wbLoading").hidden = true;
      if (el("wbContent")) el("wbContent").hidden = true;
    }
    if (refresh) refresh.disabled = false;
  }

  function download(content, filename, type) {
    const url = URL.createObjectURL(new Blob([content], {type}));
    const a = document.createElement("a"); a.href = url; a.download = filename; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  if (el("wbRefresh")) el("wbRefresh").addEventListener("click", load);
  if (el("wbDownloadJson")) el("wbDownloadJson").addEventListener("click", () => { if (report) download(JSON.stringify(report, null, 2), "workbuddy-three-year-report.json", "application/json"); });
  if (el("wbDownloadCsv")) el("wbDownloadCsv").addEventListener("click", () => {
    if (!report) return;
    const fields = ["run_id", "symbol", "signal_ms", "entry_ms", "settlement_ms", "direction", "entry", "exit", "outcome", "stake", "net_units"];
    const quote = value => '"' + String(value ?? "").replace(/"/g, '""') + '"';
    const lines = [fields.join(","), ...report.trade_ledger.map(trade => fields.map(key => quote(key === "run_id" ? report.provenance.run_id : trade[key])).join(","))];
    download(lines.join("\r\n") + "\r\n", "workbuddy-three-year-trades.csv", "text/csv;charset=utf-8");
  });
  let resize;
  window.addEventListener("resize", () => { clearTimeout(resize); resize = setTimeout(() => { if (report) drawEquity(report); }, 100); });
  window.WorkBuddyReport = {validate, render, load};
  load();
})();
