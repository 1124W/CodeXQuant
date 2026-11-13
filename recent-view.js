/* Recent-period research replay. No order, notification or server-setting API. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root.document) api.mount(root.document);
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const MINUTE = 60000, DAY = 86400000, PAGE_SIZE = 40;
  const STORAGE_KEY = 'codexquant.recent.stake';
  const SYMBOLS = ['BTCUSDT', 'ETHUSDT'];
  const LABELS = { win: '盈利', loss: '亏损', refund: '平价退款' };
  const assert = (ok, message) => { if (!ok) throw new Error(message); };
  const percent = value => value == null ? '—（无已决交易）' : `${(value * 100).toFixed(2)}%`;
  const money = value => `${value > 0 ? '+' : ''}${value.toFixed(2)}`;
  const plainMoney = value => value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const side = trade => trade.direction === 1 ? '看涨 / 多' : '看跌 / 空';
  function cents(value, label = '金额') {
    assert(typeof value === 'number' || typeof value === 'string', `${label}格式无效`);
    const text = String(value).trim();
    assert(/^\d+(\.\d{1,2})?$/.test(text), `${label}须填写数字，最多两位小数`);
    const [whole, fraction = ''] = text.split('.');
    const result = Number(whole) * 100 + Number(fraction.padEnd(2, '0'));
    assert(Number.isSafeInteger(result) && result > 0, `${label}格式无效`);
    return result;
  }
  function parseStake(value) {
    const result = cents(value, '每笔金额');
    assert(result >= 500 && result <= 25000, '每笔金额必须在 5.00–250.00 USDT 之间');
    return result;
  }
  function comparePrices(a, b) {
    const parts = value => {
      assert(typeof value === 'string' && /^\d+(\.\d+)?$/.test(value), '原始价格格式无效');
      const [whole, fraction = ''] = value.split('.');
      return { value: BigInt(whole + fraction), decimals: fraction.length };
    };
    const x = parts(a), y = parts(b), scale = Math.max(x.decimals, y.decimals);
    const left = x.value * 10n ** BigInt(scale - x.decimals), right = y.value * 10n ** BigInt(scale - y.decimals);
    return left > right ? 1 : left < right ? -1 : 0;
  }
  function dateText(ms, timezone = 'UTC') {
    assert(timezone === 'UTC' || timezone === 'Asia/Shanghai', '时区无效');
    return new Date(ms + (timezone === 'Asia/Shanghai' ? 28800000 : 0)).toISOString().slice(0, 19).replace('T', ' ');
  }
  function budgetCents(assumptions, stakeCents) {
    return assumptions.daily_risk_mode === 'fixed' ? cents(assumptions.daily_risk_limit, '每日风险上限') : stakeCents * assumptions.daily_risk_multiplier;
  }
  function simulate(data, periodId, stake) {
    const a = data.assumptions, stakeCents = parseStake(stake), initial = cents(a.initial_capital, '初始本金');
    const period = data.periods.find(row => row.id === periodId);
    assert(period, '研究区间不存在');
    const budget = budgetCents(a, stakeCents);
    assert(Number.isSafeInteger(budget) && budget > 0, '每日风险上限无效');
    const raw = data.opportunities.filter(row => row.signal_ms >= period.start_ms && row.settlement_ms < period.end_exclusive_ms)
      .sort((x, y) => x.signal_ms - y.signal_ms || SYMBOLS.indexOf(x.symbol) - SYMBOLS.indexOf(y.symbol));
    const dailyLoss = new Map(), accepted = [];
    let pending = [], balance = initial, skippedOverlap = 0, skippedDaily = 0, skippedFunds = 0;
    for (const source of raw) {
      const remains = [];
      for (const trade of pending) {
        if (trade.settlement_ms < source.signal_ms) {
          balance += trade._net;
          if (trade.outcome === 'loss') {
            const day = Math.floor(trade.settlement_ms / DAY);
            dailyLoss.set(day, (dailyLoss.get(day) || 0) + stakeCents);
          }
        } else remains.push(trade);
      }
      pending = remains;
      if (pending.some(trade => trade.settlement_ms > source.entry_ms)) { skippedOverlap++; continue; }
      const reserved = pending.length * stakeCents, day = Math.floor(source.entry_ms / DAY);
      if ((dailyLoss.get(day) || 0) + reserved + stakeCents > budget) { skippedDaily++; continue; }
      const available = balance - reserved;
      if (stakeCents > available) { skippedFunds++; continue; }
      const net = source.outcome === 'win' ? Math.floor((stakeCents * 85 + 50) / 100) : source.outcome === 'loss' ? -stakeCents : 0;
      const trade = { ...source, opportunity_id: source.id, id: accepted.length + 1, stake: stakeCents / 100,
        net_units: net / 100, equity_at_signal: balance / 100, available_at_signal: available / 100,
        daily_budget_usdt: budget / 100, _net: net };
      accepted.push(trade); pending.push(trade);
    }
    accepted.sort((x, y) => x.settlement_ms - y.settlement_ms || x.symbol.localeCompare(y.symbol));
    let equity = initial, peak = initial, minimum = initial, drawdown = 0, maxPercent = 0, losing = 0, longest = 0;
    let wins = 0, losses = 0, refunds = 0;
    const curve = [{ time_ms: period.start_ms, equity: initial / 100, drawdown_usdt: 0 }];
    for (const trade of accepted) {
      equity += trade._net; peak = Math.max(peak, equity); minimum = Math.min(minimum, equity);
      drawdown = Math.max(drawdown, peak - equity); maxPercent = Math.max(maxPercent, (peak - equity) / peak);
      if (trade.outcome === 'win') wins++; else if (trade.outcome === 'loss') losses++; else refunds++;
      losing = trade.outcome === 'loss' ? losing + 1 : 0; longest = Math.max(longest, losing);
      trade.balance_after_settlement = equity / 100;
      curve.push({ time_ms: trade.settlement_ms, equity: equity / 100, drawdown_usdt: (peak - equity) / 100, trade });
      delete trade._net;
    }
    return { trades: accepted, equity_curve: curve, stats: {
      signals: accepted.length, wins, losses, refunds, decisive_trades: wins + losses,
      win_rate: wins + losses ? wins / (wins + losses) : null,
      net_usdt: (equity - initial) / 100, ending_balance: equity / 100,
      max_drawdown_usdt: drawdown / 100, max_drawdown_pct: maxPercent, minimum_balance: minimum / 100,
      longest_losing_streak: longest, raw_opportunities: raw.length,
      skipped_overlap: skippedOverlap, skipped_daily_limit: skippedDaily, skipped_insufficient: skippedFunds
    } };
  }
  function validate(data) {
    assert(data && data.schema_version === 1 && Array.isArray(data.periods) && data.periods.length === 2, '研究数据版本或区间无效');
    const a = data.assumptions;
    assert(a && a.default_stake === 30 && a.min_stake === 5 && a.max_stake === 250 && a.stake_step === .01 &&
      a.net_payout === .85 && a.fee === 0 && a.horizon_minutes === 30 && a.entry_delay_minutes === 1 &&
      a.max_open_positions === 1 && a.payout_rounding === 'half_up_cent', '研究金额与合约口径不一致');
    cents(a.initial_capital, '初始本金');
    assert(a.daily_risk_mode === 'fixed' || (!a.daily_risk_mode || a.daily_risk_mode === 'stake_multiple') && a.daily_risk_multiplier === 5, '日风险规则不支持');
    const expected = { year_to_date: ['2026-01-01', '2026-11-12'], recent_six_months: ['2026-05-12', '2026-11-12'] };
    assert(new Set(data.periods.map(row => row.id)).size === 2, '研究区间重复');
    for (const p of data.periods) {
      const dates = expected[p.id];
      assert(dates && p.start_utc === dates[0] && p.end_exclusive_utc === dates[1] && p.start_ms === Date.parse(`${dates[0]}T00:00:00Z`) &&
        p.end_exclusive_ms === Date.parse(`${dates[1]}T00:00:00Z`), '研究区间边界不一致');
    }
    assert(data.strategy && data.strategy.is_live_strategy === false && data.strategy.posthoc === true, '研究候选标识缺失');
    assert(Array.isArray(data.opportunities) && data.opportunities.length > 0 && Array.isArray(data.limitations), '原始候选或研究说明缺失');
    const seen = new Set(); let previous = null;
    const year = data.periods.find(p => p.id === 'year_to_date');
    const unavailable = (a.known_unavailable_dates_utc || []).map(value => Date.parse(`${value}T00:00:00Z`));
    assert(unavailable.every(Number.isFinite), '排除日期格式无效');
    for (const [i, row] of data.opportunities.entries()) {
      assert(row.id === i + 1 && SYMBOLS.includes(row.symbol) && [1, -1].includes(row.direction), '原始候选身份无效');
      const key = `${row.symbol}:${row.signal_ms}`;
      assert(!seen.has(key), '原始候选重复'); seen.add(key);
      assert([row.signal_ms, row.entry_ms, row.settlement_ms].every(Number.isSafeInteger) && row.signal_ms % MINUTE === 0 &&
        row.entry_ms === row.signal_ms + MINUTE && row.settlement_ms === row.entry_ms + 30 * MINUTE &&
        row.signal_ms >= year.start_ms && row.settlement_ms < year.end_exclusive_ms, '原始候选时间无效');
      assert(!previous || row.signal_ms > previous.signal_ms || row.signal_ms === previous.signal_ms && row.symbol > previous.symbol, '原始候选排序无效');
      previous = row;
      assert(!unavailable.some(day => row.signal_ms < day + DAY && row.settlement_ms >= day), '原始候选触及已排除日期');
      assert([row.entry, row.exit].every(value => Number.isFinite(value) && value > 0) && row.entry === Number(row.entry_exact) && row.exit === Number(row.exit_exact), '原始指数价格无效');
      const change = comparePrices(row.exit_exact, row.entry_exact) * row.direction;
      assert(row.outcome === (change > 0 ? 'win' : change < 0 ? 'loss' : 'refund'), '候选胜负与原始指数不一致');
    }
    for (const p of data.periods) {
      const replayed = simulate(data, p.id, a.default_stake), published = data.published && data.published[p.id];
      assert(published && published.stats && Array.isArray(published.trades), '默认 30U 发布结果缺失');
      for (const key of Object.keys(replayed.stats)) {
        const actual = replayed.stats[key], reference = published.stats[key];
        assert(actual === reference || typeof actual === 'number' && typeof reference === 'number' && Math.abs(actual - reference) < 1e-10, `默认 30U 统计校验失败：${key}`);
      }
      assert(replayed.trades.length === published.trades.length, '默认 30U 账本长度不一致');
      replayed.trades.forEach((trade, index) => {
        const publishedTrade = published.trades[index];
        for (const key of ['opportunity_id', 'symbol', 'direction', 'signal_ms', 'entry_ms', 'settlement_ms', 'outcome', 'stake', 'net_units', 'equity_at_signal', 'available_at_signal', 'daily_budget_usdt', 'balance_after_settlement']) {
          assert(trade[key] === publishedTrade[key], `默认 30U 账本校验失败：${key}`);
        }
      });
      const intervals = [published.weekly_interval, ...(published.stress || []).map(row => row.weekly_interval)];
      assert(Array.isArray(published.stress) && published.stress.length === 3 &&
        published.stress.every((row, i) => row.adverse_bps === [.5, 1, 2][i] && row.stats &&
          Number.isFinite(row.stats.net_usdt) && Number.isInteger(row.stats.signals) && row.stats.win_rate >= 0 && row.stats.win_rate <= 1), '默认金额压力结果缺失');
      assert(intervals.every(row => row && row.block_days === 7 &&
        ['net_lower_95', 'net_upper_95', 'win_rate_lower_95', 'win_rate_upper_95'].every(key => Number.isFinite(row[key]))), '默认金额区间估计缺失');
      assert(Array.isArray(published.monthly) && published.monthly.reduce((n, row) => n + row.signals, 0) === published.stats.signals &&
        published.monthly.reduce((n, row) => n + Math.round(row.net_usdt * 100), 0) === Math.round(published.stats.net_usdt * 100), '默认金额月份结果不一致');
    }
    return data;
  }
  function tradeSummary(trades) {
    const wins = trades.filter(t => t.outcome === 'win').length, losses = trades.filter(t => t.outcome === 'loss').length;
    return { count: trades.length, wins, losses, win_rate: wins + losses ? wins / (wins + losses) : null,
      net_usdt: trades.reduce((sum, t) => sum + Math.round(t.net_units * 100), 0) / 100 };
  }
  function csvForTrades(trades, period, amount) {
    const fields = ['id', 'opportunity_id', 'symbol', 'direction', 'signal_utc', 'entry_utc', 'expiry_utc', 'signal_ms', 'entry_ms', 'settlement_ms',
      'entry_exact', 'exit_exact', 'outcome', 'stake', 'net_units', 'balance_after_settlement', 'period', 'configured_stake'];
    const quote = value => `"${String(value ?? '').replace(/"/g, '""')}"`;
    const rows = trades.map(t => ({ ...t, signal_utc: dateText(t.signal_ms), entry_utc: dateText(t.entry_ms), expiry_utc: dateText(t.settlement_ms), period, configured_stake: amount }));
    return '\uFEFF' + [fields.map(quote).join(','), ...rows.map(row => fields.map(key => quote(row[key])).join(','))].join('\r\n') + '\r\n';
  }
  function savedStake(storage, fallback = 30) {
    try { const value = storage && storage.getItem(STORAGE_KEY); return value == null ? fallback : parseStake(value) / 100; }
    catch (_) { return fallback; }
  }
  function mount(doc) {
    const view = doc.getElementById('recentView'); if (!view) return;
    const $ = name => doc.getElementById(name), win = doc.defaultView || globalThis;
    const make = (tag, text, className) => { const node = doc.createElement(tag); if (text != null) node.textContent = text; if (className) node.className = className; return node; };
    const cells = (row, values) => values.forEach(value => row.append(make('td', value)));
    const svgEl = (tag, attrs, text) => { const node = doc.createElementNS('http://www.w3.org/2000/svg', tag); Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value)); if (text != null) node.textContent = text; return node; };
    let data, amount = 30, results = {}, filtered = [], selected = 0, page = 0, chart = null;
    const timezone = () => $('recentTimezone').value;
    const period = () => data.periods.find(p => p.id === $('recentPeriod').value);
    function cards(target, items) {
      $(target).replaceChildren(...items.map(([name, value, note, cls]) => {
        const card = make('article'); card.append(make('span', name), make('strong', value, cls), make('small', note)); return card;
      }));
    }
    function drawChart() {
      const svg = $('recentEquity'); svg.replaceChildren(); chart = null;
      if (!filtered.length) { svg.append(svgEl('text', { x: 500, y: 145, fill: '#8993a8', 'text-anchor': 'middle', 'font-size': 18 }, '当前区间或币种没有准入交易')); return; }
      let equityCents = cents(data.assumptions.initial_capital);
      const points = filtered.map(trade => ({ time: trade.settlement_ms, value: (equityCents += Math.round(trade.net_units * 100)) / 100, trade }));
      const initial = data.assumptions.initial_capital, first = period().start_ms, end = period().end_exclusive_ms;
      const lo = Math.min(initial, ...points.map(p => p.value)), hi = Math.max(initial, ...points.map(p => p.value));
      const pad = Math.max(10, (hi - lo) * .12), low = lo - pad, high = hi + pad;
      const x = time => 86 + (time - first) / (end - first) * 880, y = value => 239 - (value - low) / (high - low) * 207;
      for (let n = 0; n < 5; n++) {
        const value = low + (high - low) * n / 4, yy = y(value);
        svg.append(svgEl('line', { x1: 86, x2: 966, y1: yy, y2: yy, stroke: '#202739' }));
        svg.append(svgEl('text', { x: 76, y: yy + 4, fill: '#8993a8', 'text-anchor': 'end', 'font-size': 12 }, value.toFixed(0)));
      }
      svg.append(svgEl('line', { x1: 86, x2: 966, y1: y(initial), y2: y(initial), stroke: '#8993a8', 'stroke-dasharray': '5 5', opacity: .6 }));
      const path = [`M86 ${y(initial)}`, ...points.map(p => `H${x(p.time).toFixed(2)} V${y(p.value).toFixed(2)}`), 'H966'].join(' ');
      svg.append(svgEl('path', { d: path, fill: 'none', stroke: '#57d8ff', 'stroke-width': 2, 'stroke-linejoin': 'round' }));
      svg.append(svgEl('text', { x: 86, y: 270, fill: '#8993a8', 'font-size': 12 }, dateText(first, timezone()).slice(0, 10)));
      svg.append(svgEl('text', { x: 966, y: 270, fill: '#8993a8', 'text-anchor': 'end', 'font-size': 12 }, `${dateText(end, timezone()).slice(0, 10)} · ${timezone() === 'UTC' ? 'UTC' : 'UTC+8'}`));
      const cursor = svgEl('circle', { r: 5, fill: '#c6ff67', stroke: '#0a0d14', 'stroke-width': 2 }); svg.append(cursor);
      chart = { points, x, y, cursor };
    }
    function selectTrade(index) {
      selected = Math.max(0, Math.min(index, filtered.length - 1));
      const trade = filtered[selected], box = $('recentSelected'); box.replaceChildren();
      if (!trade) { box.append(make('p', '没有符合当前筛选的交易。', 'note')); return; }
      box.append(make('h3', `#${trade.id} · ${trade.symbol.replace('USDT', '')} ${side(trade)} · ${LABELS[trade.outcome]} ${money(trade.net_units)} USDT`));
      const grid = make('div', null, 'detail-grid');
      for (const [label, value] of [['信号时间', dateText(trade.signal_ms, timezone())], ['开仓时间 · 信号后 1 分钟', dateText(trade.entry_ms, timezone())],
        ['到期时间 · 持有 30 分钟', dateText(trade.settlement_ms, timezone())], ['开仓指数 · 原始精度', trade.entry_exact], ['到期指数 · 原始精度', trade.exit_exact],
        ['每笔本金 / 本笔净收益', `${plainMoney(trade.stake)} / ${money(trade.net_units)} USDT`]]) {
        const cell = make('div'); cell.append(make('span', label), make('strong', value)); grid.append(cell);
      }
      box.append(grid);
      if (chart) { chart.cursor.setAttribute('cx', chart.x(chart.points[selected].time)); chart.cursor.setAttribute('cy', chart.y(chart.points[selected].value)); }
      for (const row of $('recentTradeRows').children) row.classList.toggle('is-selected', row.getAttribute('data-trade-id') === String(trade.id));
    }
    function ledger() {
      const maxPage = Math.max(0, Math.ceil(filtered.length / PAGE_SIZE) - 1); page = Math.min(page, maxPage);
      $('recentTradeRows').replaceChildren(...filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE).map((trade, offset) => {
        const row = make('tr'); row.setAttribute('data-trade-id', trade.id);
        const first = make('td'), button = make('button', `#${trade.id}`); button.type = 'button'; button.setAttribute('aria-label', `查看第 ${trade.id} 笔 ${trade.symbol.replace('USDT', '')} 交易`);
        button.addEventListener('click', () => selectTrade(page * PAGE_SIZE + offset)); first.append(button); row.append(first);
        cells(row, [`${trade.symbol.replace('USDT', '')} ${side(trade)}`, dateText(trade.signal_ms, timezone()), dateText(trade.entry_ms, timezone()),
          trade.entry.toFixed(2), dateText(trade.settlement_ms, timezone()), trade.exit.toFixed(2), trade.stake.toFixed(2), LABELS[trade.outcome], money(trade.net_units)]);
        return row;
      }));
      $('recentPage').textContent = `${filtered.length ? page + 1 : 0} / ${maxPage + 1} 页 · ${filtered.length} 笔`;
      $('recentPrevious').disabled = page <= 0; $('recentNext').disabled = page >= maxPage;
      $('downloadRecent').disabled = filtered.length === 0;
      selectTrade(selected);
    }
    function renderLedger(reset = true) {
      const all = results[period().id].trades, asset = $('recentSymbol').value;
      filtered = all.filter(trade => asset === 'ALL' || trade.symbol === asset);
      if (reset) { page = 0; selected = Math.max(0, filtered.length - 1); }
      $('ledgerScope').textContent = `每笔 ${amount.toFixed(2)} USDT · ${period().label} · ${asset === 'ALL' ? '两币合计' : asset.replace('USDT', '') + '贡献'} · ${timezone() === 'UTC' ? 'UTC' : '北京时间 UTC+8'}。币种筛选只展示已准入账本子集，不单独重跑策略或风控。`;
      drawChart(); ledger();
    }
    function render() {
      const a = data.assumptions, p = period(), result = results[p.id], s = result.stats;
      const budget = budgetCents(a, parseStake(amount)) / 100;
      $('scenarioText').textContent = `当前每笔 ${amount.toFixed(2)} USDT${amount === a.default_stake ? '（默认发布口径）' : '（你在本浏览器设置的金额）'}；盈利 +${(Math.floor((parseStake(amount) * 85 + 50) / 100) / 100).toFixed(2)} USDT，亏损 −${amount.toFixed(2)} USDT。`;
      $('amountBadge').textContent = `当前 ${amount.toFixed(2)}U / 默认 ${a.default_stake.toFixed(2)}U`;
      $('riskSummary').replaceChildren(...[`初始本金 ${plainMoney(a.initial_capital)} USDT`, `每日亏损及待结算风险 ${plainMoney(budget)} USDT`, '两币共享最多 1 笔持仓', '净赔付 85% · 零额外手续费'].map(text => make('span', text)));
      $('periodText').textContent = `${p.label}：${p.start_utc} 00:00 → ${p.end_exclusive_utc} 00:00 UTC，结束时刻不含。数据截至该边界；显示时区不改变区间或每日风控的 UTC 日界。`;
      cards('recentCards', [['胜率', percent(s.win_rate), `${s.wins} 盈利 / ${s.losses} 亏损 / ${s.refunds} 退款`],
        ['准入交易', s.signals.toLocaleString('en-US'), `${s.raw_opportunities.toLocaleString('en-US')} 个原始机会`],
        ['模拟净收益 USDT', money(s.net_usdt), `期末资金 ${plainMoney(s.ending_balance)}`, s.net_usdt >= 0 ? 'positive' : 'negative'],
        ['最大回撤 USDT', s.max_drawdown_usdt.toFixed(2), `相对当时权益高点 ${percent(s.max_drawdown_pct)}`]]);
      $('admissionText').textContent = `风控跳过：持仓重叠 ${s.skipped_overlap} 次、日风险预算 ${s.skipped_daily_limit} 次、资金不足 ${s.skipped_insufficient} 次。最长连续亏损 ${s.longest_losing_streak} 笔；最低已结算资金 ${plainMoney(s.minimum_balance)} USDT。`;
      $('recentPeriodRows').replaceChildren(...data.periods.map(p => { const row = make('tr'), stat = results[p.id].stats; cells(row, [`${p.label} · ${p.start_utc} → ${p.end_exclusive_utc}`, stat.signals, percent(stat.win_rate), money(stat.net_usdt), stat.max_drawdown_usdt.toFixed(2)]); return row; }));
      cards('assetCards', SYMBOLS.map(symbol => { const stat = tradeSummary(result.trades.filter(t => t.symbol === symbol)); return [symbol.replace('USDT', '') + ' · 已准入贡献', `${percent(stat.win_rate)} / ${money(stat.net_usdt)} U`, `${stat.count} 笔 · 不单独分配一套本金`, stat.net_usdt >= 0 ? 'positive' : 'negative']; }));
      const published = data.published[p.id], interval = published.weekly_interval;
      $('uncertaintyText').textContent = `${p.label} · 默认每笔 30U：7 日块重采样胜率 95% 区间 ${percent(interval.win_rate_lower_95)}—${percent(interval.win_rate_upper_95)}；净收益区间 ${money(interval.net_lower_95)}—${money(interval.net_upper_95)} USDT。当前候选来自历史择优，未证实稳定达到 58%。`;
      $('stressRows').replaceChildren(...[{ adverse_bps: 0, stats: published.stats, weekly_interval: interval }, ...published.stress].map(row => {
        const tr = make('tr'), bounds = row.weekly_interval; cells(tr, [row.adverse_bps === 0 ? '无偏差 · 基准' : `${row.adverse_bps}bp / ${(row.adverse_bps / 100).toFixed(3)}%`, row.stats.signals, percent(row.stats.win_rate), money(row.stats.net_usdt), `${money(bounds.net_lower_95)} ～ ${money(bounds.net_upper_95)}`]); return tr;
      }));
      $('monthlyRows').replaceChildren(...published.monthly.map(row => { const tr = make('tr'); cells(tr, [row.month, row.signals, percent(row.win_rate), money(row.net_usdt)]); return tr; }));
      renderLedger();
    }
    function applyAmount(value, persist = true) {
      try {
        amount = parseStake(value) / 100;
        results = Object.fromEntries(data.periods.map(p => [p.id, simulate(data, p.id, amount)]));
        $('recentStake').value = amount.toFixed(2); $('recentStake').setAttribute('aria-invalid', 'false'); $('stakeError').hidden = true;
        if (persist) { try { win.localStorage.setItem(STORAGE_KEY, amount.toFixed(2)); } catch (_) { /* Browser storage can be unavailable. */ } }
        render(); return true;
      } catch (error) {
        $('stakeError').textContent = error.message; $('stakeError').hidden = false; $('recentStake').setAttribute('aria-invalid', 'true'); return false;
      }
    }
    $('stakeForm').addEventListener('submit', event => { event.preventDefault(); if (data) applyAmount($('recentStake').value); });
    $('resetStake').addEventListener('click', () => { if (data) applyAmount(data.assumptions.default_stake); });
    $('recentPeriod').addEventListener('change', () => { if (data) render(); });
    $('recentSymbol').addEventListener('change', () => { if (data) renderLedger(); });
    $('recentTimezone').addEventListener('change', () => { if (data) renderLedger(false); });
    $('recentPrevious').addEventListener('click', () => { if (page > 0) { page--; ledger(); } });
    $('recentNext').addEventListener('click', () => { if ((page + 1) * PAGE_SIZE < filtered.length) { page++; ledger(); } });
    $('recentEquity').addEventListener('keydown', event => {
      if (!filtered.length || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault(); selectTrade(event.key === 'Home' ? 0 : event.key === 'End' ? filtered.length - 1 : selected + (event.key === 'ArrowLeft' ? -1 : 1));
    });
    $('recentEquity').addEventListener('pointermove', event => {
      if (!chart) return;
      const rect = $('recentEquity').getBoundingClientRect(), px = (event.clientX - rect.left) / rect.width * 1000;
      let best = 0; for (let i = 1; i < chart.points.length; i++) if (Math.abs(chart.x(chart.points[i].time) - px) < Math.abs(chart.x(chart.points[best].time) - px)) best = i;
      selectTrade(best);
    });
    $('downloadRecent').addEventListener('click', () => {
      if (!filtered.length) return;
      const url = win.URL.createObjectURL(new win.Blob([csvForTrades(filtered, period().id, amount)], { type: 'text/csv;charset=utf-8' }));
      const anchor = make('a'); anchor.href = url; anchor.download = `CodeXQuant-${period().id}-${amount.toFixed(2)}U-${$('recentSymbol').value}.csv`;
      doc.body.append(anchor); anchor.click(); anchor.remove(); win.setTimeout(() => win.URL.revokeObjectURL(url), 1000);
    });
    async function load() {
      $('recentRetry').disabled = true; $('recentRetry').hidden = true; $('recentError').hidden = true; $('recentState').textContent = '正在读取并核验研究数据…';
      try {
        const fetcher = win.fetch ? win.fetch.bind(win) : globalThis.fetch;
        const response = await fetcher(view.getAttribute('data-research-url'), { cache: 'no-store', credentials: 'omit' });
        if (!response.ok) throw new Error('研究数据暂时无法读取');
        data = validate(await response.json());
        $('recentVersion').textContent = data.published_data_version || '已校验版本';
        $('strategyText').textContent = data.strategy.description || 'Logistic 与 GBM 方向一致，且双方模型方向分数均达到 0.60；分数未经胜率校准。';
        $('recentLimitations').replaceChildren(...data.limitations.map(text => make('li', text)));
        let initialAmount = data.assumptions.default_stake; try { initialAmount = savedStake(win.localStorage, initialAmount); } catch (_) { /* use published default */ }
        if (!applyAmount(initialAmount, false)) throw new Error($('stakeError').textContent);
        $('recentContent').hidden = false;
        $('recentState').textContent = `已核验默认 30U 的两组回测 · ${data.opportunities.length.toLocaleString('en-US')} 个原始候选机会 · 当前金额在本地重新模拟`;
      } catch (error) {
        data = null; $('recentContent').hidden = true; $('recentState').textContent = '研究数据未通过校验';
        $('recentError').textContent = `无法展示研究结果：${error.message}。可以重试或下载默认账本。`; $('recentError').hidden = false; $('recentRetry').hidden = false;
      } finally { $('recentRetry').disabled = false; }
    }
    $('recentRetry').addEventListener('click', load); load();
  }
  return { cents, parseStake, comparePrices, dateText, budgetCents, simulate, validate, tradeSummary, csvForTrades, savedStake, mount };
});
