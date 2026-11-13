/* Public, read-only projection of the calendar-audited historical ledger. */
(function (root, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root.document) api.mount(root.document);
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';
  const PAGE_SIZE = 50;
  const labels = { win: '盈利', loss: '亏损', refund: '持平退款' };
  const percent = n => n == null ? '—（无有效交易）' : `${(n * 100).toFixed(2)}%`;
  const money = n => `${n > 0 ? '+' : ''}${n.toFixed(2)}`;
  const side = t => t.direction === 1 ? '看涨 / 多' : '看跌 / 空';
  const symbol = t => t.symbol.replace('USDT', '');
  function assert(value, message) { if (!value) throw new Error(message); }
  function decimalParts(value) {
    assert(typeof value === 'string' && /^\d+(\.\d+)?$/.test(value), '原始价格格式无效');
    const [whole, fraction = ''] = value.split('.');
    return { units: BigInt(whole + fraction), digits: fraction.length };
  }
  function comparePrices(a, b) {
    const x = decimalParts(a), y = decimalParts(b), scale = Math.max(x.digits, y.digits);
    const left = x.units * 10n ** BigInt(scale - x.digits);
    const right = y.units * 10n ** BigInt(scale - y.digits);
    return left > right ? 1 : left < right ? -1 : 0;
  }
  function stats(trades) {
    let wins = 0, losses = 0, refunds = 0, net = 0, peak = 0, drawdown = 0;
    for (const t of [...trades].sort((a, b) => a.settlement_ms - b.settlement_ms || a.symbol.localeCompare(b.symbol))) {
      if (t.outcome === 'win') wins++; else if (t.outcome === 'loss') losses++; else refunds++;
      net += t.net_units; peak = Math.max(peak, net); drawdown = Math.max(drawdown, peak - net);
    }
    return { signals: trades.length, wins, losses, refunds, decisive_trades: wins + losses,
      win_rate: wins + losses ? wins / (wins + losses) : null, net_units: net, max_drawdown_units: drawdown };
  }
  function filterTrades(data, filters = {}) {
    const p = data.comparison.periods.find(p => p.id === (filters.period || 'continuous_post_selection'));
    assert(p, '未知研究区间');
    return data.trades.filter(t => t.signal_ms >= p.start_ms && t.settlement_ms < p.end_exclusive_ms &&
      (!filters.symbol || filters.symbol === 'ALL' || t.symbol === filters.symbol) &&
      (!filters.direction || filters.direction === 'ALL' || t.direction === Number(filters.direction)) &&
      (!filters.outcome || filters.outcome === 'ALL' || t.outcome === filters.outcome));
  }
  function dateText(ms, timezone = 'UTC') {
    assert(timezone === 'UTC' || timezone === 'Asia/Shanghai', '未知时区');
    return new Date(ms + (timezone === 'Asia/Shanghai' ? 28800000 : 0)).toISOString().slice(0, 19).replace('T', ' ');
  }
  function equityPoints(trades, initial = 1000) {
    let value = initial;
    return trades.map(t => ({ time: t.settlement_ms, value: value += t.net_units, trade: t }));
  }
  function validate(data) {
    assert(data && data.schema_version === 1 && typeof data.published_data_version === 'string', '不支持的数据版本');
    const a = data.assumptions, r = data.research, c = data.comparison;
    assert(a && a.horizon_minutes === 30 && a.entry_delay_minutes === 1 && a.net_payout === .85 &&
      a.fee === 0 && a.stake === 5 && a.initial_capital === 1000 && a.max_open_positions === 1, '合约研究口径不一致');
    assert(r && r.rounds === 8 && r.candidate_configurations === 137 && r.matched_controls === 24 &&
      r.qualified_strategies === 0 && r.forward_validated === false && Array.isArray(r.studies) && r.studies.length === 8, '研究统计不完整');
    assert(c && c.is_live_strategy === false && c.calendar_corrected === true && Array.isArray(c.periods) && c.periods.length === 3,
      '历史对照标识或区间不完整');
    assert(Array.isArray(data.trades) && data.trades.length === 1262, '逐笔数据不完整');
    assert(Array.isArray(a.known_unavailable_dates_utc) && Array.isArray(data.limitations), '研究限制缺失');
    const unavailable = a.known_unavailable_dates_utc.map(day => Date.parse(`${day}T00:00:00Z`));
    assert(unavailable.every(Number.isFinite), '排除日期格式错误');
    const seen = new Set(); let previousExpiry = 0;
    data.trades.forEach((t, i) => {
      assert(t.id === i + 1 && ['BTCUSDT', 'ETHUSDT'].includes(t.symbol) && [1, -1].includes(t.direction), '交易身份无效');
      const key = `${t.symbol}:${t.signal_ms}`;
      assert(!seen.has(key), '发现重复交易'); seen.add(key);
      assert([t.signal_ms, t.entry_ms, t.settlement_ms].every(Number.isSafeInteger) && t.signal_ms % 60000 === 0 &&
        t.entry_ms === t.signal_ms + 60000 && t.settlement_ms === t.entry_ms + 1800000 && t.entry_ms >= previousExpiry, '交易时间或共享持仓约束无效');
      previousExpiry = t.settlement_ms;
      assert(!unavailable.some(day => t.signal_ms < day + 86400000 && t.settlement_ms >= day), '交易触及已排除日期');
      assert(Number.isFinite(t.entry) && Number.isFinite(t.exit) && t.entry > 0 && t.exit > 0 &&
        t.entry === Number(t.entry_exact) && t.exit === Number(t.exit_exact), '价格与原始精度不一致');
      const move = comparePrices(t.exit_exact, t.entry_exact) * t.direction;
      assert(t.outcome === (move > 0 ? 'win' : move < 0 ? 'loss' : 'refund') &&
        t.net_units === (move > 0 ? 4.25 : move < 0 ? -5 : 0), '交易胜负或 85% 赔付不一致');
    });
    const expected = {
      validation: ['2025-05-07', '2026-08-09', 935],
      historical_confirmation: ['2026-08-09', '2026-11-12', 327],
      continuous_post_selection: ['2025-05-07', '2026-11-12', 1262]
    };
    const checkStats = (actual, published) => Object.keys(actual).forEach(k =>
      assert(actual[k] === published[k], `统计校验失败：${k}`));
    assert(new Set(c.periods.map(p => p.id)).size === 3, '区间重复');
    c.periods.forEach(p => {
      const e = expected[p.id];
      assert(e && p.start_utc === e[0] && p.end_exclusive_utc === e[1] && p.signals === e[2] &&
        p.start_ms === Date.parse(`${e[0]}T00:00:00Z`) && p.end_exclusive_ms === Date.parse(`${e[1]}T00:00:00Z`), '冻结区间已变更');
      checkStats(stats(filterTrades(data, { period: p.id })), p);
    });
    const complete = c.periods.find(p => p.id === 'continuous_post_selection');
    assert(complete.wins === 719 && complete.losses === 543 && complete.net_units === 340.75, '对照总计已变更');
    for (const asset of ['BTCUSDT', 'ETHUSDT']) checkStats(stats(data.trades.filter(t => t.symbol === asset)), c.by_symbol[asset]);
    const w = c.weekly_interval;
    assert(w && w.block_days === 7 && [w.win_rate_lower_95, w.win_rate_upper_95, w.net_lower_95, w.net_upper_95].every(Number.isFinite), '不确定性统计缺失');
    assert(r.studies.every(s => typeof s.label === 'string' && Number.isInteger(s.candidates) && Number.isInteger(s.controls) && !s.historical_gates_passed) &&
      r.studies.reduce((n, s) => n + s.candidates, 0) === 137 && r.studies.reduce((n, s) => n + s.controls, 0) === 24, '研究明细无法核对');
    return data;
  }
  function mount(doc) {
    const view = doc.getElementById('researchView'); if (!view) return;
    const $ = id => doc.getElementById(id);
    const make = (tag, content, className) => { const e = doc.createElement(tag); if (content != null) e.textContent = content; if (className) e.className = className; return e; };
    let data, filtered = [], selected = 0, page = 0, chart = null;
    const timezone = () => $('researchTimezone').value;
    const addCells = (row, values) => values.forEach(value => row.append(make('td', value)));
    function cards(target, items) {
      $(target).replaceChildren(...items.map(([label, value, note, cls]) => {
        const card = make('article'); card.append(make('span', label), make('strong', value, cls), make('small', note)); return card;
      }));
    }
    function staticContent() {
      const p = data.comparison.periods.find(p => p.id === 'continuous_post_selection'), w = data.comparison.weekly_interval;
      cards('studyCards', [['完成研究', '8 轮', '不同特征与策略方向'], ['候选配置', '137', '包含重复及相关组合'],
        ['配对对照', '24', '核对新增特征的贡献'], ['通过完整门槛', '0', '未证实稳定达到 58%']]);
      $('studyRows').replaceChildren(...data.research.studies.map(s => { const tr = make('tr'); addCells(tr, [s.label, s.candidates, s.controls, '未通过']); return tr; }));
      cards('comparisonCards', [['完整期间胜率', percent(p.win_rate), `${p.wins} 盈利 / ${p.losses} 亏损`],
        ['已准入交易', p.signals.toLocaleString('en-US'), 'BTC 459 / ETH 803'], ['模拟净收益', money(p.net_units), 'USDT · 固定每笔 5', 'positive'],
        ['最大回撤', p.max_drawdown_units.toFixed(2), 'USDT · 按结算顺序']]);
      $('assumptionText').textContent = '30 分钟事件合约 · 净赔付 85% · 无额外手续费 · 每笔本金 5 USDT · 初始资金 1,000 USDT · 共享最多 1 笔持仓 · 日亏损上限 25 USDT · 信号后 1 分钟开仓 · 盈亏平衡胜率 54.05%。';
      $('periodRows').replaceChildren(...data.comparison.periods.map(p => { const tr = make('tr'); addCells(tr, [p.label, `${p.start_utc} → ${p.end_exclusive_utc}`, p.signals, percent(p.win_rate), money(p.net_units)]); return tr; }));
      $('uncertaintyText').textContent = `7 日块重采样的 95% 区间：胜率 ${percent(w.win_rate_lower_95)}—${percent(w.win_rate_upper_95)}；净收益 ${money(w.net_lower_95)}—${money(w.net_upper_95)} USDT。收益下界仍为负，不能据此认定已找到稳定盈利策略。`;
      $('researchLimitations').replaceChildren(...data.limitations.map(s => make('li', s)));
      $('publicDataVersion').textContent = data.published_data_version;
    }
    function svgEl(tag, attributes, content) {
      const el = doc.createElementNS('http://www.w3.org/2000/svg', tag);
      Object.entries(attributes).forEach(([k, v]) => el.setAttribute(k, String(v)));
      if (content != null) el.textContent = content; return el;
    }
    function renderChart() {
      const svg = $('researchEquity'); svg.replaceChildren(); chart = null;
      if (!filtered.length) { svg.append(svgEl('text', { x: 500, y: 145, fill: '#8993a8', 'text-anchor': 'middle', 'font-size': 18 }, '当前筛选没有交易')); return; }
      const points = equityPoints(filtered), start = filtered[0].entry_ms, stop = points.at(-1).time;
      const lo = Math.min(1000, ...points.map(p => p.value)), hi = Math.max(1000, ...points.map(p => p.value));
      const pad = Math.max(10, (hi - lo) * .12), min = lo - pad, max = hi + pad;
      const x = time => 80 + (time - start) / Math.max(1, stop - start) * 890;
      const y = value => 240 - (value - min) / (max - min) * 210;
      for (let i = 0; i < 5; i++) {
        const val = min + (max - min) * i / 4, yy = y(val);
        svg.append(svgEl('line', { x1: 80, x2: 970, y1: yy, y2: yy, stroke: '#202739', 'stroke-width': 1 }));
        svg.append(svgEl('text', { x: 68, y: yy + 4, fill: '#8993a8', 'text-anchor': 'end', 'font-size': 12 }, val.toFixed(0)));
      }
      svg.append(svgEl('line', { x1: 80, x2: 970, y1: y(1000), y2: y(1000), stroke: '#8993a8', 'stroke-dasharray': '5 5', opacity: .6 }));
      const path = [`M80 ${y(1000)}`, ...points.map(p => `H${x(p.time).toFixed(2)} V${y(p.value).toFixed(2)}`)].join(' ');
      svg.append(svgEl('path', { d: path, fill: 'none', stroke: '#57d8ff', 'stroke-width': 2, 'stroke-linejoin': 'round' }));
      svg.append(svgEl('text', { x: 80, y: 271, fill: '#8993a8', 'font-size': 12 }, dateText(start, timezone()).slice(0, 10)));
      svg.append(svgEl('text', { x: 970, y: 271, fill: '#8993a8', 'text-anchor': 'end', 'font-size': 12 }, `${dateText(stop, timezone()).slice(0, 10)} · ${timezone() === 'UTC' ? 'UTC' : 'UTC+8'}`));
      const cursor = svgEl('circle', { r: 5, fill: '#c6ff67', stroke: '#0a0d14', 'stroke-width': 2 }); svg.append(cursor);
      chart = { points, x, y, cursor };
      updateCursor();
    }
    function updateCursor() {
      if (!chart || !chart.points[selected]) return;
      chart.cursor.setAttribute('cx', chart.x(chart.points[selected].time)); chart.cursor.setAttribute('cy', chart.y(chart.points[selected].value));
    }
    function detail() {
      const area = $('selectedTrade'); area.replaceChildren();
      if (!filtered.length) { area.append(make('p', '没有符合当前筛选条件的交易。')); return; }
      const t = filtered[selected], grid = make('div', null, 'detail-grid');
      area.append(make('h3', `#${t.id} · ${symbol(t)} ${side(t)} · ${labels[t.outcome]} ${money(t.net_units)} USDT`));
      const zone = timezone() === 'UTC' ? 'UTC' : '北京时间 UTC+8';
      for (const [label, value, note] of [
        ['信号时间', dateText(t.signal_ms, timezone()), zone], ['开仓时间', dateText(t.entry_ms, timezone()), '信号后 1 分钟'],
        ['到期时间', dateText(t.settlement_ms, timezone()), '开仓后 30 分钟'], ['开仓指数 · 原始精度', t.entry_exact, '表格仅显示两位小数'],
        ['到期指数 · 原始精度', t.exit_exact, '按原始精度判断胜负'], ['模拟净收益', `${money(t.net_units)} USDT`, '本金 5 USDT · 赔付 85% · 零手续费']]) {
        const cell = make('div'); cell.append(make('span', label), make('strong', value), make('small', note)); grid.append(cell);
      }
      area.append(grid); updateCursor();
    }
    function ledger() {
      const body = $('researchTradeRows'); body.replaceChildren();
      const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE)); page = Math.min(page, pages - 1);
      if (!filtered.length) { const tr = make('tr'), td = make('td', '当前筛选没有交易'); td.colSpan = 9; tr.append(td); body.append(tr); }
      filtered.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE).forEach((t, offset) => {
        const i = page * PAGE_SIZE + offset, tr = make('tr', null, i === selected ? 'is-selected' : '');
        const first = make('td'), button = make('button', `#${t.id}`); button.type = 'button'; button.setAttribute('aria-label', `查看第 ${t.id} 笔 ${symbol(t)} 交易`);
        button.addEventListener('click', () => { selected = i; detail(); ledger(); }); first.append(button); tr.append(first);
        addCells(tr, [`${symbol(t)} ${side(t)}`, dateText(t.signal_ms, timezone()), dateText(t.entry_ms, timezone()), t.entry.toFixed(2),
          dateText(t.settlement_ms, timezone()), t.exit.toFixed(2), labels[t.outcome], money(t.net_units)]);
        tr.lastElementChild.className = t.net_units > 0 ? 'positive' : t.net_units < 0 ? 'negative' : ''; body.append(tr);
      });
      $('ledgerTimezone').textContent = timezone() === 'UTC' ? '全部时间：UTC' : '全部时间：北京时间 UTC+8';
      $('ledgerPage').textContent = `${page + 1} / ${pages} 页 · ${filtered.length} 笔`;
      $('ledgerPrevious').disabled = page === 0; $('ledgerNext').disabled = page + 1 === pages;
    }
    function refresh() {
      if (!data) return;
      filtered = filterTrades(data, { symbol: $('researchSymbol').value, period: $('researchPeriod').value, direction: $('researchDirection').value, outcome: $('researchOutcome').value });
      selected = Math.max(0, filtered.length - 1); page = 0;
      const s = stats(filtered);
      $('filterSummary').textContent = `${s.signals} 笔 · 胜率 ${percent(s.win_rate)} · ${s.wins} 盈利 / ${s.losses} 亏损 / ${s.refunds} 退款 · 净收益 ${money(s.net_units)} USDT · 子集最大回撤 ${s.max_drawdown_units.toFixed(2)} USDT`;
      renderChart(); detail(); ledger();
    }
    async function load() {
      $('researchContent').hidden = true; $('researchError').hidden = true; $('researchRetry').hidden = true;
      $('researchState').textContent = '正在载入并校验逐笔研究数据…';
      const controller = new AbortController(), timer = setTimeout(() => controller.abort(), 15000);
      try {
        const response = await fetch(view.dataset.researchUrl, { cache: 'no-cache', signal: controller.signal });
        assert(response.ok, `HTTP ${response.status}`); data = validate(await response.json());
        staticContent(); refresh(); $('researchContent').hidden = false;
        $('researchState').textContent = '已校验 1,262 笔 · 719 盈利 / 543 亏损 · 研究未通过稳定 58% 门槛';
      } catch (err) {
        data = null; $('researchState').textContent = '研究数据暂不可用';
        $('researchError').textContent = `加载或核对失败：${err.name === 'AbortError' ? '请求超时' : err.message}。请重试，或下载 CSV 核对。`;
        $('researchError').hidden = false; $('researchRetry').hidden = false;
      } finally { clearTimeout(timer); }
    }
    ['researchSymbol', 'researchPeriod', 'researchDirection', 'researchOutcome'].forEach(id => $(id).addEventListener('change', refresh));
    $('researchTimezone').addEventListener('change', () => { if (data) { renderChart(); detail(); ledger(); } });
    $('ledgerPrevious').addEventListener('click', () => { page--; ledger(); });
    $('ledgerNext').addEventListener('click', () => { page++; ledger(); });
    $('researchRetry').addEventListener('click', load);
    $('researchEquity').addEventListener('pointermove', event => {
      if (!chart) return;
      const point = event.currentTarget.createSVGPoint(); point.x = event.clientX; point.y = event.clientY;
      const transform = event.currentTarget.getScreenCTM(); if (!transform) return;
      const px = point.matrixTransform(transform.inverse()).x;
      let nearest = 0, distance = Infinity;
      chart.points.forEach((p, i) => { const d = Math.abs(chart.x(p.time) - px); if (d < distance) { nearest = i; distance = d; } });
      if (nearest !== selected) { selected = nearest; detail(); ledger(); }
    });
    $('researchEquity').addEventListener('keydown', event => {
      if (!filtered.length || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
      event.preventDefault(); selected = event.key === 'Home' ? 0 : event.key === 'End' ? filtered.length - 1 : Math.max(0, Math.min(filtered.length - 1, selected + (event.key === 'ArrowLeft' ? -1 : 1)));
      page = Math.floor(selected / PAGE_SIZE); detail(); ledger();
    });
    load();
  }
  return { stats, filterTrades, dateText, equityPoints, comparePrices, validate, mount };
});
