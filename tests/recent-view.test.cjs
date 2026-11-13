const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const view = require('../recent-view.js');
const frozen = JSON.parse(fs.readFileSync('research/recent-public.json', 'utf8'));
const copy = () => structuredClone(frozen);
const MINUTE = 60000;
const START = Date.parse('2026-06-01T00:00:00Z');

function opportunity(id, minute, outcome = 'win', symbol = 'BTCUSDT') {
  const entry = '100.00000000000000001';
  const exit = outcome === 'win' ? '101' : outcome === 'loss' ? '99' : entry;
  return { id, symbol, direction: 1, signal_ms: START + minute * MINUTE,
    entry_ms: START + (minute + 1) * MINUTE, settlement_ms: START + (minute + 31) * MINUTE,
    entry: Number(entry), exit: Number(exit), entry_exact: entry, exit_exact: exit, outcome };
}
function scenario(opportunities, initial = 100, multiple = 5) {
  return { assumptions: { ...frozen.assumptions, initial_capital: initial, daily_risk_multiplier: multiple },
    periods: [{ id: 'test', start_ms: START, end_exclusive_ms: START + 10 * 86400000 }], opportunities };
}

test('default 30U replay validates every published stat and admitted financial row in both periods', () => {
  const data = view.validate(copy());
  const half = view.simulate(data, 'recent_six_months', 30);
  const year = view.simulate(data, 'year_to_date', 30);
  assert.deepEqual([half.stats.signals, half.stats.wins, half.stats.losses, half.stats.net_usdt], [338, 203, 135, 1126.5]);
  assert.deepEqual([year.stats.signals, year.stats.wins, year.stats.losses, year.stats.net_usdt], [443, 266, 177, 1473]);
  assert.equal(half.stats.max_drawdown_usdt, 327);
  assert.equal(half.stats.ending_balance, 11126.5);
  assert.equal(half.trades[0].stake, 30);
  assert.equal(half.trades[0].daily_budget_usdt, 150);
});

test('stake parser accepts both endpoints and cents, rejects malformed or out-of-range inputs', () => {
  for (const [input, result] of [[5, 500], ['5.01', 501], ['30.00', 3000], [100.23, 10023], [250, 25000], [' 30 ', 3000]]) {
    assert.equal(view.parseStake(input), result);
  }
  for (const input of ['', ' ', 0, 4.99, 250.01, -5, NaN, Infinity, '30.001', '3e1', '30u', null, {}, '.5', '5.']) {
    assert.throws(() => view.parseStake(input), undefined, String(input));
  }
});

test('larger stake replays bankroll admission instead of multiplying a fixed old ledger', () => {
  const data = scenario([opportunity(1, 0, 'loss'), opportunity(2, 32, 'win')], 30);
  const small = view.simulate(data, 'test', 5), large = view.simulate(data, 'test', 30);
  assert.equal(small.stats.signals, 2);
  assert.equal(small.stats.net_usdt, -.75);
  assert.equal(large.stats.signals, 1);
  assert.equal(large.stats.skipped_insufficient, 1);
  assert.equal(large.stats.net_usdt, -30);
  assert.notEqual(large.stats.net_usdt, small.stats.net_usdt * 6);
});

test('a settlement at the signal timestamp stays pending until the next signal minute', () => {
  const data = scenario([opportunity(1, 0), opportunity(2, 31), opportunity(3, 32)], 5);
  const result = view.simulate(data, 'test', 5);
  assert.deepEqual(result.trades.map(t => t.opportunity_id), [1, 3]);
  assert.equal(result.stats.skipped_insufficient, 1);
  assert.equal(result.trades[1].available_at_signal, 9.25);
});

test('shared position uses BTC tie priority and allows equal entry/expiry only with reserved funds', () => {
  const tied = scenario([opportunity(1, 0, 'win', 'ETHUSDT'), opportunity(2, 0)]);
  const result = view.simulate(tied, 'test', 5);
  assert.equal(result.trades[0].symbol, 'BTCUSDT');
  assert.equal(result.stats.skipped_overlap, 1);
  const boundary = view.simulate(scenario([opportunity(1, 0), opportunity(2, 30)]), 'test', 5);
  assert.equal(boundary.trades.length, 2);
  assert.equal(boundary.trades[1].equity_at_signal, 100);
  assert.equal(boundary.trades[1].available_at_signal, 95);
});

test('daily risk reserves pending principal and assigns losses to their UTC settlement day', () => {
  const reserve = view.simulate(scenario([opportunity(1, 0), opportunity(2, 30)], 100, 1), 'test', 5);
  assert.equal(reserve.stats.skipped_daily_limit, 1);
  const losses = view.simulate(scenario([opportunity(1, 0, 'loss'), opportunity(2, 32, 'loss'), opportunity(3, 64), opportunity(4, 1440)], 100, 2), 'test', 5);
  assert.deepEqual(losses.trades.map(t => t.opportunity_id), [1, 2, 4]);
  assert.equal(losses.stats.skipped_daily_limit, 1);
  const crossing = view.simulate(scenario([opportunity(1, 1430, 'loss'), opportunity(2, 1462), opportunity(3, 2880)], 100, 1), 'test', 5);
  assert.deepEqual(crossing.trades.map(t => t.opportunity_id), [1, 3]);
});

test('half-cent wins round half-up and simulation never mutates source opportunities', () => {
  const data = scenario([opportunity(1, 0), opportunity(2, 32, 'refund')]);
  const before = JSON.stringify(data), result = view.simulate(data, 'test', 5.10);
  assert.equal(result.trades[0].net_units, 4.34);
  assert.equal(result.trades[1].net_units, 0);
  assert.equal(result.stats.refunds, 1);
  assert.equal(result.stats.decisive_trades, 1);
  assert.equal(JSON.stringify(data), before);
  assert.equal(view.tradeSummary([]).win_rate, null);
});

test('tampered opportunity identity, clocks, exact prices, outcomes, totals and monthly sums reject', () => {
  const edits = [
    d => { d.opportunities[0].settlement_ms -= MINUTE; },
    d => { d.opportunities[0].id = 9; },
    d => { d.opportunities[0].entry_exact = '100'; },
    d => { d.opportunities[0].outcome = 'refund'; },
    d => { d.opportunities[1] = { ...d.opportunities[0], id: 2 }; },
    d => { d.published.recent_six_months.stats.net_usdt++; },
    d => { d.published.year_to_date.trades[0].available_at_signal--; },
    d => { d.published.recent_six_months.monthly[0].signals++; },
    d => { d.published.recent_six_months.stress = []; },
    d => { d.strategy.is_live_strategy = true; },
    d => { d.periods[0].end_exclusive_ms++; },
  ];
  for (const edit of edits) { const data = copy(); edit(data); assert.throws(() => view.validate(data)); }
});

test('exact price comparison, UTC display and CSV preserve audit details', () => {
  assert.equal(view.comparePrices('100.00000000000000001', '100.00000000000000000'), 1);
  assert.equal(view.comparePrices('123.4500', '123.45'), 0);
  const time = Date.parse('2026-11-11T22:31:00Z');
  assert.equal(view.dateText(time), '2026-11-11 22:31:00');
  assert.equal(view.dateText(time, 'Asia/Shanghai'), '2026-11-12 06:31:00');
  const result = view.simulate(scenario([opportunity(1, 0)]), 'test', 5.10);
  const csv = view.csvForTrades(result.trades, 'test', 5.10);
  assert.ok(csv.startsWith('\uFEFF'));
  assert.ok(csv.includes('"100.00000000000000001"'));
  assert.ok(csv.includes('"2026-06-01 00:31:00"'));
  assert.ok(csv.includes('"4.34"'));
  assert.ok(csv.endsWith('"test","5.1"\r\n'));
});

test('browser storage returns default when missing, inaccessible or invalid', () => {
  assert.equal(view.savedStake(null), 30);
  assert.equal(view.savedStake({ getItem: () => null }), 30);
  assert.equal(view.savedStake({ getItem: () => '100.23' }), 100.23);
  assert.equal(view.savedStake({ getItem: () => '251' }), 30);
  assert.equal(view.savedStake({ getItem: () => { throw new Error('blocked'); } }), 30);
});

function browser(fetcher, saved = null) {
  const html = fs.readFileSync('demo/recent.html', 'utf8');
  const ids = new Set([...html.matchAll(/id="([^"]+)"/g)].map(m => m[1]));
  const elements = new Map(), storage = new Map(), downloads = [];
  if (saved !== null) storage.set('codexquant.recent.stake', saved);
  const make = tagName => ({ tagName, textContent: '', className: '', hidden: false, children: [], listeners: {}, attributes: {},
    classList: { toggle() {} }, append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    addEventListener(name, fn) { this.listeners[name] = fn; },
    setAttribute(k, v) { this.attributes[k] = String(v); }, getAttribute(k) { return this.attributes[k]; },
    click() { if (tagName === 'a') downloads.push(this); }, remove() {},
    getBoundingClientRect() { return { left: 0, width: 1000 }; },
  });
  const get = id => {
    assert.ok(ids.has(id), `Missing DOM element #${id}`);
    if (!elements.has(id)) elements.set(id, make('div'));
    return elements.get(id);
  };
  get('recentView').setAttribute('data-research-url', '../research/recent-public.json');
  get('recentSymbol').value = 'ALL'; get('recentTimezone').value = 'UTC'; get('recentPeriod').value = 'recent_six_months';
  const requests = [], blobs = [];
  const doc = { getElementById: get, createElement: make, createElementNS: (_, tag) => make(tag), body: make('body'),
    defaultView: {
      fetch: async (...args) => { requests.push(args); return fetcher(...args); },
      localStorage: { getItem: key => storage.get(key) ?? null, setItem: (key, value) => storage.set(key, value) },
      URL: { createObjectURL: blob => { blobs.push(blob); return 'blob:download'; }, revokeObjectURL() {} },
      Blob, setTimeout: fn => fn(),
    } };
  view.mount(doc);
  return { get, storage, requests, downloads, blobs };
}
const tick = () => new Promise(resolve => setImmediate(resolve));
const successful = async () => ({ ok: true, json: async () => copy() });

test('page loads verified data, paginates, filters, switches periods and displays Shanghai time', async () => {
  const page = browser(successful); await tick();
  assert.equal(page.get('recentContent').hidden, false);
  assert.match(page.get('recentState').textContent, /2,221/);
  assert.equal(page.get('recentTradeRows').children.length, 40);
  assert.match(page.get('recentPage').textContent, /1 \/ 9 页 · 338 笔/);
  page.get('recentNext').listeners.click(); assert.match(page.get('recentPage').textContent, /2 \/ 9/);
  page.get('recentPrevious').listeners.click(); assert.match(page.get('recentPage').textContent, /1 \/ 9/);
  page.get('recentPeriod').value = 'year_to_date'; page.get('recentPeriod').listeners.change();
  assert.match(page.get('recentPage').textContent, /443 笔/);
  page.get('recentSymbol').value = 'BTCUSDT'; page.get('recentSymbol').listeners.change();
  assert.ok(page.get('recentTradeRows').children.every(row => row.children[1].textContent.startsWith('BTC')));
  page.get('recentTimezone').value = 'Asia/Shanghai'; page.get('recentTimezone').listeners.change();
  assert.match(page.get('ledgerScope').textContent, /北京时间 UTC\+8/);
  assert.equal(page.requests.length, 1);
  assert.equal(page.requests[0][1].credentials, 'omit');
});

test('page amount update replays locally, preserves fixed 30U stress and rejects invalid input', async () => {
  const page = browser(successful); await tick();
  const stress = page.get('stressRows').children.map(row => row.children.map(cell => cell.textContent));
  page.get('recentStake').value = '5.10'; page.get('stakeForm').listeners.submit({ preventDefault() {} });
  assert.match(page.get('scenarioText').textContent, /5.10 USDT/);
  assert.match(page.get('scenarioText').textContent, /盈利 \+4.34 USDT/);
  assert.equal(page.storage.get('codexquant.recent.stake'), '5.10');
  assert.deepEqual(page.get('stressRows').children.map(row => row.children.map(cell => cell.textContent)), stress);
  const previous = page.get('scenarioText').textContent;
  page.get('recentStake').value = '250.01'; page.get('stakeForm').listeners.submit({ preventDefault() {} });
  assert.equal(page.get('stakeError').hidden, false);
  assert.equal(page.get('recentStake').getAttribute('aria-invalid'), 'true');
  assert.equal(page.get('scenarioText').textContent, previous);
  page.get('resetStake').listeners.click();
  assert.equal(page.get('recentStake').value, '30.00');
  assert.equal(page.get('stakeError').hidden, true);
  assert.equal(page.requests.length, 1, 'amount changes must never mutate server or bot settings');
});

test('page restores valid saved amount, exports current ledger and selects detailed prices', async () => {
  const page = browser(successful, '100.23'); await tick();
  assert.equal(page.get('recentStake').value, '100.23');
  page.get('recentEquity').listeners.keydown({ key: 'Home', preventDefault() {} });
  assert.match(page.get('recentSelected').children[0].textContent, /#1 ·/);
  page.get('recentTradeRows').children[1].children[0].children[0].listeners.click();
  assert.match(page.get('recentSelected').children[0].textContent, /#2 ·/);
  page.get('downloadRecent').listeners.click();
  assert.match(page.downloads[0].download, /recent_six_months-100.23U-ALL\.csv$/);
  assert.match(await page.blobs[0].text(), /"recent_six_months","100.23"/);
});

test('offline or corrupt data stays hidden and retry can recover', async () => {
  let mode = 'offline';
  const page = browser(async () => {
    if (mode === 'offline') throw new Error('offline');
    const data = copy(); if (mode === 'corrupt') data.published.recent_six_months.stats.wins++;
    return { ok: true, json: async () => data };
  });
  await tick();
  assert.equal(page.get('recentContent').hidden, true);
  assert.match(page.get('recentError').textContent, /offline/);
  mode = 'corrupt'; await page.get('recentRetry').listeners.click();
  assert.equal(page.get('recentContent').hidden, true);
  assert.match(page.get('recentError').textContent, /校验失败/);
  mode = 'ok'; await page.get('recentRetry').listeners.click();
  assert.equal(page.get('recentContent').hidden, false);
  assert.equal(page.get('recentError').hidden, true);
});
