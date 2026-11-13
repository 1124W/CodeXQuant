const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const view = require('../research-view.js');
const frozen = JSON.parse(fs.readFileSync('research/high-win-public.json', 'utf8'));
const copy = () => structuredClone(frozen);

test('verified public cohort reconciles to the exact audited totals and UTC periods', () => {
  const d = view.validate(copy());
  assert.equal(view.stats(d.trades).win_rate, 719 / 1262);
  assert.equal(view.stats(d.trades).net_units, 340.75);
  assert.equal(view.filterTrades(d, { period: 'validation' }).length, 935);
  assert.equal(view.filterTrades(d, { period: 'historical_confirmation' }).length, 327);
  assert.equal(view.filterTrades(d, { symbol: 'BTCUSDT' }).length, 459);
  assert.equal(view.filterTrades(d, { symbol: 'ETHUSDT' }).length, 803);
});

test('filtering retains original admission decisions, prices and equity arithmetic', () => {
  const before = JSON.stringify(frozen);
  const btc = view.filterTrades(frozen, { symbol: 'BTCUSDT' });
  assert.ok(btc.every(t => frozen.trades.includes(t)));
  assert.equal(view.equityPoints(btc).at(-1).value, 1054.5);
  assert.equal(view.equityPoints(frozen.trades).at(-1).value, 1340.75);
  assert.equal(JSON.stringify(frozen), before);
  assert.equal(frozen.trades[0].entry_exact, '1827.24186047');
});

test('empty cohort has no invented win rate and time display crosses midnight correctly', () => {
  assert.equal(view.stats(view.filterTrades(frozen, { outcome: 'refund' })).win_rate, null);
  const time = Date.parse('2026-11-11T22:31:00Z');
  assert.equal(view.dateText(time), '2026-11-11 22:31:00');
  assert.equal(view.dateText(time, 'Asia/Shanghai'), '2026-11-12 06:31:00');
});

test('exact decimal outcome comparison does not collapse near-equal prices', () => {
  assert.equal(view.comparePrices('100.00000000000000001', '100.00000000000000000'), 1);
  assert.equal(view.comparePrices('123.4500', '123.45'), 0);
});

test('tampered clocks, payout, prices, duplicate identities and period totals reject', () => {
  const edits = [
    d => { d.trades[0].settlement_ms -= 60000; },
    d => { d.trades[0].net_units = 4.25; },
    d => { d.trades[0].entry_exact = '100'; },
    d => { d.trades[1] = { ...d.trades[0], id: 2 }; },
    d => { d.comparison.periods[0].wins++; },
    d => { d.comparison.periods[0].end_exclusive_ms++; },
  ];
  for (const edit of edits) { const d = copy(); edit(d); assert.throws(() => view.validate(d)); }
});

function browser(fetcher) {
  const html = fs.readFileSync('demo/research.html', 'utf8');
  const ids = new Set([...html.matchAll(/id="([^"]+)"/g)].map(m => m[1]));
  const elements = new Map();
  const make = tagName => ({ tagName, textContent: '', className: '', hidden: false, children: [], listeners: {}, attributes: {},
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    get lastElementChild() { return this.children.at(-1); },
    addEventListener(name, fn) { this.listeners[name] = fn; },
    setAttribute(k, v) { this.attributes[k] = v; },
  });
  const get = id => {
    assert.ok(ids.has(id), `Missing DOM element #${id}`);
    if (!elements.has(id)) elements.set(id, make('div'));
    return elements.get(id);
  };
  get('researchView').dataset = { researchUrl: '../research/high-win-public.json' };
  for (const id of ['researchSymbol', 'researchDirection', 'researchOutcome']) get(id).value = 'ALL';
  get('researchPeriod').value = 'continuous_post_selection'; get('researchTimezone').value = 'UTC';
  const sandbox = vm.createContext({ document: { getElementById: get, createElement: make, createElementNS: (_, tag) => make(tag) },
    fetch: fetcher, AbortController, setTimeout, clearTimeout });
  vm.runInContext(fs.readFileSync('research-view.js', 'utf8'), sandbox);
  return { get };
}
const tick = () => new Promise(resolve => setImmediate(resolve));

test('browser loads, paginates, filters, changes timezone and handles an empty cohort', async () => {
  const page = browser(async () => ({ ok: true, json: async () => copy() }));
  await tick();
  assert.equal(page.get('researchContent').hidden, false);
  assert.equal(page.get('researchTradeRows').children.length, 50);
  assert.match(page.get('ledgerPage').textContent, /1 \/ 26/);
  page.get('ledgerNext').listeners.click(); assert.match(page.get('ledgerPage').textContent, /2 \/ 26/);
  page.get('researchSymbol').value = 'BTCUSDT'; page.get('researchSymbol').listeners.change();
  assert.match(page.get('filterSummary').textContent, /459 笔/);
  page.get('researchTimezone').value = 'Asia/Shanghai'; page.get('researchTimezone').listeners.change();
  assert.match(page.get('ledgerTimezone').textContent, /北京时间/);
  page.get('researchOutcome').value = 'refund'; page.get('researchOutcome').listeners.change();
  assert.match(page.get('filterSummary').textContent, /无有效交易/);
  assert.equal(page.get('researchTradeRows').children.length, 1);
});

test('browser failure is visible and retry recovers without showing false numbers', async () => {
  let fail = true;
  const page = browser(async () => { if (fail) throw new Error('offline'); return { ok: true, json: async () => copy() }; });
  await tick();
  assert.equal(page.get('researchContent').hidden, true);
  assert.equal(page.get('researchError').hidden, false);
  assert.match(page.get('researchError').textContent, /offline/);
  fail = false; await page.get('researchRetry').listeners.click();
  assert.equal(page.get('researchContent').hidden, false);
  assert.equal(page.get('researchError').hidden, true);
});
