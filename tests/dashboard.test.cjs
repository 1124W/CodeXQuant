const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const tick = () => new Promise(resolve => setImmediate(resolve));
const historical = JSON.parse(fs.readFileSync("research/results.json", "utf8"));

function browser(script, fetcher) {
  let now = Date.parse("2026-11-08T00:00:00Z");
  const elements = new Map();
  const canvasCalls = [];
  const context2d = new Proxy({}, { get: (_, key) => key === "createLinearGradient" ? () => ({ addColorStop() {} }) : (...args) => { canvasCalls.push({ method: key, args }); } });
  const html = fs.readFileSync("index.html", "utf8");
  const ids = new Set([...html.matchAll(/id="([^"]+)"/g)].map(match => match[1]));
  const makeElement = tagName => ({
    tagName, textContent: "", style: {}, className: "", disabled: false, hidden: false, children: [],
    addEventListener() {}, setAttribute() {}, getBoundingClientRect: () => ({ width: 700, height: 340 }),
    getContext: () => context2d, appendChild(child) { this.children.push(child); return child; },
    replaceChildren(...children) { this.children = children; },
  });
  const getElementById = id => {
    assert.ok(ids.has(id), `Missing element #${id}`);
    if (!elements.has(id)) elements.set(id, makeElement("div"));
    return elements.get(id);
  };
  class FakeDate extends Date {
    constructor(...args) { super(...(args.length ? args : [now])); }
    static now() { return now; }
  }
  class FakeSocket {
    static CONNECTING = 0;
    static OPEN = 1;
    static CLOSING = 2;
    static CLOSED = 3;
    readyState = 1;
    constructor(url) { this.url = url; }
    close() { this.readyState = 3; this.onclose?.(); }
  }
  const timers = [], intervals = [];
  const sandbox = vm.createContext({
    Date: FakeDate, Intl, console: { warn() {} }, AbortController, AbortSignal,
    devicePixelRatio: 1, WebSocket: FakeSocket,
    document: { getElementById, createElement: makeElement, querySelectorAll: () => [], addEventListener() {} },
    window: { addEventListener() {} },
    setInterval(fn, delay) { intervals.push({ fn, delay }); return intervals.length; }, setTimeout(fn, delay) { timers.push({ fn, delay }); return timers.length; }, clearTimeout() {},
    fetch: (...args) => fetcher(...args),
  });
  vm.runInContext(fs.readFileSync(script, "utf8"), sandbox);
  return {
    sandbox, elements, timers, intervals, canvasCalls, get: getElementById,
    advance: ms => { now += ms; },
    run: code => vm.runInContext(code, sandbox),
  };
}

function rows() {
  const end = Date.parse("2026-11-08T00:00:00Z");
  return Array.from({ length: 240 }, (_, i) => [end - (239 - i) * 1800000, "100", "102", "99", "101"]);
}

test("market failure uses demo; recovery loads real data; failure then preserves last snapshot", async () => {
  let fail = true;
  const page = browser("app.js", async () => {
    if (fail) throw new Error("offline");
    return { ok: true, json: async () => rows() };
  });
  await tick();
  assert.equal(page.get("dataMode").textContent, "模拟数据");
  assert.match(page.get("chartFreshness").textContent, /不代表真实市场/);
  fail = false;
  await page.run("refreshCandles()");
  assert.equal(page.get("dataMode").textContent, "行情快照");
  const lastReal = page.run("JSON.stringify(state.candles.BTCUSDT)");
  fail = true;
  await page.run("loadCandles('BTCUSDT')");
  assert.equal(page.get("dataMode").textContent, "数据已过期");
  assert.equal(page.run("JSON.stringify(state.candles.BTCUSDT)"), lastReal);
});

test("invalid OHLC, gaps and outdated snapshots cannot appear as real recent data", async () => {
  for (const mutate of [data => { data[10][2] = "1"; }, data => { data[10][0] += 1; },
                        data => data.forEach(row => { row[0] -= 7200000; })]) {
    const data = rows(); mutate(data);
    const page = browser("app.js", async () => ({ ok: true, json: async () => data }));
    await tick();
    assert.equal(page.get("dataMode").textContent, "模拟数据");
  }
});

test("snapshot age and quote silence revoke freshness; malformed stream frames are ignored", async () => {
  const page = browser("app.js", async () => ({ ok: true, json: async () => rows() }));
  await tick();
  page.run("state.socket.onmessage({data: 'invalid JSON'})");
  page.run("state.socket.onmessage({data: JSON.stringify({data:{e:'markPriceUpdate',s:'UNKNOWN',i:'100',E:Date.now()}})})");
  page.run("for (const s of SYMBOLS) state.socket.onmessage({data: JSON.stringify({data:{e:'markPriceUpdate',s,i:'100',p:'999',E:Date.now()}})})");
  assert.equal(page.get("connectionText").textContent, "指数价格实时更新");
  const socket = page.run("state.socket");
  page.advance(16000);
  page.run("checkMarketHealth()");
  assert.equal(page.get("BTCUSDT-freshness").textContent, "指数价格已过期");
  assert.notEqual(page.get("connectionText").textContent, "指数价格实时更新");
  page.run("updateQuote('BTCUSDT', 200, Date.now()-20000, 'REST')");
  assert.equal(page.get("BTCUSDT-price").textContent, "100.00");
  page.advance(80000);
  page.run("render('BTCUSDT'); checkMarketHealth()");
  assert.equal(page.get("dataMode").textContent, "数据已过期");
  assert.equal(socket.readyState, 3);
});

test("official index endpoints and i field are used instead of spot or mark price", async () => {
  const requests = [];
  const page = browser("app.js", async url => {
    requests.push(url);
    return { ok: true, json: async () => url.includes("premiumIndex") ?
      { symbol: new URL(url).searchParams.get("symbol"), indexPrice: "101.12345678", markPrice: "999.9999", time: Date.parse("2026-11-08T00:00:00Z") } : rows() };
  });
  await tick();
  assert.equal(page.get("BTCUSDT-price").textContent, "101.12");
  assert.match(page.get("BTCUSDT-source").textContent, /指数快照/);
  assert.match(page.get("BTCUSDT-time").textContent, /08:00:00.000/);
  assert.ok(requests.some(url => url.includes("/indexPriceKlines?pair=BTCUSDT")));
  assert.ok(requests.some(url => url.includes("/indexPriceKlines?pair=ETHUSDT")));
  assert.ok(requests.every(url => url.startsWith("https://fapi.binance.com/fapi/v1/")));
  assert.match(page.run("state.socket.url"), /fstream\.binance\.com\/market\/stream\?streams=btcusdt@markPrice@1s\/ethusdt@markPrice@1s/);
  page.run("state.socket.onmessage({data: JSON.stringify({data:{e:'markPriceUpdate',s:'BTCUSDT',i:'123.456789',p:'999',E:Date.now()}})})");
  assert.equal(page.get("BTCUSDT-price").textContent, "101.12");
  page.intervals.find(timer => timer.delay === 2000 && String(timer.fn).includes("flushMarketDisplay")).fn();
  assert.equal(page.get("BTCUSDT-price").textContent, "123.46");
  assert.match(page.get("BTCUSDT-source").textContent, /2s 显示刷新/);
  for (const body of ["{e:'markPriceUpdate',s:'BTCUSDT',p:'200',E:Date.now()+1}",
    "{e:'markPriceUpdate',s:'BTCUSDT',i:'200',st:2,E:Date.now()+1}",
    "{e:'24hrMiniTicker',s:'BTCUSDT',i:'200',E:Date.now()+1}",
    "{e:'markPriceUpdate',s:'BTCUSDT',i:'NaN',E:Date.now()+1}"]) {
    page.run(`state.socket.onmessage({data:JSON.stringify({data:${body}})})`);
    assert.equal(page.get("BTCUSDT-price").textContent, "123.46");
  }
});

test("out-of-order, repeated and future timestamps cannot renew or overwrite an index quote", async () => {
  const page = browser("app.js", async () => ({ ok: true, json: async () => rows() }));
  await tick();
  page.run("updateQuote('BTCUSDT', 100, Date.now(), 'WS')");
  const original = page.run("state.quotes.BTCUSDT.receivedAt");
  page.advance(1000);
  page.run("updateQuote('BTCUSDT', 999, Date.now()-1000, 'REST')");
  page.run("updateQuote('BTCUSDT', 999, Date.now()-2000, 'WS')");
  page.run("updateQuote('BTCUSDT', 999, Date.now()+6000, 'WS')");
  assert.equal(page.get("BTCUSDT-price").textContent, "100.00");
  assert.equal(page.run("state.quotes.BTCUSDT.receivedAt"), original);
  page.advance(15000);
  page.run("updateHealth()");
  assert.equal(page.get("BTCUSDT-freshness").textContent, "指数价格已过期");
});

test("fallback polls only symbols without a healthy stream and restores realtime when it resumes", async () => {
  const requests = [];
  const page = browser("app.js", async url => {
    requests.push(url);
    return { ok: true, json: async () => url.includes("premiumIndex") ?
      { symbol: new URL(url).searchParams.get("symbol"), indexPrice: "101", time: Date.parse("2026-11-08T00:00:00Z") } : rows() };
  });
  await tick();
  page.run("updateQuote('BTCUSDT', 102, Date.now(), 'WS')");
  requests.length = 0;
  await page.run("refreshIndexQuotes()");
  assert.equal(requests.length, 1);
  assert.match(requests[0], /symbol=ETHUSDT/);
  assert.match(page.get("connectionText").textContent, /指数快照/);
  page.run("updateQuote('ETHUSDT', 102, Date.now(), 'WS')");
  assert.equal(page.get("connectionText").textContent, "指数价格实时更新");
});

test("a slow REST snapshot cannot replace a newer streamed price", async () => {
  const pending = [];
  const page = browser("app.js", async url => url.includes("premiumIndex") ?
    new Promise(resolve => pending.push({ symbol: new URL(url).searchParams.get("symbol"), resolve })) :
    { ok: true, json: async () => rows() });
  await tick();
  page.run("updateQuote('BTCUSDT', 105, Date.now(), 'WS')");
  for (const { symbol, resolve } of pending) resolve({ ok: true, json: async () =>
    ({ symbol, indexPrice: "99", time: Date.parse("2026-11-07T23:59:59Z") }) });
  await tick();
  assert.equal(page.get("BTCUSDT-price").textContent, "105.00");
  assert.equal(page.run("state.quotes.BTCUSDT.source"), "WS");
});

test("disconnect keeps the last value, retries once and ignores messages from the old socket", async () => {
  const page = browser("app.js", async () => ({ ok: true, json: async () => rows() }));
  await tick();
  const oldSocket = page.run("state.socket");
  page.run("updateQuote('BTCUSDT', 100, Date.now(), 'WS'); state.socket.close()");
  assert.equal(page.get("BTCUSDT-price").textContent, "100.00");
  assert.equal(page.run("state.quotes.BTCUSDT.source"), "DISCONNECTED");
  const retry = page.timers.find(timer => timer.delay === 1000);
  assert.ok(retry);
  assert.equal(page.run("state.socket"), null);
  retry.fn();
  assert.ok(page.run("state.socket"));
  const active = page.run("state.socket");
  page.run("connectStream()");
  assert.equal(page.run("state.socket"), active);
  oldSocket.onmessage({ data: JSON.stringify({ data: { e: "markPriceUpdate", s: "BTCUSDT", i: "999", E: Date.parse("2026-11-08T00:00:00Z") + 1 } }) });
  oldSocket.onclose();
  assert.equal(page.get("BTCUSDT-price").textContent, "100.00");
  assert.equal(page.run("state.socket"), active);
});

test("index snapshot timeout clears pending work so the next poll can recover", async () => {
  const page = browser("app.js", async (url, options) => url.includes("premiumIndex") ?
    new Promise((resolve, reject) => options.signal.addEventListener("abort", () => reject(new Error("timeout")))) :
    { ok: true, json: async () => rows() });
  await tick();
  assert.equal(page.run("state.quotePending.size"), 2);
  page.timers.filter(timer => timer.delay === 8000).forEach(timer => timer.fn());
  await tick();
  assert.equal(page.run("state.quotePending.size"), 0);
  assert.equal(page.get("BTCUSDT-price").textContent, "");
  assert.equal(page.get("BTCUSDT-freshness").textContent, "等待指数价格");
});

function modernReport() {
  return { schema_version: 2, start_utc: "2026-08-08", cutoff_utc: "2026-10-07", end_exclusive_utc: "2026-11-06",
    payout: .8, fee_per_trade: 0, break_even_win_rate: 1 / 1.8, publishable: false,
    chosen: { train: { signals: 150, win_rate: .6, win_rate_lower_95: .5, net_units: 12, max_drawdown_units: 8 },
      test: { signals: 100, win_rate: .54, win_rate_lower_95: .44, win_rate_upper_95: .64, net_units: -2.8, max_drawdown_units: 13 } } };
}

test("legacy report is clearly marked as needing boundary revalidation", async () => {
  const legacy = JSON.parse(JSON.stringify(historical));
  delete legacy.schema_version;
  const page = browser("research.js", async () => ({ ok: true, json: async () => legacy }));
  await tick();
  assert.equal(page.get("researchBadge").textContent, "旧版报告");
  assert.match(page.get("researchNote").textContent, /重新运行/);
});

test("report handles no selection, failed gates, passed gates and stale historical results", async () => {
  const report = modernReport();
  const page = browser("research.js", async () => ({ ok: true, json: async () => report }));
  await tick();
  assert.match(page.get("researchTitle").textContent, /未通过/);
  page.sandbox.fixture = { ...report, chosen: null };
  page.run("renderResearch(fixture)");
  assert.match(page.get("researchTitle").textContent, /训练样本不足/);
  assert.equal(page.get("testRate").textContent, "—");
  report.chosen.test = { signals: 100, win_rate: .8, win_rate_lower_95: .7, win_rate_upper_95: .87, net_units: 44, max_drawdown_units: 4 };
  report.publishable = true;
  report.validation_checks = Object.fromEntries(["selected_on_training_only", "holdout_min_50_signals", "holdout_lower_95_above_break_even", "holdout_positive_net"].map(key => [key, true]));
  page.sandbox.fixture = report;
  page.run("renderResearch(fixture)");
  assert.match(page.get("researchTitle").textContent, /仍需前向验证/);
  page.advance(10 * 86400000);
  page.run("renderResearch(fixture)");
  assert.equal(page.get("researchBadge").textContent, "历史快照");
});

test("failed fetch or malformed report cannot display validation success", async () => {
  for (const fetcher of [async () => { throw new Error("offline"); },
    async () => ({ ok: true, json: async () => ({ ...modernReport(), payout: -1 }) }),
    async () => ({ ok: true, json: async () => ({ ...modernReport(), chosen: { train: {} } }) })]) {
    const page = browser("research.js", fetcher);
    await tick();
    assert.equal(page.get("researchTitle").textContent, "研究报告暂不可用");
    assert.equal(page.get("researchBadge").textContent, "加载失败");
  }
});

function eventFixture() {
  const sample = (wins, losses, payout = .8, fee = .01) => {
    const count = wins + losses;
    return { signals: count, wins, losses, refunds: 0, decisive_trades: count, win_rate: count ? wins / count : null,
      win_rate_lower_95: count ? .1 : null, win_rate_upper_95: count ? .9 : null,
      block_bootstrap_lower_95: count ? .05 : null, block_bootstrap_upper_95: count ? .95 : null,
      block_bootstrap_net_lower_95: count ? -3 : null, block_bootstrap_net_upper_95: count ? 3 : null,
      effective_break_even_win_rate: count ? (1 + fee) / (1 + payout) : null,
      net_units: wins * payout - losses - count * fee, max_drawdown_units: count ? 1.2 : 0 };
  };
  const start = "2026-01-01", walkStart = "2026-04-01", cutoff = "2026-06-01", end = "2026-07-01";
  const first = Date.parse(`${walkStart}T00:00:00Z`);
  const ledger = Array.from({ length: 4 }, (_, index) => ({ symbol: index < 2 ? "BTCUSDT" : "ETHUSDT", signal_ms: first + index * 86400000,
    entry_ms: first + index * 86400000 + 60000, settlement_ms: first + index * 86400000 + 1860000,
    outcome: index % 2 ? "loss" : "win", net_units: index % 2 ? -1.01 : .79, split: "walk_forward" }));
  let balance = 1000;
  const equity = ledger.map(trade => { balance += trade.net_units; return { timestamp_ms: trade.settlement_ms, balance_units: balance, net_units: balance - 1000, drawdown_units: 0, split: "walk_forward" }; });
  return { schema_version: 3, live_approved: false, publishable: false, start_utc: start, cutoff_utc: cutoff, end_exclusive_utc: end,
    target_win_rate: .58, payout: .8, fee_per_trade: .01, break_even_win_rate: 1.01 / 1.8,
    config: { stake: 1, initial_bankroll: 1000, min_test_trades: 100, holdout_days: 30, latency_minutes: 1, horizon_minutes: 30 },
    walk_forward: { start_utc: walkStart, aggregate: sample(2, 2), active_folds: 1, positive_folds: 0,
      folds: [{ train_start: start, train_end: walkStart, test_start: walkStart, test_end: cutoff,
        chosen: { parameters: { family: "mean_reversion" } }, test: sample(2, 2) }], by_symbol: { BTCUSDT: sample(1, 1), ETHUSDT: sample(1, 1) } },
    final_holdout: { start_utc: cutoff, end_exclusive_utc: end, chosen: null, test: sample(0, 0), by_symbol: { BTCUSDT: sample(0, 0), ETHUSDT: sample(0, 0) } },
    validation_checks: { selected_on_training_only: true, walk_forward_target_met: false, holdout_target_met: false },
    trade_ledger: ledger, equity_curve: equity, validation_context: { pristine_unobserved_holdout: false },
    cost_scenarios: [.8, .75, .7].map(payout => ({ name: "base", payout, fee_per_trade: .01, walk_forward: sample(2, 2, payout), final_holdout: sample(0, 0, payout) })) };
}
const eventHistorical = eventFixture();
const eventBrowser = report => browser("research.js", async url => ({ ok: true, json: async () => url.includes("event-results") ? report : modernReport() }));

test("event research shows real outcomes, empty final holdout, frozen stress scenarios and equity", async () => {
  const page = eventBrowser(eventHistorical);
  await tick();
  assert.equal(page.get("eventWalkRate").textContent, `${(eventHistorical.walk_forward.aggregate.win_rate * 100).toFixed(1)}%`);
  assert.equal(page.get("eventTarget").textContent, "58.0%");
  assert.equal(page.get("eventWalkConfidence").textContent, "5.0% – 95.0%");
  assert.equal(page.get("eventWalkDrawdown").textContent, "1.20 单位");
  assert.equal(page.get("eventHoldoutRate").textContent, "无样本");
  assert.match(page.get("eventTitle").textContent, /空仓/);
  assert.match(page.get("eventHoldoutDetail").textContent, /保持空仓/);
  assert.equal(page.get("eventDownloads").hidden, false);
  assert.equal(page.get("eventFoldRows").children.length, eventHistorical.walk_forward.folds.length);
  assert.equal(page.get("eventScenarioRows").children.length, eventHistorical.cost_scenarios.length);
  assert.equal(page.get("eventSymbolRows").children.length, 4);
  assert.ok(page.canvasCalls.filter(call => call.method === "lineTo").length >= 4);
  assert.ok(page.canvasCalls.filter(call => call.method === "lineTo").every(call => call.args.every(Number.isFinite)));
  assert.match(page.get("eventNote").textContent, /不属于全新的独立证据/);
});

test("forged pass flags cannot turn a failed event report into validation success", async () => {
  const report = structuredClone(eventHistorical);
  report.publishable = true;
  report.validation_checks = Object.fromEntries(Object.keys(report.validation_checks).map(key => [key, true]));
  const page = eventBrowser(report);
  await tick();
  assert.doesNotMatch(page.get("eventTitle").textContent, /历史门槛通过/);
  assert.equal(page.get("eventBadge").textContent, "空仓 / 无样本");
});

test("event counts, ledger, costs and chronology must agree before rendering results", async () => {
  const mutations = [
    report => { report.walk_forward.aggregate.win_rate = .9; },
    report => { report.trade_ledger.pop(); },
    report => { report.break_even_win_rate = .5; },
    report => { report.walk_forward.folds[0].test_start = "2025-01-01"; },
    report => { report.final_holdout.test.win_rate = 0; },
    report => { report.equity_curve[0].balance_units = -100; },
  ];
  for (const mutate of mutations) {
    const report = structuredClone(eventHistorical); mutate(report);
    const page = eventBrowser(report);
    await tick();
    assert.match(page.get("eventTitle").textContent, /暂不可用/);
    assert.equal(page.get("eventBadge").textContent, "未验证");
    assert.equal(page.get("eventDownloads").hidden, true);
  }
});

test("missing event report leaves legacy research usable and shows no profit claim", async () => {
  const page = browser("research.js", async url => url.includes("event-results") ? { ok: false, status: 404 } : { ok: true, json: async () => modernReport() });
  await tick();
  assert.match(page.get("eventTitle").textContent, /暂不可用/);
  assert.equal(page.get("eventDownloads").hidden, true);
  assert.match(page.get("researchTitle").textContent, /未通过/);
});
