"use strict";

const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");

const actual = JSON.parse(fs.readFileSync("research/workbuddy-results.json", "utf8"));
const clone = value => JSON.parse(JSON.stringify(value));

function browser() {
  const elements = new Map();
  const makeElement = () => ({
    textContent: "", className: "", hidden: false, disabled: false, children: [],
    classList: { add() {}, remove() {} }, addEventListener() {},
    replaceChildren() { this.children = []; },
    appendChild(child) { this.children.push(child); },
    getAttribute(name) { return name === "data-wb-report" ? "research/workbuddy-results.json" : null; },
  });
  for (const id of ["workbuddyResearch", "wbStatus", "wbOutcome", "wbContent", "wbLoading", "wbError", "wbFriction", "wbStressRows"]) elements.set(id, makeElement());
  const context = {
    document: { getElementById: id => elements.get(id) || null, createElement: makeElement },
    window: { addEventListener() {}, devicePixelRatio: 1 },
    location: { href: "https://1124w.github.io/CodeXQuant/index.html" },
    // Leave the automatic load unresolved: validation tests use their own input,
    // make no network requests, and cannot race a fetch against file replacement.
    fetch: () => new Promise(() => {}), AbortSignal, URL, Blob, Date,
    setTimeout, clearTimeout, console,
  };
  vm.runInNewContext(fs.readFileSync("workbuddy.js", "utf8"), context);
  return { api: context.window.WorkBuddyReport, elements };
}

function updateSummary(stats, ledger) {
  const wins = ledger.filter(t => t.outcome === "win").length;
  const losses = ledger.filter(t => t.outcome === "loss").length;
  Object.assign(stats, {
    signals: ledger.length, wins, losses, refunds: ledger.length - wins - losses,
    decisive_trades: wins + losses,
    win_rate: wins + losses ? wins / (wins + losses) : null,
    net_units: Number(ledger.reduce((sum, t) => sum + t.net_units, 0).toFixed(8)),
  });
}

function frictionScenario(fee) {
  const report = clone(actual);
  report.config.fee = fee;
  for (const trade of report.trade_ledger) trade.net_units = Number((report.config.stake *
    ((trade.outcome === "win" ? report.config.payout : trade.outcome === "loss" ? -1 : 0) - fee)).toFixed(8));
  updateSummary(report.test, report.trade_ledger);
  const recent = report.latest_three_months.stats || report.latest_three_months;
  const recentDate = report.latest_three_months_start_utc || report.latest_three_months.start_utc;
  updateSummary(recent, report.trade_ledger.filter(t => t.signal_ms >= Date.parse(recentDate)));
  for (const symbol of report.symbols) updateSummary(report.by_symbol[symbol], report.trade_ledger.filter(t => t.symbol === symbol));
  return report;
}

test("actual three-year report reconciles with every financial execution", () => {
  const { api } = browser();
  assert.ok(actual.trade_ledger.length > 0, "The integration fixture must contain actual executions");
  assert.equal(api.validate(actual), actual);
});

test("zero additional fee baseline and one-percent friction stress remain distinct", () => {
  const { api, elements } = browser();
  const baseline = frictionScenario(0);
  const stress = frictionScenario(.01);
  api.validate(baseline);
  api.validate(stress);
  assert.ok(Math.abs(baseline.test.net_units - stress.test.net_units -
    baseline.trade_ledger.length * baseline.config.stake * .01) < 1e-6);
  api.render(baseline);
  assert.match(elements.get("wbFriction").textContent, /无额外手续费/);
  const rows = elements.get("wbStressRows").children;
  assert.ok(rows.some(row => row.children[0].textContent === "额外摩擦 1% 压力"));
});

test("forged summary counts or win rate cannot qualify a report", () => {
  const { api } = browser();
  for (const field of ["wins", "losses", "refunds", "signals", "decisive_trades"]) {
    const bad = clone(actual);
    bad.test[field] += 1;
    assert.throws(() => api.validate(bad), /汇总与逐笔流水不一致/);
  }
  const bad = clone(actual);
  bad.test.win_rate = .99;
  assert.throws(() => api.validate(bad), /汇总与逐笔流水不一致/);
});

test("a forged ledger profit fails even when the headline total is changed with it", () => {
  const { api } = browser();
  const bad = clone(actual);
  bad.trade_ledger[0].net_units += 100;
  bad.test.net_units += 100;
  assert.throws(() => api.validate(bad), /流水赔付或成本不一致/);
});

test("ten-minute events, wrong payouts, and assets outside BTC ETH are rejected", () => {
  const { api } = browser();
  for (const mutate of [r => r.config.horizon_minutes = 10, r => r.config.payout = .8,
                        r => r.symbols.push("SOLUSDT"), r => r.symbols = ["BTCUSDT", "BTCUSDT"]]) {
    const bad = clone(actual);
    mutate(bad);
    assert.throws(() => api.validate(bad), /仅包含 BTC\/ETH/);
  }
  const bad = clone(actual);
  bad.trade_ledger[0].horizon_minutes = 10;
  assert.throws(() => api.validate(bad), /非 30 分钟事件/);
});

test("settlement direction, exact thirty-minute expiry, and time order are validated", () => {
  const { api } = browser();
  for (const mutate of [
    t => { t.exit = t.entry; },
    t => { t.settlement_ms = t.entry_ms + 10 * 60000; },
    t => { t.direction = 0; },
    t => { t.entry = NaN; },
  ]) {
    const bad = clone(actual);
    mutate(bad.trade_ledger[0]);
    assert.throws(() => api.validate(bad), /流水/);
  }
  const bad = clone(actual);
  const last = bad.trade_ledger.pop();
  bad.trade_ledger.unshift(last);
  assert.throws(() => api.validate(bad), /流水时间/);
});

test("training labels settled at or after the prediction boundary are rejected", () => {
  const { api } = browser();
  assert.ok(actual.folds.length > 0);
  for (const name of ["train_latest_settlement_ms", "latest_training_settlement_ms"]) {
    const bad = clone(actual);
    delete bad.folds[0].train_latest_settlement_ms;
    delete bad.folds[0].latest_training_settlement_ms;
    bad.folds[0][name] = Date.parse(bad.folds[0].test_start_utc);
    assert.throws(() => api.validate(bad), /训练标签越过预测边界/);
  }
});

test("recent-quarter and symbol aggregates must reconcile independently", () => {
  const { api } = browser();
  const quarter = clone(actual);
  (quarter.latest_three_months.stats || quarter.latest_three_months).net_units += 1;
  assert.throws(() => api.validate(quarter), /最近三个月/);
  const symbol = clone(actual);
  symbol.by_symbol.BTCUSDT.wins += 1;
  assert.throws(() => api.validate(symbol), /BTCUSDT/);
});

test("forged pass flags cannot turn an unmet fifty-eight-percent target into success", () => {
  const { api, elements } = browser();
  const forged = clone(actual);
  assert.ok(forged.test.win_rate < .58, "This actual research run does not meet the target");
  forged.demo_pass = true;
  forged.research_qualified = true;
  forged.validation_checks = {target_win_rate: true, positive_net: true};
  api.render(forged);
  assert.doesNotMatch(elements.get("wbStatus").textContent, /达到数值目标/);
  assert.match(elements.get("wbOutcome").textContent, /58%/);
});
