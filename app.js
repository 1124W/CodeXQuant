const API = "https://fapi.binance.com/fapi/v1";
const STREAM = "wss://fstream.binance.com/market/stream?streams=btcusdt@markPrice@1s/ethusdt@markPrice@1s";
const state = { symbol: "BTCUSDT", candles: {}, modes: {}, updated: {}, pending: new Set(), quotes: {}, quotePending: new Set(), quoteDirty: new Set(), socket: null, retries: 0, streamSince: 0, streamLast: 0, reconnectTimer: null };
const SYMBOLS = ["BTCUSDT", "ETHUSDT"];
const CANDLE_STALE_MS = 90000;
const QUOTE_STALE_MS = 15000;
const STREAM_STALE_MS = 5000;
const el = id => document.getElementById(id);
const money = value => new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value);
const indexPrice = value => new Intl.NumberFormat("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }).format(value);
const exchangeTime = timestamp => new Intl.DateTimeFormat("zh-CN", { timeZone: "Asia/Shanghai", hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit", fractionalSecondDigits: 3 }).format(new Date(timestamp));

function tickClock() {
  el("clock").textContent = new Intl.DateTimeFormat("zh-CN", { timeZone: "Asia/Shanghai", hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(new Date());
}

function demoCandles(symbol) {
  let price = symbol === "BTCUSDT" ? 68000 : 3500;
  let seed = symbol === "BTCUSDT" ? 1124 : 2048;
  const random = () => ((seed = seed * 1664525 + 1013904223 >>> 0) / 4294967296);
  return Array.from({ length: 240 }, (_, index) => {
    const open = price;
    price *= 1 + (random() - .5) * .009;
    const close = price;
    const spread = price * .002;
    return { time: Date.now() - (239 - index) * 1800000, open, close, high: Math.max(open, close) + spread, low: Math.min(open, close) - spread };
  });
}

async function loadCandles(symbol) {
  if (state.pending.has(symbol)) return;
  state.pending.add(symbol);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 10000);
  try {
    const response = await fetch(`${API}/indexPriceKlines?pair=${symbol}&interval=30m&limit=240`, { signal: controller.signal, cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const rows = await response.json();
    if (!Array.isArray(rows) || rows.length < 21) throw new Error("Insufficient candles");
    const candles = rows.map(row => ({ time: +row[0], open: +row[1], high: +row[2], low: +row[3], close: +row[4] }));
    if (candles.some((c, i) => !Number.isFinite(c.time) || c.time % 1800000 !== 0 ||
        ![c.open, c.high, c.low, c.close].every(v => Number.isFinite(v) && v > 0) ||
        c.low > Math.min(c.open, c.close) || c.high < Math.max(c.open, c.close) ||
        (i > 0 && c.time - candles[i - 1].time !== 1800000))) throw new Error("Invalid candle data");
    if (Date.now() - candles.at(-1).time > 3600000 || candles.at(-1).time > Date.now()) throw new Error("Outdated candle data");
    state.candles[symbol] = candles;
    state.updated[symbol] = Date.now();
    state.modes[symbol] = "RECENT";
  } catch (error) {
    console.warn(`${symbol} candles unavailable`, error);
    if (state.updated[symbol]) state.modes[symbol] = "STALE";
    else {
      state.candles[symbol] = demoCandles(symbol);
      state.modes[symbol] = "DEMO";
    }
  } finally {
    clearTimeout(timeout);
    state.pending.delete(symbol);
  }
  if (symbol === state.symbol) render(symbol);
}

function ema(values, period) {
  const factor = 2 / (period + 1);
  let current = values[0];
  return values.map((value, index) => current = index ? value * factor + current * (1 - factor) : value);
}

function drawChart(candles) {
  const canvas = el("chart"), bounds = canvas.getBoundingClientRect(), scale = devicePixelRatio || 1;
  canvas.width = bounds.width * scale;
  canvas.height = bounds.height * scale;
  const ctx = canvas.getContext("2d");
  ctx.scale(scale, scale);
  const data = candles.slice(-80), closes = candles.map(c => c.close), fast = ema(closes, 9).slice(-80), slow = ema(closes, 21).slice(-80);
  const width = bounds.width, height = bounds.height;
  const min = Math.min(...data.map(c => c.low), ...fast, ...slow), max = Math.max(...data.map(c => c.high), ...fast, ...slow), range = max - min || 1;
  const step = (width - 20) / data.length, x = index => 10 + (index + .5) * step;
  const y = value => 12 + (max - value) / range * (height - 36);
  ctx.strokeStyle = "rgba(255,255,255,.055)";
  for (let i = 0; i < 5; i++) { const level = 12 + i * (height - 36) / 4; ctx.beginPath(); ctx.moveTo(0, level); ctx.lineTo(width, level); ctx.stroke(); }
  const bodyWidth = Math.max(2, Math.min(7, step * .62));
  data.forEach((c, i) => {
    const rising = c.close >= c.open, color = rising ? "#66e39a" : "#ff647c", center = x(i);
    ctx.strokeStyle = color; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(center, y(c.high)); ctx.lineTo(center, y(c.low)); ctx.stroke();
    const top = y(Math.max(c.open, c.close)), bottom = y(Math.min(c.open, c.close));
    ctx.fillStyle = color; ctx.fillRect(center - bodyWidth / 2, top, bodyWidth, Math.max(1, bottom - top));
  });
  for (const [series, color, lineWidth] of [[fast, "#b8f348", 2], [slow, "#57d8ff", 1.4]]) {
    ctx.beginPath(); series.forEach((value, i) => i ? ctx.lineTo(x(i), y(value)) : ctx.moveTo(x(i), y(value)));
    ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.stroke();
  }
  ctx.fillStyle = "#8991a3"; ctx.font = "10px ui-monospace,monospace";
  ctx.fillText(money(min), 8, height - 5); ctx.textAlign = "right"; ctx.fillText(money(max), width - 8, 12);
}

function applyQuoteToCandle(symbol, quote) {
  const candles = state.candles[symbol];
  if (!candles?.length || state.modes[symbol] !== "RECENT") return;
  const bucket = Math.floor(quote.eventTime / 1800000) * 1800000, last = candles.at(-1);
  if (last.time !== bucket) return;
  last.high = Math.max(last.high, quote.price); last.low = Math.min(last.low, quote.price); last.close = quote.price;
}

function renderQuote(symbol) {
  const quote = state.quotes[symbol]; if (!quote) return;
  el(`${symbol}-price`).textContent = indexPrice(quote.price);
  el(`${symbol}-time`).textContent = `行情时间 ${exchangeTime(quote.eventTime)} · UTC+8`;
  state.quoteDirty.delete(symbol);
}

function flushMarketDisplay() {
  for (const symbol of [...state.quoteDirty]) { renderQuote(symbol); applyQuoteToCandle(symbol, state.quotes[symbol]); }
  updateHealth();
  if (state.candles[state.symbol]) render(state.symbol);
}

function render(symbol) {
  const candles = state.candles[symbol]; if (!candles?.length) return;
  const recent = candles.slice(-48);
  el("chartTitle").textContent = `${symbol.slice(0, 3)} / USDT 指数`;
  const stale = state.updated[symbol] && Date.now() - state.updated[symbol] > CANDLE_STALE_MS;
  const mode = stale ? "STALE" : state.modes[symbol];
  el("dataMode").textContent = { RECENT: "行情快照", STALE: "数据已过期", DEMO: "模拟数据" }[mode];
  el("chartFreshness").textContent = mode === "DEMO" ? "当前为模拟走势，不代表真实市场。每分钟自动尝试恢复。" :
    `指数 K 线快照更新于 ${exchangeTime(state.updated[symbol])}（UTC+8）· 每分钟刷新 · 最后一根 K 线可能尚未收盘${mode === "STALE" ? " · 数据过期，请勿据此决策" : ""}`;
  el("periodHigh").textContent = money(Math.max(...recent.map(c => c.high)));
  el("periodLow").textContent = money(Math.min(...recent.map(c => c.low)));
  const last = candles.at(-1), previous = candles.at(-2);
  el("volatility").textContent = `${Math.abs((last.close / previous.close - 1) * 100).toFixed(2)}%`;
  el("chartLoading").style.display = "none";
  drawChart(candles);
}

function updateQuote(symbol, price, eventTime, source) {
  const now = Date.now(), previous = state.quotes[symbol];
  if (!SYMBOLS.includes(symbol) || !Number.isFinite(price) || price <= 0 ||
      !Number.isInteger(eventTime) || now - eventTime > QUOTE_STALE_MS || eventTime > now + 5000) return false;
  // An older REST response must not overwrite a newer streamed index price.
  if (previous && (eventTime < previous.eventTime ||
      (eventTime === previous.eventTime && (source !== "WS" || previous.source === "WS")))) return false;
  state.quotes[symbol] = { price, eventTime, receivedAt: now, source };
  state.quoteDirty.add(symbol);
  if (!el(`${symbol}-price`).textContent || el(`${symbol}-price`).textContent === "—") renderQuote(symbol);
  updateHealth();
  return true;
}

function freshQuote(quote, maxAge = QUOTE_STALE_MS) {
  return Boolean(quote && Date.now() - quote.eventTime <= maxAge && Date.now() - quote.receivedAt <= maxAge);
}

async function loadIndexQuote(symbol, force = false) {
  const quote = state.quotes[symbol];
  if (state.quotePending.has(symbol) || (!force && quote?.source === "WS" && freshQuote(quote, STREAM_STALE_MS))) return;
  state.quotePending.add(symbol);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch(`${API}/premiumIndex?symbol=${symbol}`, { signal: controller.signal, cache: "no-store" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const snapshot = await response.json();
    if (snapshot.symbol !== symbol || typeof snapshot.indexPrice !== "string" || !snapshot.indexPrice.trim()) throw new Error("Invalid index snapshot");
    updateQuote(symbol, +snapshot.indexPrice, snapshot.time, "REST");
  } catch (error) { console.warn(`${symbol} index quote unavailable`, error); }
  finally { clearTimeout(timeout); state.quotePending.delete(symbol); updateHealth(); }
}

const refreshIndexQuotes = (force = false) => Promise.all(SYMBOLS.map(symbol => loadIndexQuote(symbol, force)));

function updateHealth() {
  const live = SYMBOLS.every(symbol => state.quotes[symbol]?.source === "WS" && freshQuote(state.quotes[symbol], STREAM_STALE_MS));
  const recent = SYMBOLS.every(symbol => freshQuote(state.quotes[symbol]));
  el("statusDot").className = `status-dot ${live ? "live" : recent ? "" : "offline"}`;
  el("connectionText").textContent = live ? "指数价格实时更新" : recent ? "指数快照 · 实时流连接中" : "指数价格连接中或已过期";
  SYMBOLS.forEach(symbol => {
    const quote = state.quotes[symbol], fresh = freshQuote(quote);
    const streaming = quote?.source === "WS" && freshQuote(quote, STREAM_STALE_MS);
    el(`${symbol}-source`).textContent = streaming ? "实时指数 · 2s 显示刷新" : quote?.source === "REST" ? "指数快照 · 2s 轮询" : "Binance 指数价 · USDT";
    el(`${symbol}-freshness`).textContent = !quote ? "等待指数价格" : !fresh ? "指数价格已过期" :
      `${streaming ? "实时" : "快照"} · ${Math.max(0, Math.floor((Date.now() - quote.eventTime) / 1000))} 秒前`;
    el(`${symbol}-price`).className = fresh ? "" : "stale-quote";
  });
}

function scheduleReconnect() {
  if (state.reconnectTimer !== null) return;
  const delay = Math.min(30000, 1000 * 2 ** Math.min(state.retries++, 5));
  state.reconnectTimer = setTimeout(() => { state.reconnectTimer = null; connectStream(); }, delay);
}

function connectStream() {
  if (state.socket && state.socket.readyState < WebSocket.CLOSING) return;
  let socket;
  try { socket = new WebSocket(STREAM); }
  catch (error) { console.warn("Index stream unavailable", error); scheduleReconnect(); return; }
  state.socket = socket; state.streamSince = Date.now(); state.streamLast = 0;
  socket.onmessage = event => {
    if (socket !== state.socket) return;
    try {
      const frame = JSON.parse(event.data), ticker = frame.data ?? frame;
      if (ticker.e !== "markPriceUpdate" || (ticker.st !== undefined && ticker.st !== 1) ||
          typeof ticker.i !== "string" || !ticker.i.trim()) return;
      // Binance's i is INDEX price. p is MARK price and must never be substituted.
      if (updateQuote(ticker.s, +ticker.i, ticker.E, "WS")) {
        state.retries = 0; state.streamLast = Date.now();
      }
    } catch (error) { console.warn("Invalid market message", error); }
  };
  socket.onerror = () => socket.close();
  socket.onclose = () => {
    if (socket !== state.socket) return;
    state.socket = null;
    // Keep the last observed value and timestamp, but downgrade disconnected streams.
    for (const quote of Object.values(state.quotes)) if (quote.source === "WS") quote.source = "DISCONNECTED";
    updateHealth(); refreshIndexQuotes(); scheduleReconnect();
  };
}

function checkMarketHealth() {
  const socket = state.socket;
  if (socket && (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN) &&
      Date.now() - Math.max(state.streamSince, state.streamLast) > 15000) socket.close();
  updateHealth();
}

document.querySelectorAll(".price-card").forEach(card => card.addEventListener("click", () => {
  document.querySelectorAll(".price-card").forEach(item => { item.classList.remove("active"); item.setAttribute("aria-pressed", "false"); });
  card.setAttribute("aria-pressed", "true");
  card.classList.add("active"); state.symbol = card.dataset.symbol;
  if (state.candles[state.symbol]) render(state.symbol);
}));
window.addEventListener("resize", () => state.candles[state.symbol] && render(state.symbol));
setInterval(tickClock, 1000); tickClock();
const refreshCandles = () => Promise.all(SYMBOLS.map(loadCandles));
el("refreshMarket").addEventListener("click", async () => {
  el("refreshMarket").disabled = true;
  try { await Promise.all([refreshCandles(), refreshIndexQuotes(true)]); }
  finally { el("refreshMarket").disabled = false; }
});
setInterval(refreshCandles, 60000);
setInterval(() => refreshIndexQuotes(), 2000);
setInterval(checkMarketHealth, 1000);
setInterval(flushMarketDisplay, 2000);
document.addEventListener("visibilitychange", () => { if (!document.hidden) { refreshCandles(); refreshIndexQuotes(); checkMarketHealth(); } });
refreshCandles(); refreshIndexQuotes(); connectStream();
