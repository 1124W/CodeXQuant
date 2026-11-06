const API = "https://api.binance.com/api/v3";
const STREAM = "wss://stream.binance.com:9443/stream?streams=btcusdt@miniTicker/ethusdt@miniTicker";
const state = { symbol: "BTCUSDT", candles: {}, socket: null, retries: 0 };
const el = id => document.getElementById(id);
const money = value => new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: value >= 1000 ? 0 : 2 }).format(value);

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
  try {
    const response = await fetch(`${API}/klines?symbol=${symbol}&interval=30m&limit=240`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const rows = await response.json();
    state.candles[symbol] = rows.map(row => ({ time: row[0], open: +row[1], high: +row[2], low: +row[3], close: +row[4] }));
    if (!state.candles[symbol].length) throw new Error("No candles");
    if (symbol === state.symbol) el("dataMode").textContent = "LIVE";
  } catch (error) {
    console.warn(`${symbol} candles unavailable`, error);
    state.candles[symbol] = demoCandles(symbol);
    if (symbol === state.symbol) el("dataMode").textContent = "DEMO";
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
  const min = Math.min(...data.map(c => c.low)), max = Math.max(...data.map(c => c.high)), range = max - min || 1;
  const x = index => 8 + index * (width - 20) / (data.length - 1);
  const y = value => 12 + (max - value) / range * (height - 36);
  ctx.strokeStyle = "rgba(255,255,255,.055)";
  for (let i = 0; i < 5; i++) { const level = 12 + i * (height - 36) / 4; ctx.beginPath(); ctx.moveTo(0, level); ctx.lineTo(width, level); ctx.stroke(); }
  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, "rgba(184,243,72,.16)"); gradient.addColorStop(1, "rgba(184,243,72,0)");
  ctx.beginPath(); data.forEach((c, i) => i ? ctx.lineTo(x(i), y(c.close)) : ctx.moveTo(x(i), y(c.close)));
  ctx.lineTo(x(data.length - 1), height); ctx.lineTo(x(0), height); ctx.closePath(); ctx.fillStyle = gradient; ctx.fill();
  for (const [series, color, lineWidth] of [[fast, "#b8f348", 2], [slow, "#57d8ff", 1.4]]) {
    ctx.beginPath(); series.forEach((value, i) => i ? ctx.lineTo(x(i), y(value)) : ctx.moveTo(x(i), y(value)));
    ctx.strokeStyle = color; ctx.lineWidth = lineWidth; ctx.stroke();
  }
  ctx.fillStyle = "#8991a3"; ctx.font = "10px ui-monospace,monospace";
  ctx.fillText(money(min), 8, height - 5); ctx.textAlign = "right"; ctx.fillText(money(max), width - 8, 12);
}

function render(symbol) {
  const candles = state.candles[symbol]; if (!candles?.length) return;
  const recent = candles.slice(-48);
  el("chartTitle").textContent = `${symbol.slice(0, 3)} / USDT`;
  el("periodHigh").textContent = money(Math.max(...recent.map(c => c.high)));
  el("periodLow").textContent = money(Math.min(...recent.map(c => c.low)));
  const last = candles.at(-1), previous = candles.at(-2);
  el("volatility").textContent = `${Math.abs((last.close / previous.close - 1) * 100).toFixed(2)}%`;
  el("chartLoading").style.display = "none";
  drawChart(candles);
}

function updateQuote(symbol, price, change) {
  el(`${symbol}-price`).textContent = money(price);
  const changeEl = el(`${symbol}-change`);
  changeEl.textContent = `24h ${change >= 0 ? "+" : ""}${change.toFixed(2)}%`;
  changeEl.className = change >= 0 ? "up" : "down";
}

function connectStream() {
  const socket = new WebSocket(STREAM); state.socket = socket;
  socket.onopen = () => { state.retries = 0; el("statusDot").className = "status-dot live"; el("connectionText").textContent = "实时行情已连接"; };
  socket.onmessage = event => { const ticker = JSON.parse(event.data).data; updateQuote(ticker.s, +ticker.c, (+ticker.c / +ticker.o - 1) * 100); };
  socket.onerror = () => socket.close();
  socket.onclose = () => { el("statusDot").className = "status-dot offline"; el("connectionText").textContent = "行情重连中"; setTimeout(connectStream, Math.min(30000, 1000 * 2 ** state.retries++)); };
}

document.querySelectorAll(".price-card").forEach(card => card.addEventListener("click", () => {
  document.querySelectorAll(".price-card").forEach(item => item.classList.remove("active"));
  card.classList.add("active"); state.symbol = card.dataset.symbol;
  if (state.candles[state.symbol]) render(state.symbol);
}));
window.addEventListener("resize", () => state.candles[state.symbol] && render(state.symbol));
setInterval(tickClock, 1000); tickClock();
loadCandles("BTCUSDT"); loadCandles("ETHUSDT"); connectStream();
