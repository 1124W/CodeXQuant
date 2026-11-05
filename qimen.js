// Records a user's own Qimen interpretation. This module does not generate a chart or a forecast.
(() => {
  const KEY = "codexquant-qimen-records-v1";
  const STEP = 30 * 60 * 1000;
  const byId = (id) => document.getElementById(id);
  const formatPrice = (value) => Number.isFinite(value) ? new Intl.NumberFormat("en-US", { maximumFractionDigits: 2, minimumFractionDigits: 2 }).format(value) : "—";
  const formatTime = (value) => new Intl.DateTimeFormat("zh-CN", { timeZone: "Asia/Shanghai", year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", hour12: false }).format(value);

  function readRecords() {
    try {
      const value = JSON.parse(localStorage.getItem(KEY) || "[]");
      return Array.isArray(value) ? value.filter(r => r && ["BTCUSDT", "ETHUSDT"].includes(r.symbol) && ["up", "down"].includes(r.direction) && Number.isFinite(r.entryPrice) && Number.isFinite(r.targetAt)) : [];
    } catch { return []; }
  }

  let records = readRecords();
  const save = () => localStorage.setItem(KEY, JSON.stringify(records));
  const cell = (row, value) => { const td = document.createElement("td"); td.textContent = value; row.append(td); return td; };

  function render() {
    const settled = records.filter(r => r.result === "win" || r.result === "loss");
    byId("qimenTotal").textContent = settled.length;
    byId("qimenAccuracy").textContent = settled.length ? `${(settled.filter(r => r.result === "win").length / settled.length * 100).toFixed(1)}%` : "—";
    const body = byId("qimenRows"); body.replaceChildren();
    if (!records.length) { const row = document.createElement("tr"); cell(row, "还没有奇门判断记录").colSpan = 7; body.append(row); return; }
    for (const record of [...records].reverse()) {
      const row = document.createElement("tr");
      cell(row, formatTime(record.recordedAt));
      cell(row, record.symbol.slice(0, 3));
      cell(row, record.direction === "up" ? "看涨" : "看跌");
      cell(row, formatPrice(record.entryPrice));
      cell(row, record.result ? formatPrice(record.exitPrice) : `${formatTime(record.targetAt)} 待结算`);
      const outcome = cell(row, record.result === "win" ? "命中" : record.result === "loss" ? "未命中" : "等待");
      outcome.className = record.result === "win" ? "good" : record.result === "loss" ? "bad" : "";
      cell(row, record.reason || "—");
      body.append(row);
    }
  }

  async function settleOne(record) {
    if (record.result || Date.now() < record.targetAt + 3000) return false;
    const startTime = record.targetAt - STEP;
    const url = `https://api.binance.com/api/v3/klines?symbol=${record.symbol}&interval=30m&startTime=${startTime}&limit=1`;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Binance HTTP ${response.status}`);
    const rows = await response.json();
    if (!rows.length || rows[0][0] !== startTime || rows[0][6] > Date.now()) return false;
    const exitPrice = Number(rows[0][4]);
    if (!Number.isFinite(exitPrice) || exitPrice <= 0) throw new Error("结算价格无效");
    record.exitPrice = exitPrice;
    record.result = (exitPrice - record.entryPrice) * (record.direction === "up" ? 1 : -1) > 0 ? "win" : "loss";
    return true;
  }

  async function refresh() {
    const pending = records.filter(r => !r.result && Date.now() >= r.targetAt + 3000);
    if (!pending.length) { byId("qimenMessage").textContent = "暂无到期记录；结算会在下一个 30 分钟收盘后更新。"; return; }
    byId("qimenRefresh").disabled = true;
    let settled = 0;
    try {
      for (const record of pending) if (await settleOne(record)) settled++;
      if (settled) { save(); render(); }
      byId("qimenMessage").textContent = settled ? `已核对 ${settled} 条到期记录。` : "结算 K 线尚未公布，请稍后刷新。";
    } catch (error) {
      console.warn("Qimen settlement unavailable", error);
      byId("qimenMessage").textContent = "暂时无法获取 Binance 结算价；记录仍保留，请稍后刷新。";
    } finally { byId("qimenRefresh").disabled = false; }
  }

  byId("qimenForm").addEventListener("submit", (event) => {
    event.preventDefault();
    const symbol = byId("qimenSymbol").value;
    const quote = window.quantMarket?.getQuote(symbol);
    if (!quote || quote.source !== "live" || Date.now() - quote.observedAt > 60000 || !Number.isFinite(quote.price) || quote.price <= 0) {
      byId("qimenMessage").textContent = "实时行情未就绪，请等 BTC/ETH 价格连接成功后再记录。";
      return;
    }
    const recordedAt = Date.now();
    const targetAt = Math.floor(recordedAt / STEP) * STEP + STEP;
    records.push({ symbol, direction: byId("qimenDirection").value, reason: byId("qimenReason").value.trim(), entryPrice: quote.price, recordedAt, targetAt });
    records = records.slice(-200);
    try { save(); } catch { byId("qimenMessage").textContent = "浏览器存储不可用，本次判断未保存。"; records.pop(); return; }
    byId("qimenReason").value = "";
    byId("qimenMessage").textContent = `已记录，将在 ${formatTime(targetAt)} 收盘后结算。`;
    render();
  });
  byId("qimenRefresh").addEventListener("click", refresh);
  render();
  if (records.some(r => !r.result && Date.now() >= r.targetAt + 3000)) refresh();
  setInterval(() => { if (records.some(r => !r.result && Date.now() >= r.targetAt + 3000)) refresh(); }, 60000);
})();
