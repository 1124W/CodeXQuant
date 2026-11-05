const API = "https://api.binance.com/api/v3";
const STREAM = "wss://stream.binance.com:9443/stream?streams=btcusdt@miniTicker/ethusdt@miniTicker";
const state = { symbol: "BTCUSDT", candles: {}, quotes: {}, socket: null, retries: 0 };
const $ = (id) => document.getElementById(id);

const money = (value) => new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: value >= 1000 ? 0 : 2 }).format(value);
const pct = (value, digits = 2) => `${value >= 0 ? "+" : ""}${value.toFixed(digits)}%`;

function tickClock() {
  $("clock").textContent = new Intl.DateTimeFormat("zh-CN", { timeZone: "Asia/Shanghai", hour12: false, hour: "2-digit", minute: "2-digit", second: "2-digit" }).format(new Date());
}

async function loadCandles(symbol) {
  $("chartLoading").style.display = "grid";
  try {
    const response = await fetch(`${API}/klines?symbol=${symbol}&interval=30m&limit=240`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const rows = await response.json();
    state.candles[symbol] = rows.map((row) => ({ time: row[0], open:+row[1], high:+row[2], low:+row[3], close:+row[4] }));
    $("dataMode").textContent = "LIVE";
  } catch (error) {
    console.warn("Binance history unavailable, using demo series", error);
    state.candles[symbol] = demoCandles(symbol);
    $("dataMode").textContent = "DEMO";
  }
  if (symbol === state.symbol) render(symbol);
  $("chartLoading").style.display = "none";
}

function demoCandles(symbol) {
  const start = symbol === "BTCUSDT" ? 68200 : 3650;
  let price = start;
  return Array.from({ length: 240 }, (_, index) => {
    const open = price;
    price *= 1 + Math.sin(index / 11) * .0011 + (Math.random() - .49) * .006;
    const close = price;
    const spread = close * (.002 + Math.random() * .004);
    return { time: Date.now() - (239 - index) * 1800000, open, close, high:Math.max(open, close) + spread, low:Math.min(open, close) - spread };
  });
}

function ema(values, period) {
  const k = 2 / (period + 1); let current = values[0];
  return values.map((value, index) => current = index ? value * k + current * (1 - k) : value);
}

function rsi(values, period = 14) {
  if (values.length <= period) return 50;
  let gains = 0, losses = 0;
  for (let i = values.length - period; i < values.length; i++) { const diff = values[i] - values[i - 1]; gains += Math.max(diff, 0); losses += Math.max(-diff, 0); }
  return losses === 0 ? 100 : 100 - (100 / (1 + gains / losses));
}

function volatility(values, period = 20) {
  const returns = values.slice(-period - 1).map((v, i, arr) => i ? (v / arr[i - 1] - 1) * 100 : 0).slice(1);
  const avg = returns.reduce((a,b) => a+b, 0) / returns.length;
  return Math.sqrt(returns.reduce((sum,v) => sum + (v-avg) ** 2, 0) / returns.length);
}

function analyze(candles) {
  const closes = candles.map(c => c.close), fast = ema(closes, 9), slow = ema(closes, 21);
  const currentRsi = rsi(closes), vol = volatility(closes), last = closes.at(-1);
  const bullish = fast.at(-1) > slow.at(-1) && last > fast.at(-1);
  const bearish = fast.at(-1) < slow.at(-1) && last < fast.at(-1);
  const volSafe = vol < 1.15;
  let signal = "观望", confidence = 48;
  if (bullish && currentRsi > 48 && currentRsi < 70 && volSafe) { signal = "偏多"; confidence = 72 + Math.min(18, Math.abs(fast.at(-1)/slow.at(-1)-1)*3500); }
  if (bearish && currentRsi < 52 && currentRsi > 30 && volSafe) { signal = "偏空"; confidence = 72 + Math.min(18, Math.abs(fast.at(-1)/slow.at(-1)-1)*3500); }
  return { closes, fast, slow, rsi:currentRsi, vol, bullish, bearish, volSafe, signal, confidence:Math.round(confidence) };
}

function backtest(candles) {
  const closes = candles.map(c => c.close), fast = ema(closes, 9), slow = ema(closes, 21); let wins=0, losses=0, grossWin=0, grossLoss=0, equity=0, peak=0, maxDD=0;
  for (let i=22; i<candles.length-1; i++) {
    const long = fast[i] > slow[i] && fast[i-1] <= slow[i-1];
    const short = fast[i] < slow[i] && fast[i-1] >= slow[i-1];
    if (!long && !short) continue;
    const result = (candles[i+1].close / candles[i].close - 1) * (long ? 1 : -1);
    equity += result; peak = Math.max(peak, equity); maxDD = Math.max(maxDD, peak-equity);
    if (result > 0) { wins++; grossWin += result; } else { losses++; grossLoss += Math.abs(result); }
  }
  const total=wins+losses;
  return { total, winRate:total ? wins/total*100 : 0, pf:grossLoss ? grossWin/grossLoss : 0, maxDD:maxDD*100 };
}

function render(symbol) {
  const candles = state.candles[symbol]; if (!candles?.length) return;
  const result = analyze(candles), test = backtest(candles), latest=candles.at(-1);
  $("chartTitle").textContent = `${symbol.slice(0,3)} / USDT`;
  $("periodHigh").textContent = money(Math.max(...candles.slice(-48).map(c=>c.high)));
  $("periodLow").textContent = money(Math.min(...candles.slice(-48).map(c=>c.low)));
  $("volatility").textContent = `${result.vol.toFixed(2)}%`;
  $("signal").textContent = result.signal; $("confidence").textContent = `置信度 ${result.confidence}%`;
  $("signalOrbit").className = `signal-orbit ${result.signal === "偏空" ? "sell" : result.signal === "观望" ? "neutral" : ""}`;
  $("trendFactor").textContent = result.bullish ? "多头排列" : result.bearish ? "空头排列" : "结构混合";
  $("rsiFactor").textContent = `${result.rsi.toFixed(1)} · ${result.rsi>70?"过热":result.rsi<30?"过冷":"正常"}`;
  $("volFactor").textContent = result.volSafe ? "通过" : "暂停";
  $("signalNote").textContent = result.signal === "观望" ? "趋势与动量尚未形成一致方向，等待下一根 30m K 线。" : `${result.signal}仅代表模型条件成立；实际执行仍需计入手续费、滑点和风险上限。`;
  $("winRate").textContent = `${test.winRate.toFixed(1)}%`; $("sampleCount").textContent = `最近 ${test.total} 次交叉信号`;
  $("profitFactor").textContent = test.pf ? test.pf.toFixed(2) : "—"; $("maxDrawdown").textContent = `${test.maxDD.toFixed(2)}%`;
  $("riskState").textContent = result.volSafe ? "允许研究" : "高波动暂停"; $("riskDetail").textContent = `30m 波动率 ${result.vol.toFixed(2)}%`;
  if (!state.quotes[symbol]) updateQuote(symbol, latest.close, (latest.close/latest.open-1)*100);
  drawChart(candles, result.fast, result.slow);
}

function drawChart(candles, fast, slow) {
  const canvas=$("chart"), rect=canvas.getBoundingClientRect(), dpr=window.devicePixelRatio||1; canvas.width=rect.width*dpr; canvas.height=rect.height*dpr;
  const ctx=canvas.getContext("2d"); ctx.scale(dpr,dpr); const w=rect.width,h=rect.height,pad={t:12,r:12,b:22,l:8};
  const slice=candles.slice(-80), f=fast.slice(-80), s=slow.slice(-80), all=[...slice.map(c=>c.high),...slice.map(c=>c.low)]; const min=Math.min(...all),max=Math.max(...all),range=max-min||1;
  const x=i=>pad.l+i*(w-pad.l-pad.r)/(slice.length-1), y=v=>pad.t+(max-v)/range*(h-pad.t-pad.b);
  ctx.strokeStyle="rgba(255,255,255,.055)"; ctx.lineWidth=1; for(let i=0;i<5;i++){const yy=pad.t+i*(h-pad.t-pad.b)/4;ctx.beginPath();ctx.moveTo(0,yy);ctx.lineTo(w,yy);ctx.stroke();}
  const gradient=ctx.createLinearGradient(0,0,0,h); gradient.addColorStop(0,"rgba(184,243,72,.16)");gradient.addColorStop(1,"rgba(184,243,72,0)"); ctx.beginPath(); slice.forEach((c,i)=>i?ctx.lineTo(x(i),y(c.close)):ctx.moveTo(x(i),y(c.close)));ctx.lineTo(x(slice.length-1),h);ctx.lineTo(x(0),h);ctx.closePath();ctx.fillStyle=gradient;ctx.fill();
  [[f,"#b8f348",2],[s,"#57d8ff",1.4]].forEach(([series,color,width])=>{ctx.beginPath();series.forEach((v,i)=>i?ctx.lineTo(x(i),y(v)):ctx.moveTo(x(i),y(v)));ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();});
  ctx.fillStyle="#8991a3";ctx.font="10px ui-monospace,monospace";ctx.fillText(money(min),8,h-5);ctx.textAlign="right";ctx.fillText(money(max),w-8,12);
}

function updateQuote(symbol, price, change) {
  state.quotes[symbol]={price,change}; $(`${symbol}-price`).textContent=money(price); const el=$(`${symbol}-change`);el.textContent=`24h ${pct(change)}`;el.className=change>=0?"up":"down";
}

function connectStream() {
  if (state.socket) state.socket.close(); const socket=new WebSocket(STREAM); state.socket=socket;
  socket.onopen=()=>{state.retries=0;$("statusDot").className="status-dot live";$("connectionText").textContent="实时行情已连接";};
  socket.onmessage=event=>{const msg=JSON.parse(event.data).data;updateQuote(msg.s,+msg.c,(+msg.c/+msg.o-1)*100);};
  socket.onerror=()=>socket.close(); socket.onclose=()=>{$("statusDot").className="status-dot offline";$("connectionText").textContent="行情重连中";const wait=Math.min(30000,1000*2**state.retries++);setTimeout(connectStream,wait);};
}

document.querySelectorAll(".price-card").forEach(card=>card.addEventListener("click",async()=>{document.querySelectorAll(".price-card").forEach(c=>c.classList.remove("active"));card.classList.add("active");state.symbol=card.dataset.symbol;if(!state.candles[state.symbol])await loadCandles(state.symbol);else render(state.symbol);}));
window.addEventListener("resize",()=>state.candles[state.symbol]&&render(state.symbol));
setInterval(tickClock,1000); tickClock(); loadCandles("BTCUSDT"); loadCandles("ETHUSDT"); connectStream();
