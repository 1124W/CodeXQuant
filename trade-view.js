(function (global, factory) {
  'use strict';
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (global && global.document) {
    global.CodeXTradeView = api;
    api.mount(global.document).catch(error => {
      const element = global.document.getElementById('tradeError');
      if (element) { element.hidden = false; element.textContent = '交易图暂时无法读取：' + error.message; }
    });
  }
})(typeof window !== 'undefined' ? window : null, function () {
  'use strict';
  const SYMBOLS = ['BTCUSDT', 'ETHUSDT'];
  const START = Date.UTC(2026, 7, 9), END = Date.UTC(2026, 10, 9);
  const iso = ms => new Date(ms).toISOString().slice(0, 19);
  const utc = ms => iso(ms).replace('T', ' ') + ' UTC';
  const price = value => Number(value).toLocaleString('en-US', {minimumFractionDigits:2, maximumFractionDigits:8});
  const money = value => (value > 0 ? '+' : '') + value.toFixed(2) + ' USDT';

  function validateData(data) {
    if (!data || !data.meta || !Array.isArray(data.trades) || data.trades.length !== 442) throw Error('442 笔原始流水不完整');
    if (data.meta.start_ms !== START || data.meta.end_exclusive_ms !== END || data.meta.payout !== .85 || data.meta.fee !== 0 || data.meta.stake !== 5) throw Error('回测区间或赔付口径不一致');
    const ids = new Set();
    let wins = 0, losses = 0, net = 0;
    const counts = {BTCUSDT:0, ETHUSDT:0};
    for (const trade of data.trades) {
      if (!Number.isInteger(trade.id) || ids.has(trade.id) || !SYMBOLS.includes(trade.symbol)) throw Error('交易编号或币种异常');
      ids.add(trade.id); counts[trade.symbol]++;
      if (![trade.entry_ms, trade.settlement_ms, trade.signal_ms].every(Number.isSafeInteger) || trade.signal_ms < START || trade.settlement_ms >= END || trade.entry_ms !== trade.signal_ms + 60000 || trade.settlement_ms !== trade.entry_ms + 1800000) throw Error('交易执行时间不一致');
      if (![trade.entry,trade.exit,trade.net_units].every(Number.isFinite) || trade.entry <= 0 || trade.exit <= 0 || ![1,-1].includes(trade.direction)) throw Error('执行价格异常');
      const change = (trade.exit - trade.entry) * trade.direction;
      const outcome = change > 0 ? 'win' : change < 0 ? 'loss' : 'refund';
      const expected = outcome === 'win' ? 4.25 : outcome === 'loss' ? -5 : 0;
      if (trade.outcome !== outcome || Math.abs(trade.net_units - expected) > 1e-8) throw Error('胜负或结算金额不一致');
      wins += outcome === 'win'; losses += outcome === 'loss'; net += trade.net_units;
    }
    if (wins !== 251 || losses !== 191 || Math.abs(net - 111.75) > 1e-8 || counts.BTCUSDT !== 227 || counts.ETHUSDT !== 215) throw Error('流水汇总与原报告不一致');
    for (const symbol of SYMBOLS) {
      const bars = data.candles && data.candles[symbol];
      if (!Array.isArray(bars) || bars.length !== 4416) throw Error('指数 K 线不完整');
      for (let index = 0; index < bars.length; index++) {
        const bar = bars[index];
        if (!Array.isArray(bar) || bar.length !== 5 || !bar.every(Number.isFinite) || bar[0] * 1000 !== START + index * 1800000 || bar[3] > Math.min(bar[1],bar[4]) || bar[2] < Math.max(bar[1],bar[4])) throw Error('指数 K 线网格或价格异常');
      }
    }
    return data;
  }

  function cohort(data, symbol) {
    if (!SYMBOLS.includes(symbol)) throw Error('仅支持 BTC 与 ETH');
    return data.trades.filter(trade => trade.symbol === symbol);
  }

  function markerTrace(trades, terminal, name, marker, color) {
    const stamp = terminal ? 'settlement_ms' : 'entry_ms', field = terminal ? 'exit' : 'entry';
    return {type:'scatter', mode:'markers', name,
      x:trades.map(trade => iso(trade[stamp])), y:trades.map(trade => trade[field]),
      customdata:trades.map(trade => [trade.id, trade.direction === 1 ? '看涨' : '看跌', money(trade.net_units), price(trade[field]), utc(trade.entry_ms), price(trade.entry), utc(trade.settlement_ms), price(trade.exit)]),
      marker:{symbol:marker, size:9, color, opacity:.9},
      hovertemplate:'#%{customdata[0]} · %{customdata[1]} · ' + name + '<br>%{x|%m-%d %H:%M} UTC<br>%{customdata[3]} USDT<br>开仓 %{customdata[4]}<br>%{customdata[5]} USDT<br>到期 %{customdata[6]}<br>%{customdata[7]} USDT<br>净收益 %{customdata[2]}<extra></extra>'};
  }

  function buildTraces(data, symbol, selected, colors) {
    const trades = cohort(data,symbol), candles = data.candles[symbol], connectorX = [], connectorY = [];
    for (const trade of trades) { connectorX.push(iso(trade.entry_ms),iso(trade.settlement_ms),null); connectorY.push(trade.entry,trade.exit,null); }
    return [
      {type:'candlestick', name:'指数 30m K 线', showlegend:false, x:candles.map(bar => iso(bar[0]*1000)), open:candles.map(bar=>bar[1]), high:candles.map(bar=>bar[2]), low:candles.map(bar=>bar[3]), close:candles.map(bar=>bar[4]), increasing:{line:{color:colors.muted,width:1},fillcolor:colors.bg}, decreasing:{line:{color:colors.muted,width:1},fillcolor:colors.panel}, hovertext:candles.map(bar=>utc(bar[0]*1000)+'<br>开 '+price(bar[1])+' · 高 '+price(bar[2])+'<br>低 '+price(bar[3])+' · 收 '+price(bar[4])), hoverinfo:'text'},
      {type:'scatter', mode:'lines', showlegend:false, hoverinfo:'skip', x:connectorX, y:connectorY,line:{color:colors.muted,width:1},opacity:.4},
      markerTrace(trades.filter(trade=>trade.direction===1),false,'看涨开仓','triangle-up',colors.long),
      markerTrace(trades.filter(trade=>trade.direction===-1),false,'看跌开仓','triangle-down',colors.short),
      markerTrace(trades,true,'到期结算','diamond-open',colors.expiry),
      {type:'scatter',mode:'lines+markers',name:'所选交易',showlegend:false,x:[iso(selected.entry_ms),iso(selected.settlement_ms)],y:[selected.entry,selected.exit],customdata:[[selected.id],[selected.id]],marker:{size:13,symbol:[selected.direction===1?'triangle-up':'triangle-down','diamond-open'],color:colors.long},line:{color:colors.long,width:2},hovertemplate:'#%{customdata[0]}<br>%{x|%m-%d %H:%M} UTC<br>%{y:.8f} USDT<extra>所选交易</extra>'}
    ];
  }

  function focusRange(trade) {
    return [iso(Math.max(START,trade.entry_ms-6*3600000)),iso(Math.min(END,trade.settlement_ms+6*3600000))];
  }

  function signalText(signal) {
    if (!signal || !SYMBOLS.includes(signal.symbol)) return '暂无信号记录';
    const direction = signal.direction;
    const side = direction === 1 || direction === 'long' || direction === 'up' ? '看涨 / 多' : direction === -1 || direction === 'short' || direction === 'down' ? '看跌 / 空' : null;
    if (!side) return '暂无方向信号';
    const stamp = signal.signal_ms || signal.timestamp_ms || signal.created_at_ms;
    return signal.symbol.slice(0,3) + ' · ' + side + (Number.isFinite(stamp) ? ' · ' + utc(stamp) : '');
  }

  const REASONS = {
    'model_score_below_0.56':'模型分数未达门槛',
    model_does_not_oppose_60m_move:'方向不满足反转条件', '60m_move_not_extreme':'近 60 分钟波动未达门槛',
    short_volatility_expanding:'短期波动正在扩张', paper_position_open:'已有模拟持仓',
    paper_daily_risk_budget:'当日风险预算已用完', paper_capital_insufficient:'模拟资金不足',
    duplicate_paper_signal:'该信号已处理', paper_risk_admitted:'已纳入模拟跟踪', signal:'策略条件满足',
    late_skipped:'预测耗时过长，信号已过期', late_signal_skipped:'预测耗时过长，信号已过期'
  };

  function decisionText(decision) {
    if (!decision || !SYMBOLS.includes(decision.symbol)) return '等待本轮扫描';
    const label = decision.symbol.slice(0,3);
    const reason = decision.paper_risk_status || decision.reason;
    const score = Number(decision.confidence_score);
    return label + ' · ' + (REASONS[reason] || '等待策略判断') +
      (Number.isFinite(score) && score >= .5 && score <= 1 ? ' · 方向分数 ' + (score*100).toFixed(1) + '%' : '');
  }

  function notificationText(value) {
    const n=value || {};
    if(n.enabled===false)return '未启用推送';
    if(!n.enabled)return '等待推送配置';
    const sent=Number(n.last_sent_at || n.last_sent_ms || 0),failed=Number(n.last_error_at || 0);
    const queued=Number(n.pending_count || 0), exhausted=Number(n.exhausted_count || 0);
    let status=sent?'已发送 '+Number(n.sent_count || 0)+' 条':'已启用，等待消息';
    if(queued>0)status+=' · '+queued+' 条待处理';
    if(n.last_error && (!sent || failed>=sent))status+=' · 最近投递异常';
    if(exhausted>0)status+=' · '+exhausted+' 条已耗尽重试';
    if(Number(n.delivery_uncertain_count)>0)status+=' · 存在回执不确定记录';
    return status;
  }

  function settlementText(row) {
    if(!row || !SYMBOLS.includes(row.symbol))return '暂无已结算模拟信号';
    const outcome={win:'赢',loss:'亏',refund:'持平退回'}[row.outcome];
    if(!outcome || !Number.isFinite(row.net_usdt))return '等待结算结果';
    const direction=row.direction===1?'看涨 / 多':'看跌 / 空';
    const stamp=row.settlement_ms || row.expiry_due_ms;
    return row.symbol.slice(0,3)+' · '+direction+' · '+outcome+' · '+money(row.net_usdt)+(Number.isFinite(stamp)?' · '+utc(stamp):'');
  }

  function paperText(paper) {
    if(!paper || !Number.isFinite(paper.settled))return '等待服务器模拟记录';
    const rate=Number.isFinite(paper.win_rate)?(paper.win_rate*100).toFixed(2)+'%':'暂无已决胜负';
    return paper.settled+' 笔已结算 · '+Number(paper.wins || 0)+' 胜 / '+Number(paper.losses || 0)+' 负 / '+Number(paper.refunds || 0)+' 平 · 胜率 '+rate+
      (Number.isFinite(paper.net_usdt)?' · '+money(paper.net_usdt):'')+' · '+Number(paper.pending || 0)+' 笔待结算';
  }

  function scanText(scan) {
    if(!scan || !Number.isFinite(scan.evaluated_slots))return '等待扫描统计';
    return '近 60 分钟完成 '+scan.evaluated_slots+' 轮 · 满足策略 '+Number(scan.candidates || 0)+' 个 · 风控接纳 '+Number(scan.admitted || 0)+' 个'+
      (Number(scan.late_slots)>0?' · 超时跳过 '+scan.late_slots+' 轮':'');
  }

  function monitorText(state,now) {
    const age=Number.isFinite(state.lastPrediction)?now-state.lastPrediction:Infinity;
    const active=state.dataHealth==='fresh' && age>=-60000 && age<120000;
    return (active?'每分钟机会扫描中':'扫描等待或异常')+' · 自动下单关闭';
  }

  function normalizeStatus(payload) {
    const state = payload && payload.data && typeof payload.data === 'object' ? payload.data : payload || {};
    const symbols = state.symbols || state.prices || {};
    const quotes = {};
    for (const symbol of SYMBOLS) {
      const item = symbols[symbol] || {};
      const value = Number(item.index_price ?? item.price ?? item.indexPrice);
      const timestamp = Number(item.price_time_ms ?? item.timestamp_ms ?? item.time ?? 0);
      quotes[symbol] = {price:Number.isFinite(value) && value>0 ? value : null, timestamp:Number.isFinite(timestamp) && timestamp>0 ? timestamp : null, stale:Boolean(item.stale)};
    }
    const model = state.model || {};
    return {quotes, mode:state.mode || 'research', modelId:state.model_id || model.id || model.model_id || '等待模型准备',
      signal:state.last_signal || (Array.isArray(state.signals) && state.signals.length ? state.signals[state.signals.length-1] : null),
      notifications:state.notifications || state.notifier || {}, updatedAt:state.updated_at_ms || null,
      lastPrediction:state.last_prediction_ms || null, decisions:Array.isArray(state.last_decisions)?state.last_decisions.filter(row=>row&&SYMBOLS.includes(row.symbol)):[],
      predictionSchedule:state.prediction_schedule || '', dataHealth:state.data_health || 'unknown', lastError:state.last_error || null,
      scan:state.scan_stats || {}, paper:state.paper_performance || {}, settlements:Array.isArray(state.recent_settlements)?state.recent_settlements:[]};
  }

  async function mount(document) {
    const root = document.getElementById('tradeView');
    if (!root) return;
    const window = document.defaultView;
    mountLive(document,window);
    const response = await window.fetch(root.dataset.tradesUrl,{cache:'no-store'});
    if (!response.ok) throw Error('历史数据读取失败');
    const data = validateData(await response.json());
    const symbolSelect = document.getElementById('tradeSymbol'), select = document.getElementById('tradeSelect');
    const previous = document.getElementById('tradePrevious'), next = document.getElementById('tradeNext');
    const chart = document.getElementById('tradeChart'), detail = document.getElementById('tradeDetail');
    const count = document.getElementById('tradeCount'), error = document.getElementById('tradeError');
    let symbol = 'BTCUSDT', trades = [], selected, busy = false, resizeTimer;
    const css = getComputedStyle(document.documentElement);
    const colors={bg:css.getPropertyValue('--bg').trim(),panel:css.getPropertyValue('--panel').trim(),text:css.getPropertyValue('--text').trim(),muted:css.getPropertyValue('--muted').trim(),border:css.getPropertyValue('--line').trim(),long:css.getPropertyValue('--cyan').trim(),short:css.getPropertyValue('--amber').trim(),expiry:css.getPropertyValue('--lime').trim()};
    function details() {
      const index=trades.indexOf(selected),wins=trades.filter(trade=>trade.outcome==='win').length,net=trades.reduce((sum,trade)=>sum+trade.net_units,0);
      count.textContent=symbol.slice(0,3)+' '+trades.length+' 笔 · '+wins+' 胜 · '+money(net)+' · 全部 442 笔';
      previous.disabled=busy||index<=0;next.disabled=busy||index>=trades.length-1;select.disabled=busy;symbolSelect.disabled=busy;
      detail.replaceChildren();
      const heading=document.createElement('strong');heading.textContent='#'+selected.id+' · '+symbol.slice(0,3)+' · '+(selected.direction===1?'看涨 / 多':'看跌 / 空')+' · '+money(selected.net_units);heading.className=selected.net_units>0?'positive':selected.net_units<0?'negative':'';
      const cells=document.createElement('div');cells.className='detail-prices';
      for (const [label,stamp,value] of [['开仓',selected.entry_ms,selected.entry],['到期结算',selected.settlement_ms,selected.exit]]) {
        const cell=document.createElement('div'),labelNode=document.createElement('small'),valueNode=document.createElement('span');
        labelNode.textContent=label+' · '+utc(stamp);valueNode.textContent=price(value)+' USDT';cell.append(labelNode,valueNode);cells.append(cell);
      }
      detail.append(heading,cells);select.value=String(selected.id);
    }
    function fitY() {
      if (!window.Plotly || !chart._fullLayout?.xaxis) return;
      const range=chart._fullLayout.xaxis.range.map(value=>typeof value==='number'?value:Date.parse(value.endsWith('Z')?value:value+'Z')),values=[];
      for (const bar of data.candles[symbol]) if (bar[0]*1000>=range[0]-1800000&&bar[0]*1000<=range[1]) values.push(bar[3],bar[2]);
      for (const trade of trades) { if(trade.entry_ms>=range[0]&&trade.entry_ms<=range[1])values.push(trade.entry);if(trade.settlement_ms>=range[0]&&trade.settlement_ms<=range[1])values.push(trade.exit); }
      if(!values.length)return;const low=Math.min(...values),high=Math.max(...values),pad=Math.max((high-low)*.12,high*.0002);
      window.Plotly.relayout(chart,{'yaxis.range':[low-pad,high+pad]});
    }
    async function render() {
      details();
      if(!window.Plotly){error.hidden=false;error.textContent='图表资源暂未加载，仍可逐笔查看准确价格。';return;}
      busy=true;details();
      const narrow=root.clientWidth<500;
      await window.Plotly.react(chart,buildTraces(data,symbol,selected,colors),{autosize:true,height:narrow?400:470,paper_bgcolor:'rgba(0,0,0,0)',plot_bgcolor:'rgba(0,0,0,0)',margin:{l:narrow?68:82,r:12,t:40,b:46},font:{family:'system-ui, sans-serif',size:13,color:colors.text},hoverlabel:{bgcolor:colors.bg,bordercolor:colors.border,font:{color:colors.text,size:12}},legend:{orientation:'h',x:0,y:1.15,font:{size:12}},xaxis:{type:'date',range:focusRange(selected),title:{text:'时间（UTC）',standoff:4},gridcolor:colors.border,zeroline:false,tickfont:{size:12},nticks:narrow?3:6,tickformat:'%m-%d<br>%H:%M',rangeslider:{visible:true,range:[iso(START),iso(END)],thickness:.16,bgcolor:colors.bg,borderwidth:0,yaxis:{rangemode:'auto'}}},yaxis:{title:{text:'USDT',standoff:4},automargin:true,gridcolor:colors.border,zeroline:false,tickfont:{size:12},tickformat:',.2f'},hovermode:'closest',dragmode:'pan',uirevision:symbol},{responsive:true,displaylogo:false,scrollZoom:true,displayModeBar:false});
      chart.removeAllListeners('plotly_click');chart.removeAllListeners('plotly_relayout');
      chart.on('plotly_click',event=>{const point=event.points.find(point=>point.customdata&&Number.isInteger(point.customdata[0]));if(point&&!busy) choose(point.customdata[0]);});
      chart.on('plotly_relayout',event=>{if(!busy&&(event['xaxis.range[0]']||event['xaxis.range']||event['xaxis.autorange']))fitY();});
      busy=false;details();fitY();
    }
    async function choose(id) {
      if(busy)return;selected=trades.find(trade=>trade.id===Number(id))||trades[0];details();
      if(window.Plotly&&chart.data){busy=true;details();await window.Plotly.restyle(chart,{x:[[iso(selected.entry_ms),iso(selected.settlement_ms)]],y:[[selected.entry,selected.exit]],customdata:[[[selected.id],[selected.id]]],'marker.symbol':[[selected.direction===1?'triangle-up':'triangle-down','diamond-open']]},[5]);await window.Plotly.relayout(chart,{'xaxis.range':focusRange(selected)});busy=false;fitY();details();}
    }
    async function changeSymbol() {
      if(busy)return;symbol=symbolSelect.value;trades=cohort(data,symbol);selected=trades[0];
      select.replaceChildren(...trades.map(trade=>{const option=document.createElement('option');option.value=trade.id;option.textContent='#'+trade.id+' · '+iso(trade.entry_ms).slice(5,16).replace('T',' ')+' · '+(trade.direction===1?'看涨':'看跌')+' · '+(trade.outcome==='win'?'赢':'亏');return option;}));
      await render();
    }
    symbolSelect.addEventListener('change',changeSymbol);select.addEventListener('change',()=>choose(select.value));
    previous.addEventListener('click',()=>choose(trades[Math.max(0,trades.indexOf(selected)-1)].id));next.addEventListener('click',()=>choose(trades[Math.min(trades.length-1,trades.indexOf(selected)+1)].id));
    if(window.ResizeObserver)new window.ResizeObserver(()=>{window.clearTimeout(resizeTimer);resizeTimer=window.setTimeout(()=>{if(window.Plotly&&chart.data)window.Plotly.Plots.resize(chart);},100);}).observe(root);
    document.getElementById('tradeProvenance').textContent='运行 '+data.meta.source_run_id+' · 884 个执行价格已核对官方指数分钟开盘价 · K 线保留两位小数，交易价格保留原始精度。';
    await changeSymbol();
  }

  function mountLive(document,window) {
    const mode=document.getElementById('liveMode'),model=document.getElementById('liveModel'),signal=document.getElementById('liveSignal'),notify=document.getElementById('liveNotify'),note=document.getElementById('liveNote');
    if(!mode)return;
    let timer=null,pending=false,stopped=false;
    const setText=(id,text)=>{const element=document.getElementById(id);if(element)element.textContent=text;};
    function staticMode(){stopped=true;mode.textContent='静态回测，实时预测需要服务端';model.textContent='静态回测';signal.textContent='部署预测服务后显示';notify.textContent='由服务器后台执行';note.textContent='历史图表可正常浏览。每分钟扫描、模拟结算和群推送仅由部署后的后台运行。';setText('liveScan','静态页面不执行预测');setText('livePaper','连接服务器后显示模拟统计');}
    if(window.location.hostname.endsWith('github.io')||window.location.protocol==='file:'){staticMode();return;}
    async function poll(){
      if(pending||stopped)return;pending=true;
      const controller=new window.AbortController(),timeout=window.setTimeout(()=>controller.abort(),8000);
      try{
        const response=await window.fetch('/api/live/status',{cache:'no-store',signal:controller.signal});
        if(response.status===404){staticMode();return;}
        if(!response.ok)throw Error('unavailable');
        const state=normalizeStatus(await response.json());mode.textContent=monitorText(state,Date.now());model.textContent=state.modelId;signal.textContent=signalText(state.signal);
        for(const [symbol,prefix]of [['BTCUSDT','Btc'],['ETHUSDT','Eth']]){const quote=state.quotes[symbol];document.getElementById('live'+prefix+'Price').textContent=quote.price===null?'—':quote.price.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2});document.getElementById('live'+prefix+'Time').textContent=quote.timestamp?utc(quote.timestamp)+(quote.stale?' · 行情已过期':''):'等待真实行情';}
        const n=state.notifications;notify.textContent=notificationText(n);
        setText('liveScan',scanText(state.scan));
        setText('liveScanTime',state.lastPrediction?'最近成功判断：'+utc(state.lastPrediction):'尚未完成本轮预测');
        for(const [symbol,prefix]of [['BTCUSDT','Btc'],['ETHUSDT','Eth']])setText('live'+prefix+'Decision',decisionText(state.decisions.find(row=>row.symbol===symbol)));
        setText('livePaper',paperText(state.paper));
        setText('liveSettlement',settlementText(state.settlements[0]));
        const kinds=n.sent_by_kind || {};
        setText('liveNotifyKinds',n.enabled?'已确认发送：信号 '+Number(kinds.signal || 0)+' · 结算 '+Number(kinds.settlement || 0)+' · 研究播报 '+Number(kinds.research_report || 0)+' · 行情 '+Number(kinds.price || 0):'推送分类统计等待服务端');
        const rows=document.getElementById('liveSettlementRows');
        if(rows){
          rows.replaceChildren();
          for(const row of state.settlements.filter(row=>SYMBOLS.includes(row.symbol)).slice(0,10)){
            const tr=document.createElement('tr');
            const values=[row.symbol.slice(0,3)+' '+(row.direction===1?'多':'空'),Number.isFinite(row.entry_ms)?utc(row.entry_ms):'—',Number.isFinite(row.entry_price)?price(row.entry_price):'—',Number.isFinite(row.settlement_ms)?utc(row.settlement_ms):'—',Number.isFinite(row.exit_price)?price(row.exit_price):'—',Number.isFinite(row.net_usdt)?money(row.net_usdt):'待结算'];
            for(const value of values){const td=document.createElement('td');td.textContent=value;tr.append(td);}rows.append(tr);
          }
        }
        note.textContent=(state.lastError?'服务提示：'+state.lastError+'。':'')+(state.predictionSchedule||'每分钟使用已收盘数据判断；有信号才提醒，计划入场后持有 30 分钟。')+' 模型方向分数未经胜率校准；当前模拟表现独立于下方 442 笔历史回测。北京时间每小时播报行情与研究汇总。';
      }catch(error){mode.textContent='预测服务暂不可用';note.textContent='历史图表不受影响；正在等待后台恢复，页面每 10 秒重试。';}
      finally{window.clearTimeout(timeout);pending=false;if(!stopped)timer=window.setTimeout(poll,10000);}
    }
    window.addEventListener('pagehide',()=>{stopped=true;window.clearTimeout(timer);});poll();
  }

  return {validateData,cohort,buildTraces,focusRange,normalizeStatus,signalText,decisionText,notificationText,settlementText,paperText,scanText,monitorText,mount};
});
