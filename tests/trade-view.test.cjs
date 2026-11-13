const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const view = require('../trade-view.js');
const data = JSON.parse(fs.readFileSync(path.join(__dirname,'../research/trade-chart-data.json'),'utf8'));
const colors = {bg:'transparent',panel:'transparent',muted:'gray',long:'blue',short:'orange',expiry:'green'};

test('audited public dataset preserves exact 442-trade cohort and all 884 prices',()=>{
  assert.equal(view.validateData(data),data);
  assert.equal(view.cohort(data,'BTCUSDT').length,227);
  assert.equal(view.cohort(data,'ETHUSDT').length,215);
  assert.equal(data.trades.reduce((sum,row)=>sum+row.net_units,0),111.75);
});

test('changing a settlement, payout, time boundary, or duplicate id fails closed',()=>{
  for(const mutate of [
    copy=>{copy.trades[0].exit=copy.trades[0].entry+1;},
    copy=>{copy.meta.payout=.8;},
    copy=>{copy.trades[0].settlement_ms=copy.meta.end_exclusive_ms;},
    copy=>{copy.trades[1].id=copy.trades[0].id;}
  ]){const copy=structuredClone(data);mutate(copy);assert.throws(()=>view.validateData(copy));}
});

test('entry and expiry markers retain true minute timestamps instead of candle snapping',()=>{
  for(const symbol of ['BTCUSDT','ETHUSDT']){
    const trades=view.cohort(data,symbol),traces=view.buildTraces(data,symbol,trades[0],colors);
    assert.equal(traces[0].x.length,4416);
    assert.equal(traces[2].x.length+traces[3].x.length,trades.length);
    assert.equal(traces[4].x.length,trades.length);
    assert.equal(traces[5].y[0],trades[0].entry);
    assert.equal(traces[5].y[1],trades[0].exit);
    for(const trace of [traces[2],traces[3]]){
      for(let index=0;index<trace.x.length;index++){
        const id=trace.customdata[index][0],trade=trades.find(row=>row.id===id);
        assert.equal(Date.parse(trace.x[index]+'Z'),trade.entry_ms);
        assert.equal(trace.y[index],trade.entry);
      }
    }
    for(let index=0;index<trades.length;index++){
      assert.equal(Date.parse(traces[4].x[index]+'Z'),trades[index].settlement_ms);
      assert.equal(traces[4].y[index],trades[index].exit);
    }
  }
});

test('selected trade focus stays inside original historical cohort',()=>{
  for(const trade of data.trades){
    const range=view.focusRange(trade).map(stamp=>Date.parse(stamp+'Z'));
    assert.ok(range[0]>=data.meta.start_ms);
    assert.ok(range[1]<=data.meta.end_exclusive_ms);
    assert.ok(range[0]<=trade.entry_ms&&range[1]>=trade.settlement_ms);
  }
});

test('live state exposes genuine positive prices, direction and sanitized public fields',()=>{
  const state=view.normalizeStatus({data:{symbols:{BTCUSDT:{index_price:64500.125,price_time_ms:1794450000000},ETHUSDT:{index_price:-1}},model:{id:'both_regularized_logistic'},last_signal:{symbol:'ETHUSDT',direction:-1,signal_ms:1794450000000},notifications:{enabled:true},private:'should not appear'}});
  assert.equal(state.quotes.BTCUSDT.price,64500.125);
  assert.equal(state.quotes.ETHUSDT.price,null);
  assert.equal(state.modelId,'both_regularized_logistic');
  assert.match(view.signalText(state.signal),/^ETH · 看跌 \/ 空/);
  assert.equal(state.private,undefined);
  assert.equal(view.signalText({symbol:'OTHER',direction:1}),'暂无信号记录');
  assert.equal(view.signalText({symbol:'BTCUSDT',direction:0}),'暂无方向信号');
});

test('HTML uses relative public dataset and no account or bot configuration',()=>{
  const html=fs.readFileSync(path.join(__dirname,'../demo/trades.html'),'utf8');
  assert.match(html,/data-trades-url="\.\.\/research\/trade-chart-data\.json"/);
  assert.match(html,/plotly\.js-dist-min@2\.35\.2/);
  assert.doesNotMatch(html,/webhook\/send|password|api[_-]?key/i);
  assert.ok(fs.readFileSync(path.join(__dirname,'../index.html'),'utf8').includes('demo/trades.html'));
});

test('live paper performance stays separate from the historical 442-trade cohort',()=>{
  const state=view.normalizeStatus({last_prediction_ms:1794450060000,last_decisions:[{symbol:'BTCUSDT',side:'NONE',reason:'model_score_below_0.56',confidence_score:.53}],scan_stats:{evaluated_slots:58,candidates:5,admitted:1},paper_performance:{settled:3,wins:1,losses:1,refunds:1,win_rate:.5,net_usdt:-.75,pending:1},recent_settlements:[{symbol:'ETHUSDT',direction:-1,outcome:'loss',net_usdt:-5,settlement_ms:1794450060000}]});
  assert.match(view.decisionText(state.decisions[0]),/BTC · 模型分数未达门槛 · 方向分数 53.0%/);
  assert.match(view.scanText(state.scan),/完成 58 轮 · 满足策略 5 个 · 风控接纳 1 个/);
  assert.match(view.paperText(state.paper),/3 笔已结算 · 1 胜 \/ 1 负 \/ 1 平 · 胜率 50.00% · -0.75 USDT · 1 笔待结算/);
  assert.match(view.settlementText(state.settlements[0]),/ETH · 看跌 \/ 空 · 亏 · -5.00 USDT/);
  assert.doesNotMatch(view.paperText(state.paper),/56.79|442|111.75/);
  assert.match(view.paperText({settled:0,wins:0,losses:0,refunds:0,win_rate:null,net_usdt:0}),/暂无已决胜负/);
});

test('notification recovery does not show an old error as an active failure',()=>{
  const recovered={enabled:true,sent_count:4,last_sent_at:200,last_error_at:100,last_error:'delivery_uncertain',pending_count:0,exhausted_count:0};
  assert.equal(view.notificationText(recovered),'已发送 4 条');
  assert.match(view.notificationText({...recovered,last_error_at:250,pending_count:1}),/1 条待处理 · 最近投递异常/);
  assert.match(view.notificationText({...recovered,exhausted_count:1}),/1 条已耗尽重试/);
  assert.equal(view.notificationText({enabled:false}),'未启用推送');
});

test('blocked signals explain the actual paper risk reason without claiming a win probability',()=>{
  const text=view.decisionText({symbol:'ETHUSDT',side:'LONG',reason:'signal',paper_risk_status:'paper_position_open',confidence_score:.62});
  assert.match(text,/已有模拟持仓/);
  assert.match(text,/方向分数 62.0%/);
  assert.doesNotMatch(text,/胜率/);
  assert.equal(view.settlementText({symbol:'OTHER',outcome:'win',net_usdt:4.25}),'暂无已结算模拟信号');
  assert.equal(view.paperText({}),'等待服务器模拟记录');
});

test('fresh prices alone do not imply that minute predictions are still running',()=>{
  const now=1794450060000;
  assert.match(view.monitorText({dataHealth:'fresh',lastPrediction:now-8000},now),/每分钟机会扫描中/);
  assert.match(view.monitorText({dataHealth:'fresh',lastPrediction:now-180000},now),/扫描等待或异常/);
  assert.match(view.monitorText({dataHealth:'fresh',lastPrediction:null},now),/扫描等待或异常/);
  assert.match(view.monitorText({dataHealth:'market_unavailable',lastPrediction:now-8000},now),/扫描等待或异常/);
});
