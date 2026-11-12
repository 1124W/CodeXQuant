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
