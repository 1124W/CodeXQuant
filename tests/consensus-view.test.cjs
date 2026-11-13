const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const view=require('../consensus-view.js');
const recent=require('../recent-view.js');
const data=JSON.parse(fs.readFileSync('research/consensus-public.json','utf8'));

test('three-year default replay matches published outcomes and financial terms',()=>{
  view.validate(structuredClone(data));
  const result=view.simulate(data,'three_years',30);
  assert.deepEqual([result.stats.signals,result.stats.wins,result.stats.losses,result.stats.net_usdt,result.stats.max_drawdown_usdt],
    [1892,1063,829,2236.5,895.5]);
  assert.equal(result.trades[0].daily_budget_usdt,150);
});

test('copied audited replay engine stays byte-for-byte financially compatible at min max and fractional stakes',()=>{
  for(const amount of [5,5.01,30,249.99,250]){
    assert.deepEqual(view.simulate(data,'three_years',amount),recent.simulate(data,'three_years',amount));
    const expected=data.provenance.numeric_replay_checks.find(row=>row.stake===amount);
    const stats=view.simulate(data,'three_years',amount).stats;
    for(const [key,value] of Object.entries(expected))if(key!=='stake')assert.equal(stats[key],value);
  }
});

test('annual contributions carry the continuous balance and preserve the negative 2024 year',()=>{
  const trades=view.simulate(data,'three_years',30).trades;
  for(const row of [...data.yearly,...data.anniversary_years]){
    const part=view.partitionStats(trades,row);
    for(const key of ['signals','win_rate','net_usdt','opening_balance','ending_balance','max_drawdown_usdt'])assert.equal(part[key],row[key]);
  }
  const row=data.yearly.find(row=>row.year===2024);
  assert.equal(view.partitionStats(trades,row).net_usdt,-373.5);
});

test('score agreement chronology frozen outcomes and adverse evidence fail closed when forged',()=>{
  for(const mutate of [d=>d.opportunities[0].p_logistic_1m=.5,
    d=>d.opportunities[0].train_latest_settlement_ms=d.opportunities[0].model_fit_cutoff_ms,
    d=>d.strategy.forward_validated=true,
    d=>d.published.three_years.trades[0].net_units=500,
    d=>d.yearly.find(row=>row.year===2024).net_usdt=373.5,
    d=>d.stress.find(row=>row.adverse_bps===2).stats.net_usdt=447]){
    const altered=structuredClone(data);mutate(altered);assert.throws(()=>view.validate(altered));
  }
});

test('amount limits and download retain selected amount exact prices and individual model scores',()=>{
  for(const value of [4.99,250.01,NaN,Infinity,'30.001','3e1'])assert.throws(()=>view.parseStake(value));
  const trades=view.simulate(data,'three_years',5.01).trades.slice(0,2);
  const csv=view.csvForTrades(trades,'three_years',5.01);
  assert.match(csv,/p_logistic_1m/);assert.match(csv,/p_gbm_deeper_5m/);assert.match(csv,/5.01/);
  assert.ok(csv.includes(trades[0].entry_exact));
});

test('the new page mounts independently and retains access to original research',()=>{
  const html=fs.readFileSync('demo/consensus.html','utf8');
  assert.match(html,/id="consensusView"/);assert.doesNotMatch(html,/id="recentView"/);
  assert.match(html,/href="recent.html"/);assert.match(html,/2bp/);assert.match(html,/373.50U/);
  assert.match(html,/min="5" max="250" step="0.01"/);
});

test('live display isolates the new strategy cohort from losing old records and shows both model scores',()=>{
  const live=require('../trade-view.js');
  const paper={settled:7,wins:1,losses:6,refunds:0,win_rate:1/7,net_usdt:-25.75,
    current_strategy:{settled:0,wins:0,losses:0,refunds:0,win_rate:null,net_usdt:0,pending:0}};
  const text=live.currentStrategyText(paper);
  assert.match(text,/0 笔已结算/);assert.match(text,/暂无已决胜负/);assert.doesNotMatch(text,/14.29|-25.75/);
  assert.match(live.paperText(paper),/-25.75/);
  assert.match(live.modelScoresText({logistic_1m:.64,gbm_deeper_5m:.38}),/Logistic 看涨 64.0% \/ GBM 看跌 62.0%/);
  assert.doesNotMatch(live.modelScoresText({logistic_1m:.64,gbm_deeper_5m:.38}),/胜率/);
  const state=live.normalizeStatus({strategy:{mode:'consensus',rule_id:data.strategy.rule_id},paper_performance:paper});
  assert.match(live.strategyText(state.strategy),/双模型同向/);
  assert.equal(state.paper.current_strategy.settled,0);
});
