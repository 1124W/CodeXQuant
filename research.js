fetch("research/results.json")
  .then(response => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); })
  .then(report => {
    const selected = report.chosen;
    if (!selected) return;
    document.getElementById("trainRate").textContent = `${(selected.train.win_rate * 100).toFixed(1)}%`;
    document.getElementById("testRate").textContent = `${(selected.test.win_rate * 100).toFixed(1)}%`;
    document.getElementById("trainDetail").textContent = `${selected.train.signals} 次 · ${selected.train.net_units > 0 ? "+" : ""}${selected.train.net_units} 单位`;
    document.getElementById("testDetail").textContent = `${selected.test.signals} 次 · ${selected.test.net_units > 0 ? "+" : ""}${selected.test.net_units} 单位`;
    document.getElementById("breakEvenRate").textContent = `${(report.break_even_win_rate * 100).toFixed(1)}%`;
    document.getElementById("researchNote").textContent = `BTC + ETH · ${report.start_utc} 至 ${report.end_exclusive_utc}（结束日期不含）· 最近 30 天仅用于独立检验。该规则未通过上线标准，暂不发布交易信号。`;
  })
  .catch(error => { console.warn("Research report unavailable", error); document.getElementById("researchNote").textContent = "研究报告暂时无法加载；目前没有通过验证的交易信号。"; });
