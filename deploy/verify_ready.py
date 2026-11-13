"""Verify public live inputs and send one explicitly authorized deployment notice."""
from pathlib import Path
import json
import os
import sys
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from live.features import build_latest_features
from live.market import BinanceMarket
from live.predictor import Predictor
from live.notify import WeComNotifier

def main():
    with urlopen('http://127.0.0.1/api/live/status',timeout=5) as response:status=json.load(response)
    assert status['mode']=='research_alerts' and status['order_execution_enabled'] is False
    assert len(status['symbols'])==2 and all(not quote['stale'] for quote in status['symbols'].values())
    market=BinanceMarket()
    assert status['scan_interval_seconds']==60
    assert 'paper_performance' in status and 'recent_settlements' in status
    slot=int(time.time()*1000)//60000*60000
    spot,premium,funding=market.features_input(slot)
    predictor=Predictor(os.environ['CODEXQUANT_MODEL_DIR'])
    rows=build_latest_features(spot,premium,funding,slot)
    decisions=predictor.predict(rows,slot)
    print(json.dumps({'public_market_and_live_predictor_verified':True,'signal_boundary_ms':slot,
                     'model_version':predictor.manifest['model_version'],
                     'decisions':[{k:d[k] for k in ['symbol','side','p_up','reason']} for d in decisions]}),flush=True)
    notifier=WeComNotifier(os.environ.get('WECOM_WEBHOOK_URL',''),Path(os.environ['CODEXQUANT_STATE_DIR'])/'notifications.sqlite3')
    message='CodeXQuant 已更新并通过真实行情分钟预测验证。\nBTC/ETH 每分钟检查已收盘数据，条件满足才提醒；持有30分钟，85%净赔付。\n新增每笔模拟结算复盘（开平价格、胜负、收益）与每小时扫描/前向模拟汇总；整点价格播报保留。\n交易图表：http://38.76.182.201/demo/trades.html\n提高扫描频率不代表胜率提高；当前仅研究提醒，不自动下单。'
    report_path=ROOT/'research'/'minute-scan-results.json'
    if report_path.exists():
        report=json.loads(report_path.read_text(encoding='utf-8'))
        baseline=report['results']['30'];minute=report['results']['1']
        message+=('\n本地历史回测：'+report['plan']['start_utc']+' 至 '+report['plan']['end_exclusive_utc']+'（UTC，结束日不含），每笔5 USDT。'
                  f"\n半小时扫描：{baseline['signals']}笔，胜率{baseline['win_rate']:.2%}，净收益{baseline['net_units']:+.2f} USDT。"
                  f"\n分钟扫描：{minute['signals']}笔，胜率{minute['win_rate']:.2%}，净收益{minute['net_units']:+.2f} USDT，最大回撤{minute['max_drawdown_units']:.2f} USDT。"
                  '\n未达到58%，尚未证明稳定盈利；历史回测与群内前向模拟统计分开记录。')
    result=notifier.send_status(message,key='minute-scan-deployment-v1-'+predictor.manifest['model_version'])
    print(json.dumps({'deployment_notification':result,'notification_status':notifier.status()},ensure_ascii=False),flush=True)
    if not result.get('sent') and not result.get('duplicate'):raise RuntimeError('Deployment notification not acknowledged')

if __name__=='__main__':main()
