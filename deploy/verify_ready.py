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
    slot=int(time.time()*1000)//1800000*1800000
    spot,premium,funding=market.features_input(slot)
    predictor=Predictor(os.environ['CODEXQUANT_MODEL_DIR'])
    rows=build_latest_features(spot,premium,funding,slot)
    decisions=predictor.predict(rows,slot)
    print(json.dumps({'public_market_and_live_predictor_verified':True,'signal_boundary_ms':slot,
                     'model_version':predictor.manifest['model_version'],
                     'decisions':[{k:d[k] for k in ['symbol','side','p_up','reason']} for d in decisions]}),flush=True)
    notifier=WeComNotifier(os.environ.get('WECOM_WEBHOOK_URL',''),Path(os.environ['CODEXQUANT_STATE_DIR'])/'notifications.sqlite3')
    result=notifier.send_status('CodeXQuant 已部署并通过真实行情预测验证。\nBTC/ETH 30分钟研究信号会即时提醒；北京时间每小时整点播报指数价格。\n交易图表：http://38.76.182.201/demo/trades.html\n当前仅研究提醒，不自动下单。',key='deployment-ready-'+predictor.manifest['model_version'])
    print(json.dumps({'deployment_notification':result,'notification_status':notifier.status()},ensure_ascii=False),flush=True)
    if not result.get('sent') and not result.get('duplicate'):raise RuntimeError('Deployment notification not acknowledged')

if __name__=='__main__':main()
