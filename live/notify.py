"""Durable, redacted WeCom research alerts; never submits trading orders.

Acknowledged sends are deduplicated across restarts. WeCom provides no caller
idempotency key: retrying a request whose acknowledgement was lost can duplicate
a message, so ambiguous failures remain explicitly marked delivery_uncertain.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import time
from urllib import error, parse, request
import uuid

from .stakes import amount_cents,stake_cents,settlement_cents


CST = timezone(timedelta(hours=8))
SYMBOLS = ("BTCUSDT", "ETHUSDT")
CONSENSUS_RULE = 'pair_logistic_1m_gbm_deeper_5m_0.60_unrestricted'
CONSENSUS_MEMBERS = ('logistic_1m','gbm_deeper_5m')
REQUEST_TIMEOUT_SECONDS = 8
MAX_ATTEMPTS = 3
MAX_REQUESTS_PER_MINUTE = 20
RETRY_DELAYS_MS = (5_000, 20_000, 60_000)


def _timestamp(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != int(value) or value <= 0:
        raise ValueError("Expected a positive millisecond timestamp")
    return int(value)


def _price(value):
    if isinstance(value, bool):
        raise ValueError("Invalid index price")
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("Invalid index price") from None
    if not math.isfinite(result) or result <= 0:
        raise ValueError("Invalid index price")
    return result


def _time(ms):
    return datetime.fromtimestamp(_timestamp(ms) / 1000, CST).strftime("%Y-%m-%d %H:%M:%S") + " 北京时间"


def _safe_label(value, maximum=120):
    # Treat model IDs/status keys as data, never permit Markdown control content.
    return re.sub(r"[^\w .:/-]", "", str(value))[:maximum]


def _event_expiry(event_key,kind):
    try:
        if kind=="signal":
            return int(event_key.split(":")[2])+60_000
        if kind in ("price","research_report"):
            return int(event_key.split(":")[1])+120_000
    except (ValueError,IndexError):
        return 0  # malformed legacy queue entries are never replayed as fresh
    return None


def validate_webhook(webhook_url):
    """Only this service's robot endpoint is accepted; errors never echo secrets."""
    if webhook_url is None or webhook_url == "":
        return None
    if not isinstance(webhook_url, str):
        raise ValueError("Invalid WeCom webhook configuration")
    try:
        url = parse.urlsplit(webhook_url)
        query = parse.parse_qs(url.query, strict_parsing=True)
        valid = (url.scheme == "https" and url.hostname == "qyapi.weixin.qq.com"
                 and url.port in (None, 443) and url.username is None and url.password is None
                 and url.path == "/cgi-bin/webhook/send" and not url.fragment
                 and set(query) == {"key"} and len(query["key"]) == 1
                 and re.fullmatch(r"[A-Za-z0-9_-]{1,256}", query["key"][0]) is not None)
    except (ValueError, TypeError):
        valid = False
    if not valid:
        raise ValueError("Invalid WeCom webhook configuration")
    return webhook_url


def format_signal(signal):
    symbol, side = signal.get("symbol"), signal.get("side")
    if symbol not in SYMBOLS or side not in ("LONG", "SHORT"):
        raise ValueError("Signals require BTC/ETH and LONG/SHORT")
    signal_ms = _timestamp(signal["signal_ms"])
    entry_ms = _timestamp(signal["entry_due_ms"])
    expiry_ms = _timestamp(signal["expiry_due_ms"])
    if entry_ms != signal_ms + 60_000 or expiry_ms != entry_ms + 1_800_000:
        raise ValueError("Signals require a one-minute delay and 30-minute expiry")
    score = signal.get("confidence_score")
    if score is None and signal.get("p_up") is not None:
        p = float(signal["p_up"])
        score = max(p, 1-p)
    if isinstance(score, bool) or score is None or not math.isfinite(float(score)) or not .5 <= float(score) <= 1:
        raise ValueError("Invalid directional model score")
    direction = "看涨 / 做多" if side == "LONG" else "看跌 / 做空"
    stake=stake_cents(signal.get('stake_usdt',5))
    if signal.get('payout',.85)!=.85:raise ValueError('Only 85% paper payout is supported')
    daily=amount_cents(signal.get('daily_loss_limit_usdt',stake/100*5),minimum=1)
    text = [f"## {symbol[:-4]} 30分钟事件信号 · {direction}",
            f"信号时间：{_time(signal_ms)}",
            f"计划开仓：{_time(entry_ms)}",
            f"计划到期：{_time(expiry_ms)}",
            f"模型方向分数：{float(score):.2%}（未经胜率校准）",
            f"本笔模拟本金：**{stake/100:.2f} USDT**；净赔付85%；无额外手续费。",
            f"判赢净收益 +{settlement_cents(stake,'win')/100:.2f} USDT；判输 -{stake/100:.2f} USDT；平价退还本金。",
            f"当前每日亏损及待结算风险预算：{daily/100:.2f} USDT。"]
    if signal.get('rule_id')==CONSENSUS_RULE or 'model_scores' in signal:
        scores=signal.get('model_scores')
        if not isinstance(scores,dict) or set(scores)!=set(CONSENSUS_MEMBERS):
            raise ValueError('Consensus alerts require both model scores')
        directional=[]
        for name in CONSENSUS_MEMBERS:
            value=scores[name]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
                raise ValueError('Invalid consensus member score')
            sided=value if side=='LONG' else 1-value
            if sided<.60:raise ValueError('Both models must confirm the alerted direction at 0.60')
            directional.append(sided)
        if not math.isclose(float(score),min(directional),rel_tol=0,abs_tol=1e-12):
            raise ValueError('Consensus score must equal the weaker directional score')
        text.append(f"双模型同向确认：Logistic {directional[0]:.2%} / GBM {directional[1]:.2%}（各自方向分数，均≥60%，不是胜率）。")
    if signal.get("index_price") is not None:
        text.append(f"提醒时参考指数：{_price(signal['index_price']):,.2f} USDT")
    if signal.get("model_id"):
        text.append(f"模型：{_safe_label(signal['model_id'])}")
    text.append("研究信号；尚未通过稳定盈利验证。计划价格不代表实际成交，本程序不自动下单。")
    return "\n".join(text)


def format_price_report(prices, slot_ms):
    slot_ms = _timestamp(slot_ms)
    if slot_ms % 3_600_000:
        raise ValueError("Price reports require a full-hour slot")
    text = ["## BTC / ETH 指数价格播报", f"整点：{_time(slot_ms)}"]
    for symbol in SYMBOLS:
        row = prices.get(symbol)
        if not isinstance(row, dict):
            raise ValueError("Require timestamped BTC and ETH index prices")
        value = row.get("index_price", row.get("indexPrice", row.get("price")))
        stamp = row.get("price_time_ms", row.get("exchange_time_ms", row.get("time")))
        text.extend([f"**{symbol[:-4]}：{_price(value):,.2f} USDT**", f"行情时间：{_time(stamp)}"])
    text.append("币安指数价；每小时整点播报，信号独立即时提醒。")
    return "\n".join(text)


def format_settlement(result):
    """Describe an observed paper outcome, never imply an actual broker fill."""
    symbol=result.get("symbol")
    direction=result.get("direction")
    if symbol not in SYMBOLS or type(direction) is not int or direction not in (-1,1):
        raise ValueError("Settlement requires BTC/ETH and an explicit direction")
    signal_ms=_timestamp(result["signal_ms"])
    entry_ms=_timestamp(result["entry_ms"])
    expiry_ms=_timestamp(result["settlement_ms"])
    if entry_ms!=signal_ms+60_000 or expiry_ms!=entry_ms+1_800_000:
        raise ValueError("Settlement must match a delayed 30-minute event")
    entry,exit_price=_price(result["entry_price"]),_price(result["exit_price"])
    change=(exit_price-entry)*direction
    outcome="win" if change>0 else "loss" if change<0 else "refund"
    stake=stake_cents(result.get('stake_usdt',5))
    if result.get('payout',.85)!=.85:raise ValueError('Only 85% paper payout is supported')
    net=settlement_cents(stake,outcome)/100
    if result.get("outcome")!=outcome or result.get("net_usdt")!=net:
        raise ValueError("Settlement outcome must match the observed index prices")
    label={"win":"盈利","loss":"亏损","refund":"平价退款"}[outcome]
    side="做多" if direction==1 else "做空"
    return "\n".join([
        f"## {symbol[:-4]} 30分钟模拟复盘 · {side} · {label}"+("（延迟补报）" if result.get('delayed_report') else ""),
        *(["归属策略："+('Logistic + GBM 双模型同向确认' if result['rule_id']==CONSENSUS_RULE else '原单模型策略（保留原入场记录）')]
          if result.get('rule_id') else []),
        f"信号时间：{_time(signal_ms)}",
        f"模拟开仓：{_time(entry_ms)} · {entry:,.2f} USDT",
        f"模拟到期：{_time(expiry_ms)} · {exit_price:,.2f} USDT",
        f"本笔固定本金：{stake/100:.2f} USDT；净赔付：85%；额外手续费：0",
        f"本笔模拟盈亏：**{net:+.2f} USDT**",
        "价格来自币安指数分钟开盘价；这是前向模拟结算，不代表实际下单或成交。"
    ])


def format_research_report(report,slot_ms):
    slot_ms=_timestamp(slot_ms)
    if slot_ms%3_600_000:
        raise ValueError("Research reports require a full-hour slot")
    scan=report["scan_stats"];paper=report["paper_performance"]
    reasons={"model_score_below_0.56":"模型方向分数不足", "model_does_not_oppose_60m_move":"未满足反转方向",
             "60m_move_not_extreme":"60分钟波动不够极端", "short_volatility_expanding":"短期波动扩张",
             "paper_position_open":"已有模拟持仓", "paper_daily_risk_budget":"达到当日风险预算",
             "paper_capital_insufficient":"模拟资金不足","research_publication_hold":"更新校验期间暂停新模拟入场",
             "model_directions_disagree":"双模型方向不同","consensus_score_below_0.60":"双模型最小分数不足60%",
             "neutral_model_direction":"模型未给出明确方向"}
    counts={**scan.get("reasons",{}),**scan.get("risk_blocks",{})}
    skipped=[f"{reasons.get(key,_safe_label(key))} {int(value)}次" for key,value in sorted(counts.items(),key=lambda item:-item[1])[:4] if key!="signal"]
    rate=paper.get("win_rate")
    rate_text=f"{rate:.2%}" if rate is not None else "暂无已决样本"
    text=["## BTC / ETH 每小时研究播报",f"截至：{_time(slot_ms)}",
          f"过去1小时：完成 {int(scan['evaluated_slots'])} 轮扫描，候选 {int(scan['candidates'])} 个，模拟准入 {int(scan['admitted'])} 笔。",
          f"延迟跳过：{int(scan['late_slots'])} 轮；每分钟扫描，仅满足条件时提醒。"]
    if skipped:text.append("未提醒原因："+"；".join(skipped))
    if not scan['admitted']:text.append("本小时无新准入信号；未人为降低门槛补足信号。")
    if report.get("prediction_error"):text.append("扫描异常："+_safe_label(report['prediction_error']))
    stake=stake_cents(paper.get('stake_usdt',5))
    daily=amount_cents(paper.get('daily_loss_limit_usdt',stake/100*5),minimum=1)
    initial=amount_cents(paper.get('initial_capital_usdt',1000),minimum=1)
    text.extend([f"当前新信号默认本金：{stake/100:.2f} USDT；每日亏损及待结算风险预算：{daily/100:.2f} USDT。",
                 f"服务前向模拟初始本金：{initial/100:,.2f} USDT；判赢净收益 +{settlement_cents(stake,'win')/100:.2f} USDT，判输 -{stake/100:.2f} USDT。",
                 f"累计前向模拟：{int(paper['settled'])} 笔已结算，{int(paper['pending'])} 笔待结算；"
                 f"胜 {int(paper['wins'])} / 负 {int(paper['losses'])} / 平 {int(paper['refunds'])}。",
                 f"已决胜率（不含平价）：{rate_text}；累计模拟盈亏：**{float(paper['net_usdt']):+.2f} USDT**。",
                 "以上为本程序真实留存的前向模拟记录，不是历史回测胜率；85%净赔付，无额外手续费。",
                 "累计盈亏按每笔原始本金计算，保留旧5U交易；调整默认金额不会重算历史盈亏。"])
    cohort=paper.get('current_strategy')
    if cohort and cohort.get('rule_id')==CONSENSUS_RULE:
        rate=cohort['win_rate'];shown=f'{rate:.2%}' if rate is not None else '暂无已决样本'
        text.extend(['', '### 当前双模型策略的独立前向记录',
                     f"已结算 {int(cohort['settled'])} 笔，待结算 {int(cohort['pending'])} 笔；胜 {int(cohort['wins'])} / 负 {int(cohort['losses'])} / 平 {int(cohort['refunds'])}。",
                     f"当前策略胜率：{shown}；模拟盈亏：{float(cohort['net_usdt']):+.2f} USDT。",
                     '上方累计账户含旧策略；本段仅统计双模型启用后准入的记录，不能把旧策略业绩算给新策略。'])
    historical=report.get('historical_research')
    if historical:
        text.extend(format_historical_research(historical))
    return "\n".join(text)


def format_historical_research(historical):
    """An explicitly separate, frozen historical comparator, never live stats."""
    rate=float(historical['win_rate']); net=float(historical['net_usdt'])
    if not math.isfinite(rate) or not 0<=rate<=1 or not math.isfinite(net):
        raise ValueError('Invalid historical research statistics')
    stake=stake_cents(historical['stake_usdt'])
    capital=amount_cents(historical['initial_capital_usdt'],minimum=1)
    current=historical.get('is_live_strategy') is True
    title='当前策略三年历史对照' if current else ('三年历史研究对照（非当前线上模型）' if historical.get('scope')=='three_year' else '近期历史研究对照（非当前线上模型）')
    output=["", "### "+title,
            f"区间：{_safe_label(historical['start_utc'])} 至 {_safe_label(historical['end_exclusive_utc'])} UTC，结束时刻不含。",
            f"固定研究本金 {capital/100:,.2f} USDT，每笔 {stake/100:.2f} USDT；{int(historical['signals'])} 笔，历史胜率 {rate:.2%}，模拟净收益 {net:+.2f} USDT。",
            "该规则来自历史择优，尚未完成前瞻验证；历史收益不并入上方服务器前向统计。" if current else
            "该策略是看过历史后的研究候选，尚未完成前瞻验证；不并入上方服务器前向统计。",
            f"[研究明细与金额测算]({historical['url']})"]
    if historical.get('scope')=='three_year':
        yearly=next((r for r in historical.get('yearly',[]) if str(r.get('year',r.get('label')))=='2024'),None)
        stress=next((r for r in historical.get('stress',[]) if r.get('adverse_bps')==2),None)
        if yearly and stress:
            output.insert(-1,f"2024自然年 {float(yearly['net_usdt']):+.2f}U；2bp不利开仓 {float(stress['stats']['net_usdt']):+.2f}U，未证实稳定58%。")
    return output


def _public_url(value):
    if not isinstance(value,str):raise ValueError('Require a public research URL')
    url=parse.urlsplit(value)
    if (url.scheme not in ('https','http') or not url.hostname or url.username or url.password
            or url.query or url.fragment or any(char in value for char in '[]()\r\n')):
        raise ValueError('Invalid public research URL')
    return value


def format_strategy_update(report,url):
    """One explicit deployment explanation, built from the audited public report."""
    url=_public_url(url)
    if (report.get('schema_version')!=1 or report['strategy']['rule_id']!=CONSENSUS_RULE
            or report['strategy']['posthoc'] is not True or report['strategy']['forward_validated'] is not False):
        raise ValueError('Require the frozen consensus historical report')
    stats=report['stats'];period=report['period'];assumptions=report['assumptions']
    if (assumptions['default_stake']!=30 or assumptions['initial_capital']!=10000
            or assumptions['net_payout']!=.85 or assumptions['fee']!=0):
        raise ValueError('Unexpected strategy publication assumptions')
    for key in ('win_rate','net_usdt','max_drawdown_usdt'):
        if isinstance(stats[key],bool) or not math.isfinite(float(stats[key])):raise ValueError('Invalid strategy statistic')
    if not 0<=stats['win_rate']<=1 or type(stats['signals']) is not int or stats['signals']<=0:
        raise ValueError('Invalid strategy counts')
    yearly=next(r for r in report['yearly'] if str(r.get('year',r.get('label')))=='2024')
    stress=next(r for r in report['stress'] if r['adverse_bps']==2)['stats']
    text='\n'.join([
        '## BTC / ETH 双模型策略已启用 · 研究提醒',
        '规则：Logistic（1分钟训练采样）与 GBM（5分钟训练采样）方向一致，且各自方向分数至少60%。分数不等于胜率。',
        '每分钟扫描，满足条件才提醒；计划下一分钟模拟开仓，持有30分钟。过去180天训练，每30天更新两个模型。',
        '新模拟信号每笔30U；判赢+25.50U、判输-30U；净赔付85%，无额外手续费。每日亏损及待结算风险上限150U，两币共享最多1笔持仓。',
        '', '### 三年历史回测（不是新策略前向业绩）',
        f"{_safe_label(period['start_utc'])} 至 {_safe_label(period['end_exclusive_utc'])} UTC，结束时刻不含。",
        f"研究初始本金10,000U；{stats['signals']:,}笔，胜率{stats['win_rate']:.2%}，净收益{stats['net_usdt']:+,.2f}U，最大回撤{stats['max_drawdown_usdt']:,.2f}U。",
        f"2024自然年：{int(yearly['signals'])}笔、胜率{float(yearly['win_rate']):.2%}、净收益{float(yearly['net_usdt']):+,.2f}U。",
        f"开仓报价不利偏差0.02%（2bp）：胜率{float(stress['win_rate']):.2%}、净收益{float(stress['net_usdt']):+,.2f}U；不是新增手续费。",
        '三年未达到58%目标；近期约60%的表现未在三年稳定保持。规则来自历史择优，仍需独立前向验证。',
        '', '### 后续群内播报',
        '信号注明BTC/ETH、多/空、两模型分数及本笔金额；到期核实指数后复盘；每小时整点报价格与研究状态。',
        '旧策略/旧金额记录原样保留，新双模型单独统计；历史回测不计入服务器前向收益。网页自选金额仅影响浏览器研究测算。',
        '本程序只发研究提醒和模拟结算，不自动下单。',f'[三年明细与金额测算]({url})'])
    if len(text.encode('utf-8'))>4096:raise ValueError('Strategy update exceeds WeCom limit')
    return text


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class WeComNotifier:
    """Persistent queue with 20/minute rate limit and at most 3 attempts/event.

    Empty webhook disables delivery. Errors/status never expose the webhook.
    ``retry_pending`` is intended to be called by the service's existing loop;
    there is no blocking sleep or independently scheduled background thread.
    """

    def __init__(self, webhook_url, state_path, *, opener=None, clock=None):
        self._webhook = validate_webhook(webhook_url)
        self.enabled = self._webhook is not None
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self._opener = opener or request.build_opener(_NoRedirect())
        self._clock = clock or time.time
        with self._db() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS wecom_events (
                  event_key TEXT PRIMARY KEY, kind TEXT NOT NULL, content TEXT NOT NULL,
                  status TEXT NOT NULL, created_ms INTEGER NOT NULL, updated_ms INTEGER NOT NULL,
                  attempts INTEGER NOT NULL DEFAULT 0, next_retry_ms INTEGER NOT NULL DEFAULT 0,
                  lease_until_ms INTEGER NOT NULL DEFAULT 0, claim_token TEXT,
                  expires_at_ms INTEGER,
                  last_error TEXT, last_errcode INTEGER,
                  delivery_uncertain INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS wecom_attempts (
                  id INTEGER PRIMARY KEY AUTOINCREMENT, at_ms INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS wecom_attempt_time ON wecom_attempts(at_ms);
            """)
            db=connection
            db.execute("BEGIN IMMEDIATE")
            if "expires_at_ms" not in {row[1] for row in db.execute("PRAGMA table_info(wecom_events)")}:
                db.execute("ALTER TABLE wecom_events ADD COLUMN expires_at_ms INTEGER")
            old=db.execute("SELECT event_key,kind FROM wecom_events WHERE expires_at_ms IS NULL AND kind IN ('signal','price')").fetchall()
            for row in old:
                db.execute("UPDATE wecom_events SET expires_at_ms=? WHERE event_key=?",
                           (_event_expiry(row["event_key"],row["kind"]),row["event_key"]))
            db.commit()

    @contextmanager
    def _db(self):
        connection = sqlite3.connect(self.state_path, timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        try:
            yield connection
        finally:
            connection.close()

    def _now(self):
        return int(self._clock()*1000)

    def send_signal(self, signal):
        text = format_signal(signal)
        model = hashlib.sha256(str(signal.get("model_id", "default")).encode()).hexdigest()[:16]
        key = f"signal:{signal['symbol']}:{int(signal['signal_ms'])}:{model}"
        return self._send(key, "signal", text)

    def send_price_report(self, prices, slot_ms):
        return self._send(f"price:{int(slot_ms)}", "price", format_price_report(prices, slot_ms))

    def send_settlement(self,result):
        result={**result,'delayed_report':self._now()>result['settlement_ms']+120_000}
        return self._send(f"settlement:{result['symbol']}:{int(result['signal_ms'])}","settlement",format_settlement(result))

    def send_research_report(self,report,slot_ms):
        return self._send(f"research_report:{int(slot_ms)}","research_report",format_research_report(report,slot_ms))

    def send_strategy_update(self,report,key):
        if not isinstance(key,str) or not key.strip():raise ValueError('Strategy update requires a stable deduplication key')
        text=format_strategy_update(report,report.get('url','https://nightsleep.eu.org/demo/consensus.html'))
        identifier=hashlib.sha256(key.encode('utf-8')).hexdigest()
        return self._send('strategy_update:'+identifier,'strategy_update',text)

    def send_status(self, text, key=None):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Status content must be nonempty text")
        # Avoid ever persisting/publishing a webhook accidentally passed as status.
        text = re.sub(r"https://qyapi\.weixin\.qq\.com/\S+", "[机器人地址已隐藏]", text)
        identifier = hashlib.sha256(str(key or uuid.uuid4().hex).encode()).hexdigest()
        return self._send("status:"+identifier, "status", text)

    def _send(self, event_key, kind, text):
        if not self.enabled:
            return {"sent":False, "duplicate":False, "errcode":None, "state":"disabled"}
        if len(text.encode("utf-8")) > 4096:
            raise ValueError("WeCom Markdown content exceeds 4096 bytes")
        now, token = self._now(), uuid.uuid4().hex
        expiry=_event_expiry(event_key,kind)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR IGNORE INTO wecom_events(event_key,kind,content,status,created_ms,updated_ms,expires_at_ms) VALUES(?,?,?,'pending',?,?,?)",
                       (event_key,kind,text,now,now,expiry))
            row = db.execute("SELECT * FROM wecom_events WHERE event_key=?",(event_key,)).fetchone()
            if row["status"] == "sent":
                db.commit()
                return {"sent":False,"duplicate":True,"errcode":0,"state":"sent"}
            if row["status"] == "sending" and row["lease_until_ms"] > now:
                db.commit()
                return {"sent":False,"duplicate":True,"errcode":None,"state":"in_flight"}
            expiry=row["expires_at_ms"]
            if expiry is not None and now>=expiry:
                db.execute("""UPDATE wecom_events SET status='expired',updated_ms=?,lease_until_ms=0,
                    claim_token=NULL,last_error='expired_before_delivery' WHERE event_key=?""",(now,event_key))
                db.commit()
                return {"sent":False,"duplicate":False,"errcode":None,"state":"expired",
                        "error":"expired_before_delivery"}
            if row["attempts"] >= MAX_ATTEMPTS:
                uncertain=bool(row["delivery_uncertain"] or row["status"]=="sending")
                db.execute("UPDATE wecom_events SET status='exhausted',updated_ms=?,delivery_uncertain=? WHERE event_key=?",(now,int(uncertain),event_key))
                db.commit()
                return {"sent":False,"duplicate":False,"errcode":row["last_errcode"],"state":"exhausted",
                        "error":"retry_exhausted","delivery_uncertain":uncertain}
            if row["status"] == "pending" and row["next_retry_ms"] > now:
                db.commit()
                return {"sent":False,"duplicate":False,"errcode":row["last_errcode"],"state":"backoff"}
            recent = db.execute("SELECT COUNT(*),MIN(at_ms) FROM wecom_attempts WHERE at_ms>?",(now-60_000,)).fetchone()
            if recent[0] >= MAX_REQUESTS_PER_MINUTE:
                db.execute("UPDATE wecom_events SET next_retry_ms=?,updated_ms=? WHERE event_key=?",(recent[1]+60_001,now,event_key))
                db.commit()
                return {"sent":False,"duplicate":False,"errcode":None,"state":"rate_limited"}
            if row["status"] == "sending":
                db.execute("UPDATE wecom_events SET delivery_uncertain=1 WHERE event_key=?",(event_key,))
            db.execute("UPDATE wecom_events SET status='sending',attempts=attempts+1,claim_token=?,lease_until_ms=?,updated_ms=? WHERE event_key=?",
                       (token,now+30_000,now,event_key))
            db.execute("INSERT INTO wecom_attempts(at_ms) VALUES(?)",(now,))
            db.execute("DELETE FROM wecom_attempts WHERE at_ms<?",(now-86_400_000,))
            db.commit()
            # The initially queued content remains unchanged across all retries.
            content, attempts = row["content"], row["attempts"]+1
        # A queued claim must also remain timely immediately before the HTTP call.
        result = ({"sent":False,"errcode":None,"error":"expired_before_delivery"}
                  if expiry is not None and self._now()>=expiry else self._post(content))
        finished = self._now()
        with self._db() as db:
            state = ("sent" if result["sent"] else "expired" if result.get("error")=="expired_before_delivery"
                     else "exhausted" if attempts >= MAX_ATTEMPTS else "pending")
            db.execute("""UPDATE wecom_events SET status=?,updated_ms=?,lease_until_ms=0,
                       claim_token=NULL,next_retry_ms=?,last_error=?,last_errcode=?,
                       delivery_uncertain=MAX(delivery_uncertain,?) WHERE event_key=? AND claim_token=?""",
                       (state,finished,finished+RETRY_DELAYS_MS[attempts-1],result.get("error"),result.get("errcode"),
                        int(result.get("delivery_uncertain",False)),event_key,token))
        return {**result,"duplicate":False,"state":state}

    def _post(self, content):
        payload=json.dumps({"msgtype":"markdown","markdown":{"content":content}},ensure_ascii=False).encode("utf-8")
        outgoing=request.Request(self._webhook,data=payload,headers={"Content-Type":"application/json; charset=utf-8"},method="POST")
        try:
            with self._opener.open(outgoing,timeout=REQUEST_TIMEOUT_SECONDS) as response:
                status = getattr(response,"status",None) or response.getcode()
                if status != 200:
                    return {"sent":False,"errcode":None,"error":f"wecom_http_{int(status)}"}
                result=json.loads(response.read(16_385))
                code=result.get("errcode") if isinstance(result,dict) else None
                if type(code) is not int:
                    return {"sent":False,"errcode":None,"error":"invalid_wecom_response","delivery_uncertain":True}
                return {"sent":code==0,"errcode":code,**({} if code==0 else {"error":f"wecom_errcode_{code}"})}
        except error.HTTPError as exc:
            return {"sent":False,"errcode":None,"error":f"wecom_http_{int(exc.code)}"}
        except (error.URLError,OSError,TimeoutError):
            return {"sent":False,"errcode":None,"error":"delivery_uncertain","delivery_uncertain":True}
        except (ValueError,TypeError,UnicodeError):
            return {"sent":False,"errcode":None,"error":"invalid_wecom_response","delivery_uncertain":True}

    def retry_pending(self, limit=5):
        if not self.enabled:
            return []
        if isinstance(limit,bool) or not isinstance(limit,int) or not 1 <= limit <= 20:
            raise ValueError("Retry limit must be between 1 and 20")
        now=self._now()
        with self._db() as db:
            db.execute("""UPDATE wecom_events SET status='expired',updated_ms=?,lease_until_ms=0,
                claim_token=NULL,last_error='expired_before_delivery' WHERE status NOT IN ('sent','expired')
                AND expires_at_ms IS NOT NULL AND expires_at_ms<=?
                AND (status!='sending' OR lease_until_ms<=?)""",(now,now,now))
            db.execute("""UPDATE wecom_events SET status='exhausted',delivery_uncertain=1,
                updated_ms=?,last_error='delivery_uncertain' WHERE status='sending'
                AND lease_until_ms<=? AND attempts>=?""",(now,now,MAX_ATTEMPTS))
            rows=db.execute("""SELECT event_key,kind,content FROM wecom_events
                WHERE attempts<? AND ((status='pending' AND next_retry_ms<=?) OR
                  (status='sending' AND lease_until_ms<=?)) ORDER BY created_ms,event_key LIMIT ?""",
                            (MAX_ATTEMPTS,now,now,limit)).fetchall()
        return [self._send(row["event_key"],row["kind"],row["content"]) for row in rows]

    def status(self):
        with self._db() as db:
            counts={row[0]:row[1] for row in db.execute("SELECT status,COUNT(*) FROM wecom_events GROUP BY status")}
            sent=db.execute("SELECT MAX(updated_ms) FROM wecom_events WHERE status='sent'").fetchone()[0]
            failed=db.execute("SELECT last_error,updated_ms FROM wecom_events WHERE last_error IS NOT NULL ORDER BY updated_ms DESC LIMIT 1").fetchone()
            uncertain=db.execute("SELECT COUNT(*) FROM wecom_events WHERE delivery_uncertain=1").fetchone()[0]
            kinds=db.execute("SELECT kind,COUNT(*),MAX(updated_ms) FROM wecom_events WHERE status='sent' GROUP BY kind").fetchall()
        return {"enabled":self.enabled,"sent_count":counts.get("sent",0),
                "pending_count":counts.get("pending",0)+counts.get("sending",0),
                "exhausted_count":counts.get("exhausted",0),"last_sent_at":sent,
                "expired_count":counts.get("expired",0),
                "last_error":failed[0] if failed else None,"last_error_at":failed[1] if failed else None,
                "sent_by_kind":{row[0]:row[1] for row in kinds},"last_sent_by_kind":{row[0]:row[2] for row in kinds},
                "delivery_uncertain_count":uncertain,"delivery_guarantee":"acknowledged_deduplication; ambiguous failures may duplicate on retry"}
