"""Deterministic public opportunity replay; amounts use integer USDT cents.

This operates on every raw eligible strategy signal, not a previously admitted
ledger. It is suitable for an independent browser implementation and never
downloads, trains, sends notifications or places orders.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, InvalidOperation
import math

MINUTE = 60_000
DAY = 86_400_000
SYMBOLS = ('BTCUSDT', 'ETHUSDT')


def money_cents(value, label='Amount'):
    if isinstance(value, bool):
        raise ValueError(label+' must be a positive exact-cent amount')
    try:
        decimal=Decimal(str(value))
    except (InvalidOperation, ValueError):
        raise ValueError(label+' must be a positive exact-cent amount') from None
    if not decimal.is_finite() or decimal<=0 or decimal*100!=(decimal*100).to_integral_value():
        raise ValueError(label+' must be a positive exact-cent amount')
    result=int(decimal*100)
    if result>9_000_000_000_000:
        raise ValueError(label+' exceeds exact browser arithmetic')
    return result


def validate_opportunities(rows):
    previous=None; seen=set()
    for index,row in enumerate(rows,1):
        if (row.get('id')!=index or row.get('symbol') not in SYMBOLS
                or type(row.get('direction')) is not int or row['direction'] not in (-1,1)):
            raise ValueError('Invalid opportunity identity')
        stamps=[row.get(k) for k in ('signal_ms','entry_ms','settlement_ms')]
        if any(type(value) is not int or value<=0 or value%MINUTE for value in stamps):
            raise ValueError('Opportunities require exact minute timestamps')
        signal,entry,expiry=stamps
        if entry!=signal+MINUTE or expiry!=entry+30*MINUTE:
            raise ValueError('Opportunity violates the delayed 30-minute clock')
        key=(signal,SYMBOLS.index(row['symbol']))
        if key in seen or (previous is not None and key<=previous):
            raise ValueError('Opportunities must be unique and ordered with BTC priority')
        seen.add(key);previous=key
        values=[]
        for field in ('entry','exit'):
            exact=row.get(field+'_exact')
            if not isinstance(exact,str):raise ValueError('Original index precision is required')
            try:price=Decimal(exact)
            except InvalidOperation:raise ValueError('Invalid exact index price') from None
            if not price.is_finite() or price<=0 or float(price)!=row.get(field):
                raise ValueError('Index display and original precision differ')
            values.append(price)
        change=(values[1]-values[0])*row['direction']
        outcome='win' if change>0 else 'loss' if change<0 else 'refund'
        if row.get('outcome')!=outcome:raise ValueError('Opportunity outcome differs from index prices')
    return rows


def ledger_stats(trades, initial_capital, start_ms):
    initial=money_cents(initial_capital,'Initial capital')
    balance=peak=minimum=initial;dd=0;dd_pct=0.;wins=losses=refunds=streak=longest=0
    for row in sorted(trades,key=lambda t:(t['settlement_ms'],t['symbol'])):
        net=round(row['net_units']*100)
        balance+=net;peak=max(peak,balance);minimum=min(minimum,balance)
        dd=max(dd,peak-balance);dd_pct=max(dd_pct,(peak-balance)/peak)
        if row['outcome']=='win':wins+=1;streak=0
        elif row['outcome']=='loss':losses+=1;streak+=1;longest=max(longest,streak)
        else:refunds+=1;streak=0
    return {'signals':len(trades),'wins':wins,'losses':losses,'refunds':refunds,
            'decisive_trades':wins+losses,'win_rate':wins/(wins+losses) if wins+losses else None,
            'net_usdt':(balance-initial)/100,'ending_balance':balance/100,
            'initial_bankroll':initial/100,'max_drawdown_usdt':dd/100,
            'max_drawdown_pct':dd_pct,'max_drawdown_initial_pct':dd/initial,
            'minimum_balance':minimum/100,'longest_losing_streak':longest}


def simulate_opportunities(rows, start_ms, end_ms, *, stake=30, initial_capital=10000,
                           daily_risk_mode='stake_multiple', daily_risk_multiplier=5,
                           daily_risk_limit=25, adverse_bps=0):
    """BTC priority; only settlements strictly before the signal are observable."""
    validate_opportunities(rows)
    cents=money_cents(stake,'Stake');initial=money_cents(initial_capital,'Initial capital')
    if not 500<=cents<=25000:raise ValueError('Stake must be between 5 and 250 USDT')
    if type(start_ms) is not int or type(end_ms) is not int or not start_ms<end_ms:
        raise ValueError('Require an increasing UTC evaluation window')
    if daily_risk_mode=='stake_multiple':
        if type(daily_risk_multiplier) is not int or daily_risk_multiplier!=5:
            raise ValueError('The audited daily multiplier is five')
        daily_budget=5*cents
    elif daily_risk_mode=='fixed':daily_budget=money_cents(daily_risk_limit,'Daily risk limit')
    else:raise ValueError('Unknown daily risk mode')
    if isinstance(adverse_bps,bool) or not isinstance(adverse_bps,(int,float)) or not math.isfinite(adverse_bps) or not 0<=adverse_bps<=2:
        raise ValueError('Require an adverse entry scenario from 0 to 2 bp')
    pending=[];accepted=[];balance=initial;daily_losses=defaultdict(int)
    counters={'skipped_overlap':0,'skipped_daily_limit':0,'skipped_insufficient':0,'raw_opportunities':0}
    for row in rows:
        if row['signal_ms']<start_ms or row['settlement_ms']>=end_ms:continue
        counters['raw_opportunities']+=1
        remaining=[]
        for trade in pending:
            if trade['settlement_ms']<row['signal_ms']:
                balance+=trade['_net_cents']
                if trade['outcome']=='loss':daily_losses[trade['settlement_ms']//DAY]+=cents
            else:remaining.append(trade)
        pending=remaining
        if any(trade['settlement_ms']>row['entry_ms'] for trade in pending):
            counters['skipped_overlap']+=1;continue
        reserved=len(pending)*cents
        if daily_losses[row['entry_ms']//DAY]+reserved+cents>daily_budget:
            counters['skipped_daily_limit']+=1;continue
        if balance-reserved<cents:
            counters['skipped_insufficient']+=1;continue
        opened=row['entry']*(1+row['direction']*adverse_bps/10000)
        change=(row['exit']-opened)*row['direction']
        outcome='win' if change>0 else 'loss' if change<0 else 'refund'
        net=(cents*85+50)//100 if outcome=='win' else -cents if outcome=='loss' else 0
        trade={**row,'id':len(accepted)+1,'opportunity_id':row['id'],'entry':opened,
               'entry_exact':repr(opened),'outcome':outcome,'stake':cents/100,'net_units':net/100,
               'equity_at_signal':balance/100,'available_at_signal':(balance-reserved)/100,
               'daily_budget_usdt':daily_budget/100,'_net_cents':net}
        pending.append(trade);accepted.append(trade)
    settled=initial
    for trade in accepted:
        settled+=trade.pop('_net_cents');trade['balance_after_settlement']=settled/100
    stats=ledger_stats(accepted,initial_capital,start_ms)
    stats.update(counters,stake_usdt=cents/100,daily_budget_usdt=daily_budget/100,
                 payout=.85,fee=0,adverse_bps=float(adverse_bps),daily_risk_mode=daily_risk_mode)
    return accepted,stats
