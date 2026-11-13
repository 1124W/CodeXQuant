"""Causal, cent-denominated bankroll simulation for fixed or fractional stakes.

No training, networking, notifications or orders. Drawdown is measured on
settled equity; an unsettled stake is reserved for admission but not marked to
market. A drawdown tolerance is a selection criterion, not a hindsight stop.
"""
from __future__ import annotations

from collections import defaultdict
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import math

import numpy as np
import pandas as pd

MINUTE_MS = 60_000
DAY_MS = 86_400_000
SYMBOLS = ('BTCUSDT', 'ETHUSDT')


def _cents(value, label):
    if isinstance(value, bool):
        raise ValueError(label+' must be positive exact cents')
    try:
        number = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(label+' must be positive exact cents') from error
    if not number.is_finite() or number <= 0 or number*100 != (number*100).to_integral_value():
        raise ValueError(label+' must be positive exact cents')
    return int(number*100)


def _policy(policy):
    if not isinstance(policy, dict) or not isinstance(policy.get('id'), str) or not policy['id']:
        raise ValueError('Require an identified sizing policy')
    if policy.get('kind') == 'fixed':
        return 'fixed', _cents(policy.get('amount'), 'Fixed stake'), None
    if policy.get('kind') == 'fraction':
        value = policy.get('fraction')
        if isinstance(value, bool):
            raise ValueError('Stake fraction must be within (0,1]')
        try:
            fraction = Fraction(str(value))
        except (ValueError, ZeroDivisionError) as error:
            raise ValueError('Stake fraction must be within (0,1]') from error
        if not 0 < fraction <= 1:
            raise ValueError('Stake fraction must be within (0,1]')
        return 'fraction', None, fraction
    raise ValueError('Unknown stake sizing policy')


def _fraction_cents(amount, fraction):
    return amount*fraction.numerator//fraction.denominator


def _dollars(cents):
    return cents/100


def _settled_statistics(ledger, initial_cents, start_ms):
    """Aggregate equal settlement timestamps before recording equity extrema."""
    grouped = defaultdict(list)
    for trade in ledger:
        grouped[trade['settlement_ms']].append(trade)
    balance = peak = minimum = initial_cents
    peak_time = start_ms
    maximum_dd = 0
    maximum_pct = 0.
    dd_peak = dd_trough = pct_peak = pct_trough = start_ms
    curve = [{'time_ms':start_ms, 'equity':_dollars(balance),
              'high_water':_dollars(peak), 'drawdown_usdt':0., 'drawdown_pct':0.}]
    for stamp, trades in sorted(grouped.items()):
        balance += sum(t['_net_cents'] for t in trades)
        if balance > peak:
            peak = balance
            peak_time = stamp
        minimum = min(minimum,balance)
        drawdown = peak-balance
        percent = drawdown/peak
        if drawdown > maximum_dd:
            maximum_dd = drawdown; dd_peak = peak_time; dd_trough = stamp
        if percent > maximum_pct:
            maximum_pct = percent; pct_peak = peak_time; pct_trough = stamp
        for trade in trades:
            trade['balance_after_settlement'] = _dollars(balance)
        curve.append({'time_ms':stamp, 'equity':_dollars(balance),
                      'high_water':_dollars(peak), 'drawdown_usdt':_dollars(drawdown),
                      'drawdown_pct':percent})
    streak = longest = 0
    for trade in ledger:
        if trade['outcome'] == 'loss':
            streak += 1; longest = max(longest,streak)
        else:
            streak = 0
    stakes = [trade['_stake_cents'] for trade in ledger]
    wins = sum(t['outcome']=='win' for t in ledger)
    losses = sum(t['outcome']=='loss' for t in ledger)
    return {'signals':len(ledger), 'wins':wins, 'losses':losses,
            'refunds':len(ledger)-wins-losses, 'decisive_trades':wins+losses,
            'win_rate':wins/(wins+losses) if wins+losses else None,
            'net_usdt':_dollars(balance-initial_cents), 'ending_balance':_dollars(balance),
            'max_drawdown_usdt':_dollars(maximum_dd), 'max_drawdown_pct':maximum_pct,
            'max_drawdown_initial_pct':maximum_dd/initial_cents,
            'max_drawdown_peak_ms':dd_peak, 'max_drawdown_trough_ms':dd_trough,
            'max_drawdown_pct_peak_ms':pct_peak, 'max_drawdown_pct_trough_ms':pct_trough,
            'minimum_balance':_dollars(minimum), 'longest_losing_streak':longest,
            'average_stake':sum(stakes)/len(stakes)/100 if stakes else 0.,
            'min_stake':_dollars(min(stakes)) if stakes else 0.,
            'max_stake':_dollars(max(stakes)) if stakes else 0.,
            'equity_curve':curve, 'drawdown_basis':'settled_equity',
            'losing_streak_refund_policy':'reset',
            'percentage_units':'fraction_0_to_1'}


def simulate_stakes(frame, direction, start_ms, end_ms, policy, adverse_bps=0,
                    initial_bankroll=10000):
    """Replay raw opportunities; never scale an already-admitted trade ledger.

    Fractions apply to available settled funds, after pending principal is
    reserved. The daily loss budget is five times the fixed stake, or five
    times the fraction of settled book equity at that entry day's FIRST raw
    opportunity. The fractional daily budget remains frozen for that UTC day.
    Only outcomes with expiry strictly earlier than the signal are available.
    Percentages in stats are ratios: 0.45 means 45%.
    """
    initial = _cents(initial_bankroll,'Initial bankroll')
    kind, fixed, fraction = _policy(policy)
    if (not isinstance(frame,pd.DataFrame) or not isinstance(start_ms,(int,np.integer))
            or not isinstance(end_ms,(int,np.integer)) or not start_ms<end_ms
            or not math.isfinite(adverse_bps) or not 0<=adverse_bps<10000):
        raise ValueError('Invalid input frame, interval, or adverse-price scenario')
    required = ('symbol','signal_ms','entry_ms','settlement_ms','entry','exit')
    if any(name not in frame for name in required):
        raise ValueError('Missing event observations')
    direction = np.asarray(direction)
    if direction.shape!=(len(frame),) or not np.isin(direction,[-1,0,1]).all():
        raise ValueError('Invalid directions')
    clocks = frame[['signal_ms','entry_ms','settlement_ms']].to_numpy(dtype=float)
    if (not np.isfinite(clocks).all() or np.any(clocks!=np.floor(clocks))
            or np.any(clocks%MINUTE_MS) or not frame.symbol.isin(SYMBOLS).all()
            or frame.duplicated(['symbol','signal_ms']).any()):
        raise ValueError('Require unique BTC/ETH minute observations')
    stamps, entries, expiries = (frame[name].to_numpy(dtype=np.int64)
                                 for name in ('signal_ms','entry_ms','settlement_ms'))
    prices = frame[['entry','exit']].to_numpy(dtype=float)
    if (np.any(entries!=stamps+MINUTE_MS) or np.any(expiries!=entries+30*MINUTE_MS)
            or not np.isfinite(prices).all() or np.any(prices<=0)):
        raise ValueError('Invalid event clocks or index prices')
    fit = latest = None
    if 'model_fit_cutoff_ms' in frame or 'train_latest_settlement_ms' in frame:
        if not {'model_fit_cutoff_ms','train_latest_settlement_ms'}.issubset(frame.columns):
            raise ValueError('Both training provenance timestamps are required')
        provenance = frame[['model_fit_cutoff_ms','train_latest_settlement_ms']].to_numpy(dtype=float)
        if not np.isfinite(provenance).all() or np.any(provenance!=np.floor(provenance)):
            raise ValueError('Invalid training provenance timestamps')
        fit = frame.model_fit_cutoff_ms.to_numpy(dtype=np.int64)
        latest = frame.train_latest_settlement_ms.to_numpy(dtype=np.int64)
        if np.any(latest>=fit) or np.any(fit>stamps):
            raise ValueError('Noncausal model training provenance')
    symbols = frame.symbol.to_numpy()
    indices = np.flatnonzero((direction!=0)&(stamps>=start_ms)&(expiries<end_ms))
    indices = indices[np.lexsort(((symbols[indices]=='ETHUSDT').astype(np.int8),stamps[indices]))]
    pending = []; accepted = []; balance = initial
    daily_losses = defaultdict(int); daily_budgets = {}
    skipped = {'skipped_overlap':0, 'skipped_daily_limit':0, 'skipped_insufficient':0}
    for index in indices:
        stamp, entry_ms = int(stamps[index]),int(entries[index])
        remains = []
        for trade in pending:
            if trade['settlement_ms']<stamp:
                balance += trade['_net_cents']
                if trade['outcome']=='loss':
                    daily_losses[trade['settlement_ms']//DAY_MS] += trade['_stake_cents']
            else:
                remains.append(trade)
        pending = remains
        day = entry_ms//DAY_MS
        if day not in daily_budgets:
            daily_budgets[day] = 5*fixed if kind=='fixed' else _fraction_cents(5*balance,fraction)
        if any(t['settlement_ms']>entry_ms for t in pending):
            skipped['skipped_overlap'] += 1; continue
        reserved = sum(t['_stake_cents'] for t in pending)
        available = balance-reserved
        stake = fixed if kind=='fixed' else _fraction_cents(max(0,available),fraction)
        if stake<1:
            skipped['skipped_insufficient'] += 1; continue
        if daily_losses[day]+reserved+stake>daily_budgets[day]:
            skipped['skipped_daily_limit'] += 1; continue
        if stake>available:
            skipped['skipped_insufficient'] += 1; continue
        side = int(direction[index]); opened = float(prices[index,0])*(1+side*adverse_bps/10000)
        closed = float(prices[index,1]); change = (closed-opened)*side
        outcome = 'win' if change>0 else 'loss' if change<0 else 'refund'
        net = (stake*85+50)//100 if change>0 else -stake if change<0 else 0
        trade = {'symbol':str(symbols[index]), 'signal_ms':stamp, 'entry_ms':entry_ms,
                 'settlement_ms':int(expiries[index]), 'entry':opened, 'exit':closed,
                 'direction':side, 'outcome':outcome, 'stake':_dollars(stake),
                 'net_units':_dollars(net), 'horizon_minutes':30,
                 'sizing_policy_id':policy['id'], 'adverse_bps':float(adverse_bps),
                 'equity_at_signal':_dollars(balance), 'available_at_signal':_dollars(available),
                 'daily_budget_usdt':_dollars(daily_budgets[day]),
                 '_stake_cents':stake, '_net_cents':net}
        if fit is not None:
            trade['model_fit_cutoff_ms'] = int(fit[index])
            trade['train_latest_settlement_ms'] = int(latest[index])
        if 'rule_id' in frame:
            trade['rule_id'] = str(frame.iloc[index]['rule_id'])
        pending.append(trade);accepted.append(trade)
    accepted.sort(key=lambda t:(t['settlement_ms'],t['symbol']))
    stats = _settled_statistics(accepted,initial,int(start_ms))
    stats.update(skipped)
    stats.update(raw_opportunities=len(indices), initial_bankroll=_dollars(initial),
                 policy=dict(policy), daily_budget_multiplier=5., payout=.85, fee=0.,
                 adverse_bps=float(adverse_bps), unsettled_principal_reserved=True)
    for trade in accepted:
        del trade['_stake_cents'];del trade['_net_cents']
    return accepted,stats
