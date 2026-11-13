"""Exact-cent paper stake configuration and fixed 85% settlement arithmetic."""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import os

LEGACY_STAKE_CENTS = 500
PAYOUT_BPS = 8500


def amount_cents(value, *, minimum=None, maximum=None):
    """Reject nonfinite, boolean and fractional-cent amounts; never round input."""
    if isinstance(value,bool):
        raise ValueError('Amount must be finite with at most two decimal places')
    try:
        number=Decimal(str(value))
    except (InvalidOperation,ValueError,TypeError):
        raise ValueError('Amount must be finite with at most two decimal places') from None
    if not number.is_finite() or number.as_tuple().exponent < -2:
        raise ValueError('Amount must be finite with at most two decimal places')
    cents=int(number*100)
    if number*100 != cents or (minimum is not None and cents<minimum) or (maximum is not None and cents>maximum):
        raise ValueError('Amount is outside the supported range')
    return cents


def stake_cents(value):
    return amount_cents(value,minimum=500,maximum=25000)


def settlement_cents(stake, outcome, payout_bps=PAYOUT_BPS):
    if type(stake) is not int or not 500<=stake<=25000 or type(payout_bps) is not int or payout_bps!=PAYOUT_BPS:
        raise ValueError('Unsupported persisted stake or payout')
    if outcome=='win':return (stake*payout_bps+5000)//10000
    if outcome=='loss':return -stake
    if outcome=='refund':return 0
    raise ValueError('Unknown paper outcome')


@dataclass(frozen=True)
class PaperSettings:
    stake_cents: int = 3000
    daily_limit_cents: int = 15000
    initial_capital_cents: int = 100000

    def __post_init__(self):
        if (type(self.stake_cents) is not int or not 500<=self.stake_cents<=25000
                or type(self.daily_limit_cents) is not int or self.daily_limit_cents<=0
                or type(self.initial_capital_cents) is not int or self.initial_capital_cents!=100000):
            raise ValueError('Invalid paper settings; existing initial capital remains 1000 USDT')

    @classmethod
    def for_stake(cls, value, daily_limit=None):
        stake=stake_cents(value)
        daily=5*stake if daily_limit is None else amount_cents(daily_limit,minimum=1)
        return cls(stake,daily)

    @classmethod
    def from_env(cls, environ=None):
        source=os.environ if environ is None else environ
        return cls.for_stake(source.get('CODEXQUANT_PAPER_STAKE','30'),source.get('CODEXQUANT_PAPER_DAILY_LIMIT'))

    def public(self):
        return {'stake_usdt':self.stake_cents/100,'daily_loss_limit_usdt':self.daily_limit_cents/100,
                'initial_capital_usdt':self.initial_capital_cents/100,'payout':.85,
                'win_net_usdt':settlement_cents(self.stake_cents,'win')/100,
                'loss_net_usdt':-self.stake_cents/100,'fee':0.,
                'min_stake_usdt':5,'max_stake_usdt':250,'amount_scope':'new_forward_paper_admissions',
                'historical_amounts_preserved':True}
