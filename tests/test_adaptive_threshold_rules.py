"""Synthetic causal and contract tests; never consult research leaderboards."""
import math
import unittest

import numpy as np
import pandas as pd

from research.adaptive_threshold_rules import (
    BREAK_EVEN, MIN_DECISIVE, WILSON_Z, apply_rule, availability_mask,
    candidates, monday_floor,
)

MINUTE = 60_000
DAY = 1_440 * MINUTE
WEEK = 7 * DAY
MONDAY = int(pd.Timestamp('2025-01-06', tz='UTC').timestamp() * 1000)
HISTORY = MONDAY - 180 * DAY


def selected(days=90, regime='unrestricted'):
    return next(r for r in candidates() if r['model'] == 'logistic_halfhour'
                and r['lookback_days'] == days and r['regime'] == regime)


def row(stamp, symbol='BTCUSDT', p=.7, change=1., **changes):
    return {'signal_ms': stamp, 'symbol': symbol, 'p_logistic_halfhour': p,
            'entry_ms': stamp + MINUTE, 'settlement_ms': stamp + 31 * MINUTE,
            'entry': 100., 'exit': 100. + change,
            'model_fit_cutoff_ms': MONDAY - 90 * DAY,
            'train_latest_settlement_ms': MONDAY - 90 * DAY - MINUTE,
            'ret_60': -2 * math.sqrt(60), 'rv_60': 1., 'rv_ratio': 1., **changes}


def history(n=120, symbol='BTCUSDT', wins=None, p=.7, end=MONDAY - 2 * DAY):
    wins = n if wins is None else wins
    return [row(end - (n - i) * 30 * MINUTE, symbol, p, 1. if i < wins else -1.)
            for i in range(n)]


def run(rows, rule=None, start=MONDAY, stop=MONDAY + WEEK, history_start=HISTORY):
    return apply_rule(pd.DataFrame(rows), rule or selected(), history_start, start, stop)


def btc(diagnostics, week=MONDAY):
    return next(d for d in diagnostics if d['symbol'] == 'BTCUSDT' and d['week_start_ms'] == week)


class AdaptiveThresholdRuleTests(unittest.TestCase):
    def test_fixed_copied_catalog_and_semantic_mutations_rejected(self):
        catalog = candidates()
        self.assertEqual(len(catalog), 12)
        self.assertEqual(len({r['id'] for r in catalog}), 12)
        self.assertTrue(all(not r['score_is_calibrated_win_probability'] for r in catalog))
        catalog[0]['thresholds'][0] = .5
        self.assertEqual(candidates()[0]['thresholds'], [.56, .60, .64, .68])
        rows = history() + [row(MONDAY)]
        for mutation in ({'thresholds': [.5]}, {'min_decisive': 5}, {'lookback_days': 60},
                         {'regime': 'trend'}, {'threshold': .5}):
            with self.assertRaises(ValueError):
                run(rows, {**selected(), **mutation})
        with self.assertRaises(ValueError):
            run(rows, 'unknown')

    def test_highest_threshold_tie_and_corrected_wilson(self):
        direction, diagnostics = run(history() + [row(MONDAY), row(MONDAY + MINUTE, p=.67)])
        np.testing.assert_array_equal(direction[-2:], [1, 0])
        self.assertEqual(direction.dtype, np.int8)
        d = btc(diagnostics)
        self.assertEqual(d['threshold'], .68)
        self.assertEqual(d['n'], MIN_DECISIVE)
        self.assertEqual(d['wins'], MIN_DECISIVE)
        self.assertAlmostEqual(d['lower'], 120 / (120 + WILSON_Z ** 2))
        self.assertGreater(d['lower'], BREAK_EVEN)
        self.assertEqual(d['latest_settlement_ms'], MONDAY - 2 * DAY + MINUTE)
        self.assertEqual(d['distinct_fit_cutoffs_ms'], [MONDAY - 90 * DAY])

    def test_threshold_is_ranked_by_lower_bound_not_number_of_signals(self):
        low = history(120, wins=40, p=.58)
        high = history(120, wins=110, p=.70, end=MONDAY - 5 * DAY)
        direction, diagnostics = run(low + high + [row(MONDAY, p=.60), row(MONDAY + MINUTE)])
        np.testing.assert_array_equal(direction[-2:], [0, 1])
        d = btc(diagnostics)
        self.assertEqual(d['threshold'], .68)
        self.assertEqual(d['n'], 120)
        self.assertEqual(d['wins'], 110)
        self.assertEqual(d['threshold_diagnostics'][0]['n'], 240)

    def test_future_prices_labels_and_metadata_are_not_read_and_input_is_unchanged(self):
        rows = history() + [row(MONDAY), row(MONDAY + 4 * DAY)]
        frame = pd.DataFrame(rows)
        before = frame.copy(deep=True)
        expected, expected_diag = apply_rule(frame, selected(), HISTORY, MONDAY, MONDAY + WEEK)
        pd.testing.assert_frame_equal(frame, before)
        for name in ('entry', 'exit', 'entry_ms', 'settlement_ms', 'y', 'is_flat',
                     'model_fit_cutoff_ms', 'train_latest_settlement_ms'):
            frame[name] = frame[name].astype(object) if name in frame else None
            frame.loc[frame.signal_ms >= MONDAY, name] = 'unreadable future value'
        actual, actual_diag = apply_rule(frame, selected(), HISTORY, MONDAY, MONDAY + WEEK)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual_diag, expected_diag)

    def test_gate_is_frozen_within_week_and_adapts_only_next_monday(self):
        losses = [row(MONDAY + (i + 1) * 30 * MINUTE, change=-1.) for i in range(200)]
        rows = history() + losses + [row(MONDAY + 6 * DAY), row(MONDAY + WEEK)]
        direction, diagnostics = run(rows, stop=MONDAY + 2 * WEEK)
        np.testing.assert_array_equal(direction[-2:], [1, 0])
        self.assertEqual(btc(diagnostics)['threshold'], .68)
        self.assertIsNone(btc(diagnostics, MONDAY + WEEK)['threshold'])
        self.assertEqual(btc(diagnostics, MONDAY + WEEK)['no_trade_reason'], 'no_eligible_threshold')

    def test_pending_and_exactly_at_cutoff_settlements_cannot_supply_labels(self):
        # Sunday 23:30 settles Monday 00:01 and cannot complete the sample minimum.
        rows = history(119) + [row(MONDAY - 30 * MINUTE, exit='not settled'), row(MONDAY)]
        direction, diagnostics = run(rows)
        self.assertEqual(direction[-1], 0)
        self.assertEqual(btc(diagnostics)['n'], 119)
        # An actual settlement reported exactly at cutoff is excluded before prices.
        exact = row(MONDAY - 60 * MINUTE, settlement_ms=MONDAY, exit='not settled')
        direction, diagnostics = run(history(119) + [exact, row(MONDAY)])
        self.assertEqual(direction[-1], 0)
        self.assertEqual(btc(diagnostics)['n'], 119)
        direction, diagnostics = run(history(119) + [row(MONDAY - 60 * MINUTE), row(MONDAY)])
        self.assertEqual(direction[-1], 1)
        self.assertEqual(btc(diagnostics)['n'], 120)

    def test_calibration_uses_only_nonoverlapping_clock_grid(self):
        off_grid = [row(MONDAY - DAY + (i * 10 + 1) * MINUTE) for i in range(100)]
        direction, diagnostics = run(history(119) + off_grid + [row(MONDAY)])
        self.assertEqual(direction[-1], 0)
        self.assertEqual(btc(diagnostics)['n'], 119)
        # Minute-level entries may pass once the grid calibration supports a gate.
        direction, _ = run(history(120) + [row(MONDAY + MINUTE)])
        self.assertEqual(direction[-1], 1)

    def test_assets_have_independent_gates_and_directional_win_labels(self):
        btc_history = history()
        eth_history = history(symbol='ETHUSDT', p=.3, wins=0)
        rows = btc_history + eth_history + [row(MONDAY), row(MONDAY, symbol='ETHUSDT', p=.3)]
        direction, diagnostics = run(rows)
        np.testing.assert_array_equal(direction[-2:], [1, -1])
        self.assertTrue(all(d['wins'] == 120 for d in diagnostics))
        rows = history(wins=0) + eth_history + [row(MONDAY), row(MONDAY, symbol='ETHUSDT', p=.3)]
        direction, _ = run(rows)
        np.testing.assert_array_equal(direction[-2:], [0, -1])

    def test_flat_outcomes_refund_and_do_not_count_toward_minimum(self):
        flats = [row(MONDAY - DAY + i * 30 * MINUTE, change=0.) for i in range(20)]
        direction, diagnostics = run(history(119) + flats + [row(MONDAY)])
        self.assertEqual(direction[-1], 0)
        self.assertEqual(btc(diagnostics)['n'], 119)
        self.assertEqual(btc(diagnostics)['flat_count'], 20)
        direction, diagnostics = run(history(120) + flats + [row(MONDAY)])
        self.assertEqual(direction[-1], 1)
        self.assertEqual(btc(diagnostics)['win_rate'], 1.)

    def test_full_calendar_warmup_and_midweek_evaluation(self):
        rows = history() + [row(MONDAY), row(MONDAY + 3 * DAY)]
        direction, diagnostics = run(rows, history_start=MONDAY - 89 * DAY)
        self.assertFalse(direction.any())
        self.assertEqual(btc(diagnostics)['no_trade_reason'], 'warmup')
        direction, diagnostics = run(rows, start=MONDAY + 3 * DAY)
        np.testing.assert_array_equal(direction[-2:], [0, 1])
        self.assertEqual(btc(diagnostics)['week_start_ms'], MONDAY)
        self.assertEqual(monday_floor(MONDAY - MINUTE), MONDAY - WEEK)
        direction, diagnostics = run(rows, selected(days=180), history_start=HISTORY)
        self.assertEqual(direction[-1], 1)

    def test_regime_is_applied_to_both_calibration_and_current_signals(self):
        rows = history() + [row(MONDAY), row(MONDAY + MINUTE, ret_60=20.),
                            row(MONDAY + 2 * MINUTE, rv_ratio=1.26)]
        direction, _ = run(rows, selected(regime='exhaustion'))
        np.testing.assert_array_equal(direction[-3:], [1, 0, 0])
        rows[0]['ret_60'] = 20.
        direction, diagnostics = run(rows, selected(regime='exhaustion'))
        self.assertFalse(direction.any())
        self.assertEqual(btc(diagnostics)['n'], 119)

    def test_known_unavailable_calendar_masks_calibration_and_live_signals(self):
        unavailable = int(pd.Timestamp('2026-01-17', tz='UTC').timestamp() * 1000)
        next_available = int(pd.Timestamp('2026-01-20', tz='UTC').timestamp() * 1000)
        np.testing.assert_array_equal(availability_mask([
            unavailable - 32 * MINUTE, unavailable - 31 * MINUTE,
            unavailable, next_available - MINUTE, next_available,
        ]), [True, False, False, False, True])
        week = int(pd.Timestamp('2026-01-19', tz='UTC').timestamp() * 1000)
        rows = history(119, end=week - 5 * DAY)
        rows += [row(unavailable), row(week), row(next_available)]
        direction, diagnostics = run(rows, start=week, stop=week + WEEK, history_start=week - 180 * DAY)
        self.assertFalse(direction.any())
        self.assertEqual(btc(diagnostics, week)['n'], 119)
        rows.insert(0, row(week - 4 * DAY))
        direction, diagnostics = run(rows, start=week, stop=week + WEEK, history_start=week - 180 * DAY)
        np.testing.assert_array_equal(direction[-2:], [0, 1])
        self.assertEqual(btc(diagnostics, week)['n'], 120)

    def test_invalid_and_neutral_scores_fail_closed_and_order_is_preserved(self):
        rows = history() + [row(MONDAY + i * MINUTE, p=p) for i, p in enumerate([.5, np.nan, np.inf, -.1, 1.1, .7])]
        direction, _ = run(rows)
        np.testing.assert_array_equal(direction[-6:], [0, 0, 0, 0, 0, 1])
        frame = pd.DataFrame(rows)
        order = np.arange(len(frame))[::-1]
        reordered, _ = apply_rule(frame.iloc[order], selected(), HISTORY, MONDAY, MONDAY + WEEK)
        np.testing.assert_array_equal(reordered, direction[order])

    def test_invalid_past_prices_clock_or_training_provenance_are_rejected(self):
        for mutation in ({'exit': -1.}, {'settlement_ms': MONDAY - 3 * DAY},
                         {'model_fit_cutoff_ms': MONDAY},
                         {'train_latest_settlement_ms': MONDAY}):
            rows = history() + [row(MONDAY)]
            rows[0].update(mutation)
            with self.assertRaises(ValueError):
                run(rows)
        for malformed in ([row(MONDAY), row(MONDAY)], [row(MONDAY, symbol='SOLUSDT')],
                          [row(MONDAY + 1)]):
            with self.assertRaises(ValueError):
                run(malformed)


if __name__ == '__main__':
    unittest.main()
