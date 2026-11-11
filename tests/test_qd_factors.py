"""Upstream parity and no-lookahead regression tests for event features."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from research.qd_factors import DEFAULT_UPSTREAM_DIR, _aggregate, build_features, load_registry, upstream_provenance


def minute_rows(count=1200):
    origin = 1_735_689_600_000
    rows = []
    previous = 100.0
    for index in range(count):
        close = 100 + index * 0.002 + np.sin(index / 17) + np.cos(index / 9) * .3
        rows.append((origin + index * 60_000, previous, max(previous, close) + .1,
                     min(previous, close) - .1, close))
        previous = close
    return rows


@unittest.skipUnless((DEFAULT_UPSTREAM_DIR / ".git").exists(), "Download the pinned QuantDinger checkout first")
class QuantDingerFeaturesTests(unittest.TestCase):
    def test_upstream_source_pin_and_actual_registry(self):
        provenance = upstream_provenance()
        self.assertEqual(provenance["commit"], "d1cf2681bc84fba2551973b4bcab011ad483e8ad")
        self.assertEqual(provenance["git_blob_sha1"], "4222d4709b61493767c56998928d3d92c2a04e4c")
        self.assertEqual(load_registry().get_factor("rsi").factor_id, "rsi")

    def test_tampered_factor_source_or_wrong_commit_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "backend_api_python/app/services/factors/registry.py"
            target.parent.mkdir(parents=True)
            target.write_text("raise RuntimeError('untrusted source')\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Git blob"):
                load_registry(directory)
        with patch("research.qd_factors.subprocess.run") as run:
            run.return_value.stdout = "0" * 40
            with self.assertRaisesRegex(ValueError, "expected"):
                load_registry()

    def test_scalar_upstream_parity_and_complete_bar_timing(self):
        rows = minute_rows()
        features = build_features(rows, 5)
        self.assertEqual(int(features.iloc[0].timestamp), rows[0][0] + 5 * 60_000)
        self.assertEqual(len(features), 240)
        self.assertTrue(features.attrs["scalar_parity"]["passed"])
        self.assertGreaterEqual(features.attrs["scalar_parity"]["factor_checks"], 18)
        registry = load_registry()
        bars = _aggregate(rows, 5)
        for length in (22, 41, 121, 239):
            self.assertAlmostEqual(features.iloc[length - 1].rsi2,
                                   registry.compute_factor("rsi", bars.iloc[:length], {"period": 2}), places=10)

    def test_future_mutation_and_partial_bar_cannot_change_past(self):
        rows = minute_rows()
        prefix = build_features(rows[:901], 5)
        changed = rows[:901] + [(r[0], r[1] * 2, r[2] * 2, r[3] * 2, r[4] * 2) for r in rows[901:]]
        full = build_features(changed, 5)
        pd.testing.assert_frame_equal(prefix.reset_index(drop=True), full.iloc[:len(prefix)].reset_index(drop=True))
        self.assertEqual(len(prefix), 180)

    def test_flat_rsi_matches_upstream_and_volume_is_not_fabricated(self):
        rows = [(r[0], 100, 100, 100, 100) for r in minute_rows(400)]
        features = build_features(rows, 5)
        self.assertEqual(float(features.iloc[-1].rsi2), 100.0)
        self.assertEqual(float(features.iloc[-1].rsi14), 100.0)
        self.assertNotIn("volume", features.columns)
        self.assertEqual(float(features.iloc[-1].z20), 0.0)

    def test_missing_duplicate_or_invalid_minutes_are_rejected(self):
        rows = minute_rows(400)
        for malformed in (rows[:20] + rows[21:], rows[:20] + [rows[19]] + rows[21:],
                          rows[:20] + [(rows[20][0], -1, 2, 1, 1.5)] + rows[21:]):
            with self.assertRaises(ValueError):
                build_features(malformed, 5)


if __name__ == "__main__":
    unittest.main()
