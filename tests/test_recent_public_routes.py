"""New public data remains read-only; full research/runtime files stay private."""
import csv
import io
import unittest
from live.app import create_app


class RecentPublicRouteTests(unittest.TestCase):
    def test_public_default_amount_and_both_windows(self):
        client=create_app().test_client()
        for path in ('/demo/recent.html','/recent-view.js','/recent-view.css'):
            with client.get(path) as response:self.assertEqual(response.status_code,200)
        with client.get('/research/recent-public.json') as response:
            data=response.get_json()
        self.assertEqual(data['assumptions']['default_stake'],30)
        self.assertEqual(data['assumptions']['initial_capital'],10000)
        self.assertEqual(data['assumptions']['daily_risk_multiplier'],5)
        self.assertEqual(data['published']['recent_six_months']['stats']['signals'],338)
        self.assertEqual(data['published']['year_to_date']['stats']['signals'],443)
        with client.get('/research/recent-public-trades.csv') as response:
            rows=list(csv.DictReader(io.StringIO(response.data.decode('utf-8-sig'))))
        self.assertEqual(len(rows),781)

    def test_no_public_amount_mutation_or_private_source_access(self):
        client=create_app().test_client()
        for path in ('/demo/recent.html','/research/recent-public.json','/api/live/status'):
            for method in ('POST','PUT','DELETE'):
                self.assertEqual(client.open(path,method=method,json={'stake':250}).status_code,405)
        for path in ('/research/public_stake_engine.py','/live/stakes.py',
                     '/research/recent-window-2026-11-12/results.json','/.private/runtime.env'):
            self.assertEqual(client.get(path).status_code,404)
