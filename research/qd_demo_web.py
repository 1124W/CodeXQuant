"""Read-only Flask blueprint for a QuantDinger event research report.

The blueprint can be mounted in an existing Flask application. The isolated
demo factory starts no database, workers, broker connectors or order endpoints.
"""

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def validate_report(report):
    if not isinstance(report, dict) or report.get("engine") not in ("quantdinger-event-demo-v1", "quantdinger-workbuddy-v1"):
        raise ValueError("Unsupported event research report")
    config, sample, ledger = report["config"], report["test"], report["trade_ledger"]
    if report["engine"] == "quantdinger-workbuddy-v1" and (
            set(report["symbols"]) != {"BTCUSDT", "ETHUSDT"}
            or config.get("horizon_minutes") != 30 or config["payout"] != .85):
        raise ValueError("This study permits only BTC/ETH 30-minute events at 85% payout")
    if not isinstance(ledger, list):
        raise ValueError("Report ledger must be a list")
    wins = losses = refunds = 0
    net = []
    start = report["windows"]["test"]["start_utc"]
    end = report["windows"]["test"]["end_exclusive_utc"]
    from .event_engine import utc_ms
    from datetime import date
    lower, upper = utc_ms(date.fromisoformat(start)), utc_ms(date.fromisoformat(end))
    previous_settlement = -1
    for trade in ledger:
        outcome = trade["outcome"]
        if outcome not in ("win", "loss", "refund"):
            raise ValueError("Unknown event outcome")
        if trade["rule_id"] != report["selected_rule"]["id"] or trade["symbol"] not in report["symbols"]:
            raise ValueError("Report rule or symbol disagrees with the ledger")
        if (not lower <= trade["signal_ms"] < trade["entry_ms"] < trade["settlement_ms"] < upper
                or trade["settlement_ms"] < previous_settlement
                or trade["entry_ms"] != trade["signal_ms"] + config["latency_minutes"] * 60000
                or trade["settlement_ms"] != trade["entry_ms"] + report["selected_rule"]["horizon_minutes"] * 60000):
            raise ValueError("Ledger chronology disagrees with event execution")
        previous_settlement = trade["settlement_ms"]
        change = (trade["exit"] - trade["entry"]) * trade["direction"]
        expected_outcome = "win" if change > 0 else "loss" if change < 0 else "refund"
        expected_net = config["stake"] * ((config["payout"] if outcome == "win" else -1 if outcome == "loss" else 0) - config["fee"])
        if (outcome != expected_outcome or not math.isclose(trade["net_units"], expected_net, abs_tol=1e-7)
                or not math.isclose(trade["stake"], config["stake"], abs_tol=1e-9)):
            raise ValueError("Ledger settlement or modeled cost disagrees with its configuration")
        wins += outcome == "win"
        losses += outcome == "loss"
        refunds += outcome == "refund"
        net.append(trade["net_units"])
    decisive = wins + losses
    rate = wins / decisive if decisive else None
    if (sample["signals"] != len(ledger) or sample["decisive_trades"] != decisive
            or (sample["wins"], sample["losses"], sample["refunds"]) != (wins, losses, refunds)
            or not math.isclose(sample["net_units"], math.fsum(net), abs_tol=1e-6)
            or (rate is None) != (sample["win_rate"] is None)
            or rate is not None and not math.isclose(rate, sample["win_rate"], abs_tol=1e-12)):
        raise ValueError("Summary disagrees with the authoritative trade ledger")
    numerical_pass = decisive >= config["min_test_trades"] and rate is not None and rate >= config["target_win_rate"] and math.fsum(net) > 0
    expected_pass = bool(report["selection_eligible"] and numerical_pass)
    if report["demo_pass"] != expected_pass or report["live_approved"] is not False:
        raise ValueError("Report qualification flag is inconsistent")
    return report


def create_blueprint(report_path, trades_path):
    from flask import Blueprint, jsonify, make_response
    blueprint = Blueprint("event_contract_research", __name__, url_prefix="/api/event-contracts")
    report_path, trades_path = Path(report_path), Path(trades_path)

    def read():
        return validate_report(json.loads(report_path.read_text(encoding="utf-8")))

    @blueprint.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @blueprint.get("/health")
    def health():
        try:
            report = read()
        except (OSError, ValueError, KeyError, TypeError) as error:
            return jsonify(code=0, msg=f"Report unavailable: {error}", data=None), 503
        return jsonify(code=1, msg="success", data={"mode": "research_only", "run_id": report["provenance"]["run_id"],
            "order_execution_enabled": False, "full_platform_services_started": False})

    @blueprint.get("/report")
    def report():
        try:
            payload = read()
        except (OSError, ValueError, KeyError, TypeError) as error:
            return jsonify(code=0, msg=f"Report unavailable: {error}", data=None), 503
        return jsonify(code=1, msg="success", data=payload)

    @blueprint.get("/report.json")
    def report_download():
        try:
            payload = read()
        except (OSError, ValueError, KeyError, TypeError) as error:
            return jsonify(code=0, msg=f"Report unavailable: {error}", data=None), 503
        response = make_response(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        response.headers["Content-Disposition"] = 'attachment; filename="quantdinger-event-report.json"'
        return response

    @blueprint.get("/trades.csv")
    def trades_download():
        # Generate from the canonical JSON rather than trusting a stale CSV pair.
        try:
            payload = read()
            if payload["engine"] == "quantdinger-workbuddy-v1":
                from .workbuddy_run import csv_text
                content = csv_text(payload)
            else:
                from .qd_demo import csv_payload
                content = csv_payload(payload)
        except (OSError, ValueError, KeyError, TypeError) as error:
            return jsonify(code=0, msg=f"Report unavailable: {error}", data=None), 503
        response = make_response(content)
        response.headers["Content-Type"] = "text/csv; charset=utf-8"
        response.headers["Content-Disposition"] = 'attachment; filename="quantdinger-event-trades.csv"'
        return response

    return blueprint


def create_app(report_path, trades_path):
    from flask import Flask, abort, send_file
    app = Flask("quantdinger_event_research", static_folder=None)
    app.json.ensure_ascii = False
    app.register_blueprint(create_blueprint(report_path, trades_path))

    @app.get("/")
    @app.get("/index.html")
    def index():
        response = send_file(ROOT / "index.html")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/demo/quantdinger.html")
    def demo():
        response = send_file(ROOT / "demo" / "quantdinger.html")
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/<filename>")
    def asset(filename):
        if filename not in {"app.js", "research.js", "styles.css", "workbuddy.js"}:
            abort(404)
        return send_file(ROOT / filename)

    @app.get("/research/<filename>")
    def public_report(filename):
        if filename not in {"workbuddy-results.json", "workbuddy-trades.csv", "event-results.json", "event-trades.csv", "results.json"}:
            abort(404)
        return send_file(ROOT / "research" / filename)

    return app
