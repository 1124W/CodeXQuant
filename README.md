# CodeXQuant

**A transparent quantitative trading research project.**

CodeXQuant is a personal research project exploring rule-based and AI-assisted systematic trading across multiple markets. The goal: build a transparent, fully backtestable pipeline — from market data ingestion to signal generation, risk gating, and performance analytics.

> Status: **Research prototype** — no strategy is currently approved for live use.

## Live 30m research dashboard

The browser prototype is a dependency-free static site. It includes:

- live BTC/USDT and ETH/USDT quotes from Binance public market streams;
- recent 30-minute candles with EMA overlays;
- a published research result with separate training and held-out periods;
- responsive desktop and mobile layouts, with clearly labeled demo charts when Binance is unavailable.

The research script in `research/run.py` downloads official Binance public ZIP files, verifies their SHA-256 checksums, and tests a fixed family of contrarian rules. It selects a rule using the first 60 days, then evaluates it once on the latest 30 days. The machine-readable report is `research/results.json`.

Open `index.html` directly, or serve the repository locally:

```bash
python3 -m http.server 4173
```

Then visit `http://localhost:4173`. No API key is required. Research assumes +0.80 for a win and −1.00 for a loss; actual contract terms and execution constraints may differ.

---

## Motivation

Retail quant trading usually fails for three reasons: hidden costs, overfitting, and unreliable statistics. CodeXQuant is built around three principles learned the hard way:

1. **Strict settlement** — signals are judged against real future prices (T+10m / T+30m), not lenient rules. Flat means flat, no "small-move-wins" exemptions.
2. **Cost-first math** — every strategy must clear its breakeven win rate given real spreads, commissions and payout multipliers before it deserves live simulation.
3. **Honest statistics** — confidence intervals, multiple-comparison corrections, and out-of-sample validation on every claimed edge.

## Planned Modules

| Module | Description | Status |
|---|---|---|
| `data` | Market data ingestion (Binance public klines via ccxt, MT5 symbols) | Planned |
| `regime` | Five-state market regime detection (trend / range / shock / breakout / transition) via weighted multi-factor voting | Planned |
| `signal` | Rule-based scoring engine: momentum, EMA structure, RSI/MACD filters, odds thresholds | Planned |
| `llm` | Optional LLM layer (DeepSeek API) for trade narratives only — never for direction decisions | Planned |
| `backtest` | Strict binary-option P&L model for event contracts (payout multiplier based) | Planned |
| `risk` | Spread gate, daily loss cap, consecutive-loss stop, session filter, cooldown | Planned |
| `dashboard` | Offline single-file HTML analytics: win-rate slicing, equity curves, calibration | Planned |
| `mt5` | Expert Advisors for XAUUSD scalping (MQL5) with diagnostic overlays | Planned |

## Target Markets

- **Crypto event contracts** (Binance) — binary options with payout multipliers, no API order placement (alert-only)
- **XAUUSD / forex CFDs** (MetaTrader 5) — M1 scalping EAs with full API automation

## Roadmap

- [ ] Repository structure and project skeleton
- [ ] Data ingestion module with quality gates
- [ ] Market regime classifier + validation suite
- [ ] Signal engine with strict settlement backtest
- [ ] Risk management framework
- [ ] Analytics dashboard
- [ ] Paper-trading phase (≥100 samples before any judgment)

## Disclaimers

- This project is for **research and education only**. Nothing here is financial advice.
- No live trading is performed by this codebase. Event-contract execution remains manual by design.
- Past (or backtested) performance does not guarantee anything about the future.

## License

TBD
