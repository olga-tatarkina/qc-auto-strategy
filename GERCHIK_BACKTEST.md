# Gerchik Backtest (CLI)

Quick script to run the Gerchik-style levels breakout/false-break backtest locally.

## Usage

1) Create virtualenv and install deps:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

2) Run the backtest:

```bash
python gerchik_backtest.py
```

Outputs are written to `/workspace/out`:
- `universe_summary.csv`
- `trades.csv`
- `equity.csv`
- `report.csv`
- `equity.png`

## Parameters
Adjust constants in `gerchik_backtest.py` inside the `PARAMS` dict (tickers, dates, risk, etc.).