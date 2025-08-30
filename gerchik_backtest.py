#!/usr/bin/env python3
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yfinance as yf

plt.rcParams["figure.figsize"] = (10, 4)
pd.options.display.float_format = lambda x: f"{x:,.4f}"


# --- Parameters ---
PARAMS = {
    # Universe & dates
    "TICKERS": ["AAPL", "MSFT", "NVDA", "META", "AMZN", "TSLA", "AMD", "GOOGL", "NFLX", "AVGO"],
    "START": "2022-01-01",
    "END":   None,       # None => until today
    "INTERVAL": "1d",    # '1d' recommended for this MVP

    # Liquidity & volatility filters (daily)
    "MIN_PRICE": 5.0,
    "MIN_AVG_VOL20": 1_000_000,   # shares/day
    "ATR_PCT_MIN": 0.01,          # 1%
    "ATR_PCT_MAX": 0.08,          # 8%

    # ATR & consolidation
    "ATR_PERIOD": 14,
    "CONSOL_5D_MAX_MULT_ATR": 1.5,

    # Swing/levels detection
    "SWING_WINDOW": 3,          # N bars left/right to define swing
    "LEVEL_TOL_ATR": 0.25,      # points counted as same level if within 0.25*ATR
    "MIN_TOUCHES": 2,           # min touches to consider level "strong"
    "MAX_LEVELS_PER_SIDE": 5,   # per ticker: top N supports + top N resistances by touches

    # Signal proximity (distance to level)
    "READY_DIST_ATR": 0.5,      # entry only if distance < 0.5*ATR

    # Risk & exits
    "RISK_EUR": 10.0,           # fixed risk per trade (EUR)
    "ALLOW_FRACTIONAL": True,   # fractional shares sizing
    "STOP_ATR_MULT": 1.0,       # SL distance in ATR from level
    "RR_TARGET": 2.0,           # TP = Entry +/- RR * (Entry-Stop)

    # Execution assumptions
    "USE_NEXT_OPEN_FOR_ENTRY": True,  # enter next bar at open
    "SLTP_INTRADAY_CHECK": True,      # if high/low pierce SL/TP intraday -> fill that day

    # Output
    "OUT_DIR": "/workspace/out",
    "SEED": 42,
}

np.random.seed(PARAMS["SEED"])


# --- Helpers ---

def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Flatten MultiIndex columns, if present
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = ["_".join([str(p) for p in col if p is not None and str(p) != ""]) for col in df.columns]
    else:
        df = df.copy()

    # Normalize to lowercase without spaces/underscores for matching
    def _norm(s: str) -> str:
        return str(s).lower().replace(" ", "").replace("_", "")

    colmap = {c: _norm(c) for c in df.columns}

    # Helper to find first column matching any of the keys
    def _find(keys: List[str]) -> Optional[str]:
        for orig, norm in colmap.items():
            if norm in keys:
                return orig
        # try suffix patterns like AAPL_Close
        for orig in df.columns:
            low = _norm(orig)
            for k in keys:
                if low.endswith(k):
                    return orig
        return None

    open_col = _find(["open"]) 
    high_col = _find(["high"]) 
    low_col  = _find(["low"])  
    close_col = _find(["close"]) 
    adj_close_col = _find(["adjclose", "adjustedclose"]) 
    vol_col = _find(["volume"]) 

    # Build standardized view
    out = pd.DataFrame(index=df.index)
    if open_col is not None: out["Open"] = df[open_col]
    if high_col is not None: out["High"] = df[high_col]
    if low_col is not None: out["Low"] = df[low_col]
    if close_col is not None: out["Close"] = df[close_col]
    if adj_close_col is not None: out["Adj Close"] = df[adj_close_col]
    if vol_col is not None: out["Volume"] = df[vol_col]

    return out


def download_history(ticker: str, start: str, end: Optional[str], interval: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=False, progress=False, group_by="column")
    if df.empty:
        return df
    df = _normalize_ohlcv_columns(df)
    df.index = pd.to_datetime(df.index)
    return df


def compute_atr(df: pd.DataFrame, period: int) -> pd.DataFrame:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    out = df.copy()
    out["TR"] = tr
    out[f"ATR_{period}"] = atr
    return out


def rolling_avg(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n).mean()


def detect_swings(df: pd.DataFrame, window: int) -> Tuple[pd.Series, pd.Series]:
    # swing high: high is max over [t-window, t+window]
    # swing low:  low  is min over [t-window, t+window]
    highs = df["High"]
    lows = df["Low"]
    sw_hi = (highs == highs.rolling(window*2+1, center=True).max())
    sw_lo = (lows  == lows.rolling(window*2+1, center=True).min())
    return sw_hi.fillna(False), sw_lo.fillna(False)


def cluster_levels(prices: pd.Series, atr: pd.Series, is_swing_mask: pd.Series, level_tol_atr: float) -> List[Tuple[pd.Timestamp, float, int]]:
    """
    Cluster swing points into horizontal levels if they lie within 'level_tol_atr * ATR' of each other.
    Return list of tuples: (anchor_date, level_price, touches_count) sorted by touches desc.
    """
    level_points = []
    for dt, is_swing in is_swing_mask.items():
        if is_swing:
            level_points.append((dt, float(prices.loc[dt]), float(atr.loc[dt]) if not np.isnan(atr.loc[dt]) else None))
    # simple clustering by scanning
    clusters = []  # each cluster: list of (dt, price, atr)
    for dt, price, atr_val in level_points:
        if atr_val is None or atr_val <= 0:
            continue
        assigned = False
        for cl in clusters:
            # representative price = mean of cluster
            rep = np.mean([p for (_, p, _) in cl])
            tol = level_tol_atr * atr_val
            if abs(price - rep) <= tol:
                cl.append((dt, price, atr_val))
                assigned = True
                break
        if not assigned:
            clusters.append([(dt, price, atr_val)])
    # produce levels: anchor = earliest dt in cluster, price = median, touches = len
    levels = []
    for cl in clusters:
        cl_sorted = sorted(cl, key=lambda x: x[0])
        anchor = cl_sorted[0][0]
        price = float(np.median([p for (_, p, _) in cl_sorted]))
        touches = len(cl_sorted)
        levels.append((anchor, price, touches))
    # sort by touches desc, then by recency (anchor desc)
    levels.sort(key=lambda x: (x[2], x[0]), reverse=True)
    return levels


@dataclass
class Level:
    date: pd.Timestamp
    price: float
    touches: int
    side: str  # "resistance" or "support"


@dataclass
class Signal:
    ticker: str
    date: pd.Timestamp
    setup: str      # "breakout_long" | "breakout_short" | "falsebreak_long" | "falsebreak_short"
    level: float
    entry: float
    stop: float
    tp: float
    risk_per_share: float


@dataclass
class Trade:
    ticker: str
    setup: str
    level: float
    date_entry: pd.Timestamp
    entry: float
    stop: float
    tp: float
    shares: float
    date_exit: pd.Timestamp
    exit_price: float
    exit_reason: str  # "TP" | "SL" | "EOD"
    pnl_eur: float
    R: float


# --- Build universe, compute indicators, levels ---

def prepare_ticker(ticker: str, P: Dict) -> Tuple[pd.DataFrame, List[Level]]:
    df = download_history(ticker, P["START"], P["END"], P["INTERVAL"])
    if df.empty or len(df) < 260:
        return pd.DataFrame(), []

    # ensure clean OHLCV before indicator calculation
    df = _normalize_ohlcv_columns(df)
    # if any core columns missing, skip
    required_cols = ["Open", "High", "Low", "Close", "Volume"]
    if any(c not in df.columns for c in required_cols):
        return pd.DataFrame(), []
    df = compute_atr(df, P["ATR_PERIOD"])
    df["ATRpct"] = df[f"ATR_{P['ATR_PERIOD']}"] / df["Close"]
    df["AvgVol20"] = rolling_avg(df["Volume"], 20)
    # 5D range (close max - close min over last 5 bars)
    df["CloseMax5"] = df["Close"].rolling(5).max()
    df["CloseMin5"] = df["Close"].rolling(5).min()
    df["Range5"] = df["CloseMax5"] - df["CloseMin5"]

    # Swings
    sw_hi_mask, sw_lo_mask = detect_swings(df, P["SWING_WINDOW"])
    res_levels = cluster_levels(df["High"], df[f"ATR_{P['ATR_PERIOD']}"], sw_hi_mask, P["LEVEL_TOL_ATR"])
    sup_levels = cluster_levels(df["Low"],  df[f"ATR_{P['ATR_PERIOD']}"], sw_lo_mask, P["LEVEL_TOL_ATR"])

    # Keep strong ones
    res_levels = [Level(d, p, t, "resistance") for (d, p, t) in res_levels if t >= P["MIN_TOUCHES"]][:P["MAX_LEVELS_PER_SIDE"]]
    sup_levels = [Level(d, p, t, "support")    for (d, p, t) in sup_levels if t >= P["MIN_TOUCHES"]][:P["MAX_LEVELS_PER_SIDE"]]

    levels = res_levels + sup_levels
    return df, levels


# --- Generate signals per ticker/day ---

def generate_signals(df: pd.DataFrame, levels: List[Level], P: Dict) -> List[Signal]:
    if df.empty or not levels:
        return []

    atr_col = f"ATR_{P['ATR_PERIOD']}"
    sigs: List[Signal] = []
    for i in range(max(30, P["ATR_PERIOD"]) + 1, len(df) - 1):
        dt = df.index[i]
        row = df.iloc[i]
        atr = row[atr_col]
        if np.isnan(atr) or atr <= 0:
            continue

        price = row["Close"]
        range5 = row["Range5"]
        # consolidation near level
        is_consol_ok = (range5 / atr) <= P["CONSOL_5D_MAX_MULT_ATR"]

        # Check proximity to each level
        for L in levels:
            dist_atr = abs(price - L.price) / atr
            if dist_atr > P["READY_DIST_ATR"]:
                continue

            next_open = df.iloc[i + 1]["Open"] if P["USE_NEXT_OPEN_FOR_ENTRY"] else price
            # stop distance measured from level
            if L.side == "resistance":
                # Breakout long: close crosses above level and close > previous close
                prev_close = df.iloc[i - 1]["Close"]
                breakout_long = (row["Close"] > L.price) and (prev_close <= L.price) and is_consol_ok
                # False-break short: today's High > level, but Close < level and Close < prev_close (rejection)
                falsebreak_short = (row["High"] > L.price) and (row["Close"] < L.price) and (row["Close"] < prev_close) and is_consol_ok

                if breakout_long:
                    stop = L.price - P["STOP_ATR_MULT"] * atr
                    risk_per_share = max(next_open - stop, 1e-6)
                    tp = next_open + PARAMS["RR_TARGET"] * risk_per_share
                    sigs.append(Signal(ticker=df.attrs.get("ticker", ""), date=dt, setup="breakout_long",
                                       level=L.price, entry=float(next_open), stop=float(stop),
                                       tp=float(tp), risk_per_share=float(risk_per_share)))
                if falsebreak_short:
                    stop = L.price + P["STOP_ATR_MULT"] * atr
                    risk_per_share = max(stop - next_open, 1e-6)
                    tp = next_open - PARAMS["RR_TARGET"] * risk_per_share
                    sigs.append(Signal(ticker=df.attrs.get("ticker", ""), date=dt, setup="falsebreak_short",
                                       level=L.price, entry=float(next_open), stop=float(stop),
                                       tp=float(tp), risk_per_share=float(risk_per_share)))
            else:
                # support side
                prev_close = df.iloc[i - 1]["Close"]
                breakout_short = (row["Close"] < L.price) and (prev_close >= L.price) and is_consol_ok
                falsebreak_long = (row["Low"] < L.price) and (row["Close"] > L.price) and (row["Close"] > prev_close) and is_consol_ok

                if breakout_short:
                    stop = L.price + P["STOP_ATR_MULT"] * atr
                    risk_per_share = max(stop - next_open, 1e-6)
                    tp = next_open - PARAMS["RR_TARGET"] * risk_per_share
                    sigs.append(Signal(ticker=df.attrs.get("ticker", ""), date=dt, setup="breakout_short",
                                       level=L.price, entry=float(next_open), stop=float(stop),
                                       tp=float(tp), risk_per_share=float(risk_per_share)))
                if falsebreak_long:
                    stop = L.price - P["STOP_ATR_MULT"] * atr
                    risk_per_share = max(next_open - stop, 1e-6)
                    tp = next_open + PARAMS["RR_TARGET"] * risk_per_share
                    sigs.append(Signal(ticker=df.attrs.get("ticker", ""), date=dt, setup="falsebreak_long",
                                       level=L.price, entry=float(next_open), stop=float(stop),
                                       tp=float(tp), risk_per_share=float(risk_per_share)))
    return sigs


# --- Backtest execution ---

def backtest(df: pd.DataFrame, sigs: List[Signal], P: Dict) -> List[Trade]:
    trades: List[Trade] = []
    if df.empty or not sigs:
        return trades

    # prevent overlapping trades per ticker: once in a trade, ignore signals until exit
    in_trade = False

    for sig in sigs:
        if in_trade:
            continue
        # position sizing
        shares = PARAMS["RISK_EUR"] / max(sig.risk_per_share, 1e-6)
        if not P["ALLOW_FRACTIONAL"]:
            shares = math.floor(shares)
        if shares <= 0:
            continue

        # simulate bar-by-bar from day after signal date
        start_idx = df.index.get_loc(sig.date)
        entry_idx = start_idx + 1
        if entry_idx >= len(df):
            continue

        entry_date = df.index[entry_idx]
        entry_price = float(sig.entry)

        # Determine direction
        long_side = sig.setup in ("breakout_long", "falsebreak_long")
        stop = sig.stop
        tp = sig.tp

        in_trade = True
        exit_reason = None
        exit_price = None
        exit_date = None

        for j in range(entry_idx, len(df)):
            hi = float(df.iloc[j]["High"])
            lo = float(df.iloc[j]["Low"]) 
            dt = df.index[j]

            if long_side:
                hit_sl = lo <= stop
                hit_tp = hi >= tp
                if hit_sl and hit_tp:
                    # assume worst-case: SL first if both touched (conservative)
                    exit_reason = "SL"
                    exit_price = stop
                    exit_date = dt
                    break
                elif hit_tp:
                    exit_reason = "TP"
                    exit_price = tp
                    exit_date = dt
                    break
                elif hit_sl:
                    exit_reason = "SL"
                    exit_price = stop
                    exit_date = dt
                    break
            else:
                hit_sl = hi >= stop
                hit_tp = lo <= tp
                if hit_sl and hit_tp:
                    exit_reason = "SL"
                    exit_price = stop
                    exit_date = dt
                    break
                elif hit_tp:
                    exit_reason = "TP"
                    exit_price = tp
                    exit_date = dt
                    break
                elif hit_sl:
                    exit_reason = "SL"
                    exit_price = stop
                    exit_date = dt
                    break

        # if neither SL/TP hit until last bar, close at last close
        if exit_reason is None:
            exit_reason = "EOD"
            exit_date = df.index[-1]
            exit_price = float(df.iloc[-1]["Close"])

        pnl_per_share = (exit_price - entry_price) if long_side else (entry_price - exit_price)
        pnl_eur = pnl_per_share * shares
        R = pnl_eur / max(PARAMS["RISK_EUR"], 1e-9)

        trades.append(Trade(
            ticker=df.attrs.get("ticker", ""),
            setup=sig.setup,
            level=sig.level,
            date_entry=entry_date,
            entry=entry_price,
            stop=stop,
            tp=tp,
            shares=shares,
            date_exit=exit_date,
            exit_price=exit_price,
            exit_reason=exit_reason,
            pnl_eur=pnl_eur,
            R=R
        ))
        in_trade = False

    return trades


def equity_curve(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["Date", "Equity"])
    df = pd.DataFrame([{
        "Date": t.date_exit,
        "PnL": t.pnl_eur
    } for t in trades]).sort_values("Date")
    df["Equity"] = df["PnL"].cumsum()
    return df[["Date", "Equity"]]


def report(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame({"metric": ["trades", "winrate", "avg_R", "sum_PnL_EUR", "max_drawdown_EUR"],
                             "value": [0, 0, 0, 0, 0]})
    df = pd.DataFrame([t.__dict__ for t in trades])
    wins = (df["pnl_eur"] > 0).sum()
    trades_n = len(df)
    winrate = wins / trades_n if trades_n else 0.0
    avg_R = df["R"].mean() if trades_n else 0.0
    sum_pnl = df["pnl_eur"].sum()

    # compute max drawdown on equity
    eq = df.sort_values("date_exit")["pnl_eur"].cumsum().reset_index(drop=True)
    roll_max = eq.cummax()
    dd = eq - roll_max
    max_dd = dd.min() if not dd.empty else 0.0

    return pd.DataFrame({
        "metric": ["trades", "winrate", "avg_R", "sum_PnL_EUR", "max_drawdown_EUR"],
        "value": [trades_n, winrate, avg_R, sum_pnl, max_dd]
    })


def main() -> None:
    all_trades: List[Trade] = []
    universe_summary = []

    os.makedirs(PARAMS["OUT_DIR"], exist_ok=True)

    for ticker in PARAMS["TICKERS"]:
        print(f"Processing {ticker} ...")
        df, levels = prepare_ticker(ticker, PARAMS)
        if df.empty:
            print(f"  No data for {ticker}")
            continue
        df.attrs["ticker"] = ticker

        # summary checks
        last = df.iloc[-1]
        universe_summary.append({
            "ticker": ticker,
            "last_close": last["Close"],
            "avgvol20": last["AvgVol20"],
            "atr": last[f"ATR_{PARAMS['ATR_PERIOD']}"],
            "atr_pct": last["ATRpct"],
            "levels_found": len(levels)
        })

        sigs = generate_signals(df, levels, PARAMS)
        if not sigs:
            print(f"  No signals for {ticker}")
            continue

        tds = backtest(df, sigs, PARAMS)
        all_trades.extend(tds)

    universe_df = pd.DataFrame(universe_summary)
    if not universe_df.empty and "ticker" in universe_df.columns:
        universe_df = universe_df.sort_values("ticker")
    else:
        universe_df = pd.DataFrame(columns=[
            "ticker", "last_close", "avgvol20", "atr", "atr_pct", "levels_found"
        ])
    trades_df = pd.DataFrame([t.__dict__ for t in all_trades]) if all_trades else pd.DataFrame()
    eq_df = equity_curve(all_trades)
    rep_df = report(all_trades)

    print("\n=== Universe summary (head) ===")
    if not universe_df.empty:
        print(universe_df.head(20).to_string(index=False))
    else:
        print("(empty)")

    print("\n=== Trades (last 10) ===")
    if not trades_df.empty:
        print(trades_df.tail(10).to_string(index=False))
    else:
        print("(no trades)")

    print("\n=== Report ===")
    print(rep_df.to_string(index=False))

    # --- Save outputs ---
    u_path = os.path.join(PARAMS["OUT_DIR"], "universe_summary.csv")
    t_path = os.path.join(PARAMS["OUT_DIR"], "trades.csv")
    e_path = os.path.join(PARAMS["OUT_DIR"], "equity.csv")
    r_path = os.path.join(PARAMS["OUT_DIR"], "report.csv")

    universe_df.to_csv(u_path, index=False)
    trades_df.to_csv(t_path, index=False)
    eq_df.to_csv(e_path, index=False)
    rep_df.to_csv(r_path, index=False)

    print(f"\nSaved:\n  {u_path}\n  {t_path}\n  {e_path}\n  {r_path}")

    # --- Plot equity ---
    if not eq_df.empty:
        plt.figure()
        plt.plot(eq_df["Date"], eq_df["Equity"]) 
        plt.title("Equity Curve (EUR)")
        plt.xlabel("Date")
        plt.ylabel("Equity")
        plt.grid(True)
        fig_path = os.path.join(PARAMS["OUT_DIR"], "equity.png")
        plt.tight_layout()
        plt.savefig(fig_path)
        print(f"Saved plot: {fig_path}")
    else:
        print("No equity to plot (no trades).")


if __name__ == "__main__":
    main()