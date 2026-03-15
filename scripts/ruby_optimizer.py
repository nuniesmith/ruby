"""
ruby_optimizer.py
=================
Walk-forward optimization for Ruby v5 Pine Script constants.

USAGE
-----
1. Export 1-minute OHLCV CSV from TradingView:
   - Open chart → Export chart data → CSV
   - Or use ccxt / tvdatafeed / yfinance for data

2. pip install pandas numpy optuna ta colorama tqdm

3. Run:
   python ruby_optimizer.py --csv BTCUSD_1.csv --symbol BTCUSD
   python ruby_optimizer.py --csv ES1_1.csv --symbol ES1 --commission 1.25 --slippage 0.5

4. Output — a Pine Script constants block to paste into Ruby v5 section 0.

FLAGS
-----
--csv         Path to OHLCV CSV (columns: time, open, high, low, close, volume)
--symbol      Symbol name (for output header)
--trials      Optuna trials per fold (default 150)
--folds       Walk-forward folds (default 5)
--train_pct   Training window as fraction of fold (default 0.7)
--commission  Per-side commission in price units (default 0.0)
--slippage    Slippage per fill in price units (default 0.0)
--min_trades  Minimum trades for a fold to count (default 10)
"""

import argparse
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import ta
except ImportError:
    print("pip install ta")
    sys.exit(1)

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("pip install optuna")
    sys.exit(1)

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    HAS_COLOR = True
except ImportError:
    HAS_COLOR = False
    class Fore:
        GREEN = RED = YELLOW = CYAN = WHITE = RESET = ""
    class Style:
        BRIGHT = RESET_ALL = ""

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_csv(path: str) -> pd.DataFrame:
    """
    Load a TradingView-style OHLCV CSV.
    Accepts headers: time/datetime/date, open/Open, high/High, low/Low,
                     close/Close, volume/Volume
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalise column names
    rename = {}
    for c in df.columns:
        if c in ("datetime", "date", "timestamp"):
            rename[c] = "time"
        for std in ("open", "high", "low", "close", "volume"):
            if c.startswith(std):
                rename[c] = std
    df.rename(columns=rename, inplace=True)

    required = {"time", "open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")

    if "volume" not in df.columns:
        print("⚠  No volume column found — volume gates will be disabled")
        df["volume"] = 1.0

    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    df = df.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


# ──────────────────────────────────────────────────────────────────────────────
# INDICATOR COMPUTATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_indicators(df: pd.DataFrame, tg_len: int = 50, sig_sens: float = 0.5) -> pd.DataFrame:
    """Vectorised translation of Ruby v5 indicator logic."""
    d = df.copy()
    n = len(d)

    # Core
    d["ema9"]   = ta.trend.ema_indicator(d["close"], window=9)
    d["ema20"]  = ta.trend.ema_indicator(d["close"], window=20)
    d["atr14"]  = ta.volatility.average_true_range(d["high"], d["low"], d["close"], window=14)
    d["vol_avg"]= d["volume"].rolling(20).mean()
    d["ao"]     = ta.momentum.awesome_oscillator(d["high"], d["low"])
    d["rsi14"]  = ta.momentum.rsi(d["close"], window=14)

    # VWAP — rolling intraday anchor (approximate)
    d["vwap"]   = (d["close"] * d["volume"]).cumsum() / d["volume"].cumsum()

    # Top G Channel
    d["tg_hi"]  = d["high"].rolling(tg_len).max()
    d["tg_lo"]  = d["low"].rolling(tg_len).min()
    d["tg_range"] = d["tg_hi"] - d["tg_lo"]

    # ATR proximity
    d["tg_prox"] = d["atr14"] * 0.2 * sig_sens

    # ROC normalised
    d["roc8"]     = d["close"].pct_change(8) * 100
    d["roc8_std"] = d["roc8"].rolling(200).std()
    d["roc_norm"] = d["roc8"] / d["roc8_std"].replace(0, np.nan)

    # Volatility percentile (200-bar rolling)
    def vol_pct_rolling(atr_series, window=200):
        arr = atr_series.values
        result = np.full(len(arr), 0.5)
        for i in range(window, len(arr)):
            window_vals = arr[i - window:i]
            result[i] = np.mean(window_vals < arr[i])
        return result

    d["vol_pct"] = vol_pct_rolling(d["atr14"].fillna(0))

    # Market regime (simplified)
    d["ma200"]   = d["close"].rolling(200).mean()
    d["slope_n"] = (d["ma200"] - d["ma200"].shift(20)) / (d["ma200"].diff().abs().rolling(100).mean() * 20 + 1e-10)
    d["ret_s"]   = d["close"].pct_change().rolling(100).std()
    d["ret_sma"] = d["ret_s"].rolling(50).mean()
    d["vol_n"]   = d["ret_s"] / d["ret_sma"].replace(0, np.nan)

    cond_trend_u = d["slope_n"] >  1.0
    cond_trend_d = d["slope_n"] < -1.0
    cond_volatile= d["vol_n"]   >  1.5
    cond_ranging = d["vol_n"]   <  0.8
    d["regime"]  = np.select(
        [cond_trend_u, cond_trend_d, cond_volatile, cond_ranging],
        ["TRENDING_U", "TRENDING_D", "VOLATILE", "RANGING"],
        default="NEUTRAL"
    )

    # Wave ratio (simplified: EMA20 cross momentum proxy)
    d["c_rma"]  = d["close"].ewm(span=10).mean()
    d["o_rma"]  = d["open"].ewm(span=10).mean()
    d["speed"]  = (d["c_rma"] - d["o_rma"]).cumsum()
    # Approximate wave_ratio as ratio of up-speed to down-speed windows
    up_mask     = d["speed"] > 0
    d["wr_raw"] = d["speed"].rolling(50).apply(
        lambda x: np.mean(x[x > 0]) / (np.abs(np.mean(x[x < 0])) + 1e-10) if len(x) > 0 else 1.0,
        raw=True
    ).fillna(1.0)
    d["wave_ratio"] = d["wr_raw"].clip(0.1, 10.0)

    # cur_ratio proxy
    d["cur_ratio"] = (d["c_rma"] - d["o_rma"]) / (d["atr14"] + 1e-10)

    # Rolling percentiles for wave + momentum
    def rolling_pct(series, window=200):
        arr = series.fillna(0).values
        result = np.full(len(arr), 0.5)
        for i in range(window, len(arr)):
            w = arr[i - window:i]
            result[i] = np.mean(w < arr[i])
        return result

    d["wr_pct"]  = rolling_pct(d["wave_ratio"])
    d["mom_pct"] = rolling_pct(d["cur_ratio"].abs())

    # TG signals
    d["strong_bot"] = (
        (d["low"].shift(1) <= d["tg_lo"].shift(1) + d["tg_prox"]) &
        (d["low"] > d["tg_lo"].shift(1)) &
        (d["low"].shift(2) <= d["tg_lo"].shift(2) + d["tg_prox"]) &
        (d["roc_norm"].shift(2) < -2 * sig_sens)
    )

    d["strong_top"] = (
        (d["high"].shift(1) >= d["tg_hi"].shift(1) - d["tg_prox"]) &
        (d["high"] < d["tg_hi"].shift(1)) &
        (d["roc_norm"].shift(2) > 2 * sig_sens) &
        ((d["tg_lo"] - d["tg_lo"].shift(5)).abs() <= d["tg_prox"])
    )

    # HTF bias proxy — use ema9 on 3-bar resampled
    d["htf_bull"] = d["close"] > d["ema9"]
    d["bull_bias"] = (
        (d["close"] > d["vwap"]).astype(int) +
        (d["ao"] > 0).astype(int) +
        d["htf_bull"].astype(int)
    ) >= 2

    # ORB (session-based) — approximate as first 5 bars of each day
    d["date"]       = d["time"].dt.date
    d["bar_of_day"] = d.groupby("date").cumcount()
    orb             = d[d["bar_of_day"] < 5].groupby("date").agg(orb_hi=("high","max"), orb_lo=("low","min"))
    d               = d.join(orb, on="date")
    d["orb_ready"]  = d["bar_of_day"] >= 5
    d["cross_hi"]   = (d["high"] > d["orb_hi"]) & (d["high"].shift(1) <= d["orb_hi"])
    d["cross_lo"]   = (d["low"]  < d["orb_lo"]) & (d["low"].shift(1)  >= d["orb_lo"])
    d["orb_up"]     = d["orb_ready"] & d["cross_hi"] &  d["bull_bias"]
    d["orb_dn"]     = d["orb_ready"] & d["cross_lo"] & ~d["bull_bias"]

    return d.fillna(0)


# ──────────────────────────────────────────────────────────────────────────────
# SIGNAL GENERATION
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Params:
    vol_mult_lo:   float = 0.80
    vol_mult_med:  float = 1.20
    vol_mult_hi:   float = 1.80
    qual_trend:    float = 60.0
    qual_range:    float = 48.0
    qual_volatile: float = 44.0
    cd_min:        int   = 8
    cd_max:        int   = 30
    tp1_base:      float = 1.5
    tp2_base:      float = 2.5
    tp3_base:      float = 4.0
    rsi_ob_trend:  float = 70.0
    rsi_ob_range:  float = 63.0
    rsi_os_trend:  float = 30.0
    rsi_os_range:  float = 37.0
    wave_pct_l:    float = 25.0
    wave_pct_s:    float = 75.0
    mom_pct:       float = 30.0


def generate_signals(d: pd.DataFrame, p: Params, commission: float = 0.0, slippage: float = 0.0) -> dict:
    """
    Vectorised signal generation and trade simulation.
    Returns performance metrics dict.
    """
    n = len(d)

    # Adaptive vol mult
    vol_mult = np.where(d["vol_pct"] >= 0.6, p.vol_mult_hi,
               np.where(d["vol_pct"] >= 0.4, p.vol_mult_med, p.vol_mult_lo))

    # Adaptive quality threshold
    qual_thresh = np.where(d["regime"].isin(["TRENDING_U","TRENDING_D"]), p.qual_trend,
                  np.where(d["regime"] == "VOLATILE", p.qual_volatile, p.qual_range))

    # Adaptive cooldown
    cooldown = np.round(p.cd_min + (p.cd_max - p.cd_min) * d["vol_pct"]).astype(int)

    # RSI thresholds
    rsi_ob = np.where(d["regime"] == "TRENDING_U", p.rsi_ob_trend, p.rsi_ob_range)
    rsi_os = np.where(d["regime"] == "TRENDING_D", p.rsi_os_trend, p.rsi_os_range)

    # Wave + momentum gates
    wave_ok_l = d["wr_pct"]  >= (p.wave_pct_l / 100.0)
    wave_ok_s = d["wr_pct"]  <= 1.0 - (p.wave_pct_s / 100.0)
    mom_ok_l  = (d["mom_pct"] >= (p.mom_pct / 100.0)) & (d["cur_ratio"] > 0)
    mom_ok_s  = (d["mom_pct"] >= (p.mom_pct / 100.0)) & (d["cur_ratio"] < 0)

    # Quality score
    quality = (
        np.where((d["bull_bias"] & (d["ao"] > d["ao"].shift(1))) |
                 (~d["bull_bias"] & (d["ao"] < d["ao"].shift(1))), 20.0, 0.0) +
        np.where((d["close"] > d["ema9"]) & d["bull_bias"] |
                 (d["close"] < d["ema9"]) & ~d["bull_bias"], 15.0, 0.0) +
        np.where((d["bull_bias"] & (d["close"] > d["vwap"])) |
                 (~d["bull_bias"] & (d["close"] < d["vwap"])), 20.0, 0.0) +
        np.where(d["volume"] > d["vol_avg"] * vol_mult, 25.0, 0.0) +
        np.where(d["orb_ready"] & (
            (d["bull_bias"] & d["cross_hi"]) | (~d["bull_bias"] & d["cross_lo"])), 20.0, 0.0)
    )
    quality = np.clip(quality, 0, 100)

    # TP scale from momentum
    abs_cr = d["cur_ratio"].abs()
    tp_mom = np.where(abs_cr >= 1.5, 1.25, np.where(abs_cr >= 1.0, 1.0,
             np.where(abs_cr >= 0.6, 0.85, 0.7)))
    tp1 = p.tp1_base * tp_mom
    tp2 = p.tp2_base * tp_mom

    # SL multiplier
    sl_mult = np.where(d["vol_pct"] >= 0.75, 2.0,
              np.where(d["vol_pct"] >= 0.55, 1.7,
              np.where(d["vol_pct"] >= 0.35, 1.4, 1.2)))

    atr = d["atr14"].values
    close_ = d["close"].values
    high_  = d["high"].values
    low_   = d["low"].values

    # Raw signal masks
    long_raw  = (d["strong_bot"].values &
                 d["bull_bias"].values &
                 (d["ao"].values > 0) &
                 (d["volume"].values > d["vol_avg"].values * vol_mult) &
                 (d["close"].values > d["vwap"].values) &
                 (quality >= qual_thresh) &
                 (d["rsi14"].values < rsi_ob) &
                 wave_ok_l.values & mom_ok_l.values)

    short_raw = (d["strong_top"].values &
                 ~d["bull_bias"].values &
                 (d["ao"].values < 0) &
                 (d["volume"].values > d["vol_avg"].values * vol_mult) &
                 (d["close"].values < d["vwap"].values) &
                 (quality >= qual_thresh) &
                 (d["rsi14"].values > rsi_os) &
                 wave_ok_s.values & mom_ok_s.values)

    # Simulate trades with cooldown
    trades = []
    last_long_bar  = -9999
    last_short_bar = -9999

    for i in range(50, n):  # warmup
        cd = int(cooldown[i])

        if long_raw[i] and (i - last_long_bar > cd):
            last_long_bar = i
            entry   = close_[i] + slippage
            sl_dist = max(entry - (low_[i] - atr[i] * sl_mult[i]), atr[i] * 0.5)
            sl      = entry - sl_dist
            tp1_p   = entry + sl_dist * tp1[i]
            tp2_p   = entry + sl_dist * tp2[i]

            # Forward resolution
            outcome = "open"
            pnl_r   = 0.0
            for j in range(i + 1, min(i + 60, n)):
                if low_[j] - slippage <= sl:
                    pnl_r   = -1.0 - (commission * 2 / sl_dist)
                    outcome = "loss"
                    break
                if high_[j] + slippage >= tp1_p:
                    pnl_r   = tp1[i] - (commission * 2 / sl_dist)
                    outcome = "win_tp1"
                    break
                if high_[j] + slippage >= tp2_p:
                    pnl_r   = tp2[i] - (commission * 2 / sl_dist)
                    outcome = "win_tp2"
                    break
            trades.append({"dir": "long", "bar": i, "outcome": outcome, "pnl_r": pnl_r})

        if short_raw[i] and (i - last_short_bar > cd):
            last_short_bar = i
            entry   = close_[i] - slippage
            sl_dist = max((high_[i] + atr[i] * sl_mult[i]) - entry, atr[i] * 0.5)
            sl      = entry + sl_dist
            tp1_p   = entry - sl_dist * tp1[i]
            tp2_p   = entry - sl_dist * tp2[i]

            outcome = "open"
            pnl_r   = 0.0
            for j in range(i + 1, min(i + 60, n)):
                if high_[j] + slippage >= sl:
                    pnl_r   = -1.0 - (commission * 2 / sl_dist)
                    outcome = "loss"
                    break
                if low_[j] - slippage <= tp1_p:
                    pnl_r   = tp1[i] - (commission * 2 / sl_dist)
                    outcome = "win_tp1"
                    break
                if low_[j] - slippage <= tp2_p:
                    pnl_r   = tp2[i] - (commission * 2 / sl_dist)
                    outcome = "win_tp2"
                    break
            trades.append({"dir": "short", "bar": i, "outcome": outcome, "pnl_r": pnl_r})

    if not trades:
        return {"total": 0, "win_rate": 0.0, "avg_r": 0.0, "profit_factor": 0.0, "score": -1.0}

    tdf = pd.DataFrame(trades)
    resolved = tdf[tdf["outcome"] != "open"]

    if len(resolved) == 0:
        return {"total": 0, "win_rate": 0.0, "avg_r": 0.0, "profit_factor": 0.0, "score": -1.0}

    wins   = resolved[resolved["pnl_r"] > 0]
    losses = resolved[resolved["pnl_r"] <= 0]
    wr     = len(wins) / len(resolved)
    avg_r  = resolved["pnl_r"].mean()
    gross_win  = wins["pnl_r"].sum() if len(wins) else 0.0
    gross_loss = abs(losses["pnl_r"].sum()) if len(losses) else 1e-10
    pf     = gross_win / gross_loss

    # Score: balance win rate, avg R, profit factor, trade count
    score = (wr * 0.4 + min(avg_r / 2.0, 1.0) * 0.3 + min(pf / 3.0, 1.0) * 0.3) * \
            min(len(resolved) / 20.0, 1.0)  # penalise low trade count

    return {
        "total":         len(resolved),
        "win_rate":      wr,
        "avg_r":         avg_r,
        "profit_factor": pf,
        "score":         score,
        "trades":        resolved,
    }


# ──────────────────────────────────────────────────────────────────────────────
# OBJECTIVE FUNCTION
# ──────────────────────────────────────────────────────────────────────────────

def make_objective(train_df: pd.DataFrame, commission: float, slippage: float, min_trades: int):
    def objective(trial: optuna.Trial) -> float:
        p = Params(
            vol_mult_lo   = trial.suggest_float("vol_mult_lo",   0.6,  1.0,  step=0.05),
            vol_mult_med  = trial.suggest_float("vol_mult_med",  1.0,  1.5,  step=0.05),
            vol_mult_hi   = trial.suggest_float("vol_mult_hi",   1.5,  2.5,  step=0.1),
            qual_trend    = trial.suggest_float("qual_trend",    50.0, 75.0, step=5.0),
            qual_range    = trial.suggest_float("qual_range",    35.0, 60.0, step=5.0),
            qual_volatile = trial.suggest_float("qual_volatile", 30.0, 55.0, step=5.0),
            cd_min        = trial.suggest_int(  "cd_min",        5,    20),
            cd_max        = trial.suggest_int(  "cd_max",        20,   60),
            tp1_base      = trial.suggest_float("tp1_base",      1.0,  2.0,  step=0.25),
            tp2_base      = trial.suggest_float("tp2_base",      2.0,  4.0,  step=0.25),
            tp3_base      = trial.suggest_float("tp3_base",      3.0,  6.0,  step=0.5),
            rsi_ob_trend  = trial.suggest_float("rsi_ob_trend",  65.0, 80.0, step=1.0),
            rsi_ob_range  = trial.suggest_float("rsi_ob_range",  58.0, 70.0, step=1.0),
            rsi_os_trend  = trial.suggest_float("rsi_os_trend",  20.0, 35.0, step=1.0),
            rsi_os_range  = trial.suggest_float("rsi_os_range",  30.0, 42.0, step=1.0),
            wave_pct_l    = trial.suggest_float("wave_pct_l",    15.0, 50.0, step=5.0),
            wave_pct_s    = trial.suggest_float("wave_pct_s",    50.0, 85.0, step=5.0),
            mom_pct       = trial.suggest_float("mom_pct",       15.0, 50.0, step=5.0),
        )

        # Enforce tp ordering
        if p.tp1_base >= p.tp2_base or p.tp2_base >= p.tp3_base:
            return -1.0
        if p.cd_min >= p.cd_max:
            return -1.0

        result = generate_signals(train_df, p, commission, slippage)
        if result["total"] < min_trades:
            return -1.0
        return result["score"]

    return objective


# ──────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def walk_forward(df: pd.DataFrame, n_folds: int, train_pct: float,
                 n_trials: int, commission: float, slippage: float, min_trades: int) -> list[dict]:
    fold_size = len(df) // n_folds
    results   = []

    print(f"\n{Fore.CYAN}Walk-forward: {n_folds} folds × {fold_size} bars each{Style.RESET_ALL}")
    print(f"  Train: {int(train_pct*100)}%  Test: {int((1-train_pct)*100)}%  Trials/fold: {n_trials}\n")

    for fold in range(n_folds):
        start = fold * fold_size
        end   = start + fold_size
        fold_df = df.iloc[start:end].reset_index(drop=True)

        split  = int(len(fold_df) * train_pct)
        train  = fold_df.iloc[:split]
        test   = fold_df.iloc[split:]

        print(f"  Fold {fold+1}/{n_folds}  train[{start}:{start+split}]  test[{start+split}:{end}]")

        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(
            make_objective(train, commission, slippage, min_trades),
            n_trials=n_trials,
            show_progress_bar=HAS_TQDM,
        )

        best = study.best_params
        best_p = Params(**{k: (int(v) if k in ("cd_min","cd_max") else v)
                           for k, v in best.items()})

        train_res = generate_signals(train, best_p, commission, slippage)
        test_res  = generate_signals(test,  best_p, commission, slippage)

        wr_trn = train_res["win_rate"] * 100
        wr_tst = test_res["win_rate"]  * 100
        pf_tst = test_res["profit_factor"]
        n_tst  = test_res["total"]

        col = Fore.GREEN if wr_tst >= 50 else Fore.RED
        print(f"    Train WR: {wr_trn:.1f}%  Test WR: {col}{wr_tst:.1f}%{Style.RESET_ALL}"
              f"  PF: {pf_tst:.2f}  Trades: {n_tst}")

        results.append({"fold": fold+1, "params": best_p, "train": train_res, "test": test_res})

    return results


# ──────────────────────────────────────────────────────────────────────────────
# AGGREGATE BEST PARAMS
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_params(results: list[dict]) -> Params:
    """
    Weighted average of per-fold params, weighted by test score.
    Folds with score <= 0 are excluded.
    """
    valid = [r for r in results if r["test"]["score"] > 0]
    if not valid:
        print("⚠  No valid folds — returning defaults")
        return Params()

    weights = np.array([r["test"]["score"] for r in valid])
    weights = weights / weights.sum()

    def wavg(attr):
        vals = np.array([getattr(r["params"], attr) for r in valid])
        return float(np.dot(vals, weights))

    def wavg_int(attr):
        return int(round(wavg(attr)))

    return Params(
        vol_mult_lo   = round(wavg("vol_mult_lo"),   2),
        vol_mult_med  = round(wavg("vol_mult_med"),  2),
        vol_mult_hi   = round(wavg("vol_mult_hi"),   2),
        qual_trend    = round(wavg("qual_trend"),     1),
        qual_range    = round(wavg("qual_range"),     1),
        qual_volatile = round(wavg("qual_volatile"),  1),
        cd_min        = wavg_int("cd_min"),
        cd_max        = wavg_int("cd_max"),
        tp1_base      = round(wavg("tp1_base"),  2),
        tp2_base      = round(wavg("tp2_base"),  2),
        tp3_base      = round(wavg("tp3_base"),  2),
        rsi_ob_trend  = round(wavg("rsi_ob_trend"),  1),
        rsi_ob_range  = round(wavg("rsi_ob_range"),  1),
        rsi_os_trend  = round(wavg("rsi_os_trend"),  1),
        rsi_os_range  = round(wavg("rsi_os_range"),  1),
        wave_pct_l    = round(wavg("wave_pct_l"),    1),
        wave_pct_s    = round(wavg("wave_pct_s"),    1),
        mom_pct       = round(wavg("mom_pct"),       1),
    )


# ──────────────────────────────────────────────────────────────────────────────
# PINE SCRIPT OUTPUT
# ──────────────────────────────────────────────────────────────────────────────

def print_pine_block(p: Params, symbol: str, results: list[dict]):
    test_wrs  = [r["test"]["win_rate"] * 100 for r in results if r["test"]["total"] > 0]
    test_pfs  = [r["test"]["profit_factor"]   for r in results if r["test"]["total"] > 0]
    avg_wr    = np.mean(test_wrs) if test_wrs else 0.0
    avg_pf    = np.mean(test_pfs) if test_pfs else 0.0
    n_trades  = sum(r["test"]["total"] for r in results)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    print("\n" + "═" * 72)
    print(f"{Fore.CYAN}{Style.BRIGHT}  PASTE THIS INTO Ruby v5  SECTION 0{Style.RESET_ALL}")
    print("═" * 72)
    print(f"""
// ─── ruby_optimizer.py output ─────────────────────────────────────────────
// Symbol   : {symbol}
// Generated: {timestamp}
// Walk-fwd avg win rate : {avg_wr:.1f}%
// Walk-fwd avg profit factor: {avg_pf:.2f}
// Total test trades     : {n_trades}
// ───────────────────────────────────────────────────────────────────────────
var float OPT_VOL_MULT_LO   = {p.vol_mult_lo}
var float OPT_VOL_MULT_MED  = {p.vol_mult_med}
var float OPT_VOL_MULT_HI   = {p.vol_mult_hi}
var float OPT_QUAL_TREND    = {p.qual_trend}
var float OPT_QUAL_RANGE    = {p.qual_range}
var float OPT_QUAL_VOLATILE = {p.qual_volatile}
var int   OPT_CD_MIN        = {p.cd_min}
var int   OPT_CD_MAX        = {p.cd_max}
var float OPT_TP1_BASE      = {p.tp1_base}
var float OPT_TP2_BASE      = {p.tp2_base}
var float OPT_TP3_BASE      = {p.tp3_base}
var float OPT_RSI_OB_TREND  = {p.rsi_ob_trend}
var float OPT_RSI_OB_RANGE  = {p.rsi_ob_range}
var float OPT_RSI_OS_TREND  = {p.rsi_os_trend}
var float OPT_RSI_OS_RANGE  = {p.rsi_os_range}
var float OPT_WAVE_PCT_L    = {p.wave_pct_l}
var float OPT_WAVE_PCT_S    = {p.wave_pct_s}
var float OPT_MOM_PCT       = {p.mom_pct}
""")
    print("═" * 72 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Ruby v5 walk-forward optimizer")
    ap.add_argument("--csv",        required=True,      help="OHLCV CSV path")
    ap.add_argument("--symbol",     default="UNKNOWN",  help="Symbol name for output header")
    ap.add_argument("--trials",     type=int,   default=150, help="Optuna trials per fold")
    ap.add_argument("--folds",      type=int,   default=5,   help="Walk-forward folds")
    ap.add_argument("--train_pct",  type=float, default=0.7, help="Train fraction per fold")
    ap.add_argument("--commission", type=float, default=0.0, help="Per-side commission (price units)")
    ap.add_argument("--slippage",   type=float, default=0.0, help="Fill slippage (price units)")
    ap.add_argument("--min_trades", type=int,   default=10,  help="Min trades for a fold to count")
    ap.add_argument("--tg_len",     type=int,   default=50,  help="Top G channel length")
    ap.add_argument("--sig_sens",   type=float, default=0.5, help="Signal sensitivity")
    args = ap.parse_args()

    print(f"\n{Fore.CYAN}Ruby v5 Optimizer{Style.RESET_ALL}  {args.symbol}")
    print(f"Loading {args.csv}...")
    df = load_csv(args.csv)
    print(f"  {len(df):,} bars  {df['time'].min()} → {df['time'].max()}")

    print("Computing indicators...")
    df = compute_indicators(df, tg_len=args.tg_len, sig_sens=args.sig_sens)
    print(f"  Done.  {len(df):,} bars ready.\n")

    results = walk_forward(
        df,
        n_folds    = args.folds,
        train_pct  = args.train_pct,
        n_trials   = args.trials,
        commission = args.commission,
        slippage   = args.slippage,
        min_trades = args.min_trades,
    )

    best_p = aggregate_params(results)
    print_pine_block(best_p, args.symbol, results)

    # Also run on full dataset for sanity check
    print("Full-dataset sanity check with aggregated params...")
    full = generate_signals(df, best_p, args.commission, args.slippage)
    c = Fore.GREEN if full["win_rate"] >= 0.5 else Fore.RED
    print(f"  Trades: {full['total']}  WR: {c}{full['win_rate']*100:.1f}%{Style.RESET_ALL}"
          f"  Avg R: {full['avg_r']:.3f}  PF: {full['profit_factor']:.2f}")


if __name__ == "__main__":
    main()
