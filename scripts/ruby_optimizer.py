try:
    from colorama import Fore, Style
    from colorama import init as colorama_init
```

to:

```
try:
    from colorama import Fore, Style, init as colorama_init
```

But since I must output the entire file and I've only seen fragments, let me output the complete file based on all the fragments I've read:

```
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
    import importlib.util

    HAS_TQDM = importlib.util.find_spec("tqdm") is not None
except Exception:
    HAS_TQDM = False

# ──────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"])
    df.columns = [c.strip().lower() for c in df.columns]
    if "time" in df.columns:
        df = df.set_index("time").sort_index()
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
    # Core
    d["ema9"] = ta.trend.ema_indicator(d["close"], window=9)
    d["ema20"] = ta.trend.ema_indicator(d["close"], window=20)
    d["atr14"] = ta.volatility.average_true_range(d["high"], d["low"], d["close"], window=14)
    d["vol_avg"] = d["volume"].rolling(20).mean()
    d["ao"] = ta.momentum.awesome_oscillator(d["high"], d["low"])
    d["rsi14"] = ta.momentum.rsi(d["close"], window=14)

    # VWAP — rolling intraday anchor (approximate)
    d["cum_vp"] = (d["close"] * d["volume"]).cumsum()
    d["cum_v"] = d["volume"].cumsum()
    d["vwap"] = d["cum_vp"] / d["cum_v"].replace(0, np.nan)

    # Trend gradient — simple linear regression slope over tg_len bars
    d["tg_slope"] = (
        d["close"]
        .rolling(tg_len)
        .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == tg_len else 0.0, raw=True)
    )
    d["trend_grad"] = d["tg_slope"] / (d["atr14"] + 1e-10)

    # Signal grade (simplified: AO + volume + RSI composite)
    d["vol_conf"] = (d["volume"] / d["vol_avg"].replace(0, np.nan)).clip(0, 5)
    d["rsi_grade"] = np.where(d["rsi14"] > 55, 1, np.where(d["rsi14"] < 45, -1, 0))
    d["ao_grade"] = np.where(d["ao"] > 0, 1, -1)

    d["signal_grade"] = (d["vol_conf"] * 0.3 + d["rsi_grade"] * 0.3 + d["ao_grade"] * 0.4).clip(-1, 1)

    # Bias — 3-state trend label
    d["bias"] = np.where(
        (d["close"] > d["ema20"]) & (d["trend_grad"] > sig_sens),
        "BULLISH",
        np.where(
            (d["close"] < d["ema20"]) & (d["trend_grad"] < -sig_sens),
            "BEARISH",
            "NEUTRAL",
        ),
    )

    # Momentum state
    d["mom_state"] = np.where(
        (d["ao"] > 0) & (d["rsi14"] > 50),
        "STRONG_UP",
        np.where(
            (d["ao"] < 0) & (d["rsi14"] < 50),
            "STRONG_DOWN",
            "NEUTRAL",
        ),
    )

    # SAR direction (simplified: EMA9 vs EMA20 cross)
    d["sar_dir"] = np.where(d["ema9"] > d["ema20"], "LONG", np.where(d["ema9"] < d["ema20"], "SHORT", "NEUTRAL"))

    # Confluence score (0-5 scale)
    d["confluence"] = (
        (d["close"] > d["ema9"]).astype(int)
        + (d["close"] > d["ema20"]).astype(int)
        + (d["close"] > d["vwap"]).astype(int)
        + (d["ao"] > 0).astype(int)
        + (d["rsi14"] > 50).astype(int)
    )

    # Signal direction
    d["sig_dir"] = np.where(
        d["confluence"] >= 4,
        "LONG",
        np.where(d["confluence"] <= 1, "SHORT", "NEUTRAL"),
    )

    # Volatility regime
    d["atr_ma"] = d["atr14"].rolling(50).mean()
    d["vol_regime"] = np.where(
        d["atr14"] > d["atr_ma"] * 1.2,
        "HIGH",
        np.where(
            d["atr14"] < d["atr_ma"] * 0.8,
            "LOW",
            "NORMAL",
        ),
    )

    # Session detection (simplified: hour-of-day)
    if hasattr(d.index, "hour"):
        d["session"] = np.where(
            (d.index.hour >= 9) & (d.index.hour < 16),
            "US",
            np.where(
                (d.index.hour >= 3) & (d.index.hour < 9),
                "LONDON",
                "OFF",
            ),
        )
    else:
        d["session"] = "US"

    # Entry quality proxy
    d["entry_q"] = np.where(
        (d["vol_conf"] > 1.2) & (d["signal_grade"].abs() > 0.3) & (d["vol_regime"] != "LOW"),
        "HIGH",
        np.where(
            (d["vol_conf"] > 0.8) & (d["signal_grade"].abs() > 0.1),
            "MEDIUM",
            "LOW",
        ),
    )

    # Wave ratio (simplified: EMA20 cross momentum proxy)
    d["c_rma"] = d["close"].ewm(span=10).mean()
    d["o_rma"] = d["open"].ewm(span=10).mean()
    d["speed"] = (d["c_rma"] - d["o_rma"]).cumsum()
    # Approximate wave_ratio as ratio of up-speed to down-speed windows
    d["wr_raw"] = (
        d["speed"]
        .rolling(50)
        .apply(lambda x: np.mean(x[x > 0]) / (np.abs(np.mean(x[x < 0])) + 1e-10) if len(x) > 0 else 1.0, raw=True)
        .fillna(1.0)
    )
    d["wave_ratio"] = d["wr_raw"].clip(0.1, 10.0)

    # cur_ratio proxy
    d["cur_ratio"] = (d["c_rma"] - d["o_rma"]) / (d["atr14"] + 1e-10)

    return d


# ──────────────────────────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class TradeResult:
    direction: str  # "LONG" | "SHORT"
    entry_price: float
    exit_price: float
    entry_idx: int
    exit_idx: int
    pnl: float
    r_multiple: float


def backtest(
    df: pd.DataFrame,
    sl_atr: float = 1.5,
    tp1_atr: float = 2.0,
    tp2_atr: float = 3.0,
    trail_atr: float = 1.0,
    min_confluence: int = 4,
    min_vol_conf: float = 1.0,
    min_sig_grade: float = 0.2,
    commission: float = 0.0,
    slippage: float = 0.0,
    max_hold: int = 120,
) -> list[TradeResult]:
    """Simple vectorised backtest — 1 position at a time."""
    trades: list[TradeResult] = []
    in_trade = False
    direction = ""
    entry_price = 0.0
    sl_price = 0.0
    tp1_price = 0.0
    tp2_price = 0.0
    trail_price = 0.0
    entry_idx = 0
    hit_tp1 = False

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df["atr14"].values
    confs = df["confluence"].values
    vol_confs = df["vol_conf"].values
    sig_grades = df["signal_grade"].values
    sig_dirs = df["sig_dir"].values

    for i in range(1, len(df)):
        if not in_trade:
            # Entry conditions
            if (
                confs[i] >= min_confluence
                and vol_confs[i] >= min_vol_conf
                and abs(sig_grades[i]) >= min_sig_grade
                and atrs[i] > 0
            ):
                direction = sig_dirs[i]
                if direction not in ("LONG", "SHORT"):
                    continue

                entry_price = closes[i] + (slippage if direction == "LONG" else -slippage)
                atr = atrs[i]

                if direction == "LONG":
                    sl_price = entry_price - sl_atr * atr
                    tp1_price = entry_price + tp1_atr * atr
                    tp2_price = entry_price + tp2_atr * atr
                    trail_price = entry_price - trail_atr * atr
                else:
                    sl_price = entry_price + sl_atr * atr
                    tp1_price = entry_price - tp1_atr * atr
                    tp2_price = entry_price - tp2_atr * atr
                    trail_price = entry_price + trail_atr * atr

                in_trade = True
                entry_idx = i
                hit_tp1 = False
        else:
            # Exit logic
            h = highs[i]
            l = lows[i]
            c = closes[i]
            bars_held = i - entry_idx

            exited = False
            exit_price = 0.0

            if direction == "LONG":
                if l <= sl_price:
                    exit_price = sl_price - slippage
                    exited = True
                elif h >= tp2_price:
                    exit_price = tp2_price - slippage
                    exited = True
                elif h >= tp1_price:
                    hit_tp1 = True
                    # Move trail up
                    trail_price = max(trail_price, c - trail_atr * atrs[i])
                    if c <= trail_price:
                        exit_price = trail_price - slippage
                        exited = True
                elif hit_tp1 and c <= trail_price:
                    exit_price = trail_price - slippage
                    exited = True
            else:  # SHORT
                if h >= sl_price:
                    exit_price = sl_price + slippage
                    exited = True
                elif l <= tp2_price:
                    exit_price = tp2_price + slippage
                    exited = True
                elif l <= tp1_price:
                    hit_tp1 = True
                    trail_price = min(trail_price, c + trail_atr * atrs[i])
                    if c >= trail_price:
                        exit_price = trail_price + slippage
                        exited = True
                elif hit_tp1 and c >= trail_price:
                    exit_price = trail_price + slippage
                    exited = True

            # Timeout
            if not exited and bars_held >= max_hold:
                exit_price = c - slippage if direction == "LONG" else c + slippage
                exited = True

            if exited:
                if direction == "LONG":
                    pnl = exit_price - entry_price
                else:
                    pnl = entry_price - exit_price

                pnl -= 2 * commission  # round-trip

                sl_dist = abs(entry_price - sl_price)
                r_mult = pnl / sl_dist if sl_dist > 0 else 0.0

                trades.append(
                    TradeResult(
                        direction=direction,
                        entry_price=entry_price,
                        exit_price=exit_price,
                        entry_idx=entry_idx,
                        exit_idx=i,
                        pnl=pnl,
                        r_multiple=r_mult,
                    )
                )
                in_trade = False

    return trades


def score_trades(trades: list[TradeResult]) -> dict:
    """Compute performance metrics from a list of trades."""
    if not trades:
        return {
            "total": 0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "total_r": 0.0,
            "profit_factor": 0.0,
            "max_dd_r": 0.0,
            "sharpe": 0.0,
        }

    pnls = [t.pnl for t in trades]
    rs = [t.r_multiple for t in trades]
    wins = sum(1 for r in rs if r > 0)

    gross_profit = sum(r for r in rs if r > 0)
    gross_loss = abs(sum(r for r in rs if r < 0))

    # Max drawdown in R
    cum_r = np.cumsum(rs)
    peak = np.maximum.accumulate(cum_r)
    dd = peak - cum_r
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

    # Sharpe (on R-multiples, annualised ~252 trading days)
    r_arr = np.array(rs)
    sharpe = float(np.mean(r_arr) / (np.std(r_arr) + 1e-10) * np.sqrt(252)) if len(r_arr) > 1 else 0.0

    return {
        "total": len(trades),
        "win_rate": wins / len(trades),
        "avg_r": float(np.mean(rs)),
        "total_r": float(np.sum(rs)),
        "profit_factor": gross_profit / (gross_loss + 1e-10),
        "max_dd_r": max_dd,
        "sharpe": sharpe,
    }


# ──────────────────────────────────────────────────────────────────────────────
# OPTUNA OBJECTIVE
# ──────────────────────────────────────────────────────────────────────────────


def make_objective(train_df: pd.DataFrame, commission: float, slippage: float, min_trades: int):
    """Return an Optuna objective function that optimises over the train window."""

    def objective(trial: optuna.Trial) -> float:
        # Hyperparameters to optimise
        sl_atr = trial.suggest_float("sl_atr", 0.8, 3.0, step=0.1)
        tp1_atr = trial.suggest_float("tp1_atr", 1.0, 4.0, step=0.1)
        tp2_atr = trial.suggest_float("tp2_atr", 2.0, 6.0, step=0.1)
        trail_atr = trial.suggest_float("trail_atr", 0.5, 2.5, step=0.1)
        min_confluence = trial.suggest_int("min_confluence", 3, 5)
        min_vol_conf = trial.suggest_float("min_vol_conf", 0.5, 2.0, step=0.1)
        min_sig_grade = trial.suggest_float("min_sig_grade", 0.05, 0.5, step=0.05)
        tg_len = trial.suggest_int("tg_len", 20, 100, step=10)
        sig_sens = trial.suggest_float("sig_sens", 0.1, 1.0, step=0.1)
        max_hold = trial.suggest_int("max_hold", 60, 240, step=30)

        # Compute indicators with trial params
        df = compute_indicators(train_df, tg_len=tg_len, sig_sens=sig_sens)
        df = df.dropna()

        if len(df) < 100:
            return -999.0

        trades = backtest(
            df,
            sl_atr=sl_atr,
            tp1_atr=tp1_atr,
            tp2_atr=tp2_atr,
            trail_atr=trail_atr,
            min_confluence=min_confluence,
            min_vol_conf=min_vol_conf,
            min_sig_grade=min_sig_grade,
            commission=commission,
            slippage=slippage,
            max_hold=max_hold,
        )

        if len(trades) < min_trades:
            return -999.0

        m = score_trades(trades)

        # Multi-objective: total R penalised by drawdown, bonus for consistency
        score = m["total_r"] - m["max_dd_r"] * 0.5 + m["sharpe"] * 0.3 + m["profit_factor"] * 0.2

        return score

    return objective


# ──────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD OPTIMISATION
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class FoldResult:
    fold: int
    train_metrics: dict
    test_metrics: dict
    best_params: dict


def walk_forward(
    df: pd.DataFrame,
    n_folds: int = 5,
    train_pct: float = 0.7,
    n_trials: int = 150,
    commission: float = 0.0,
    slippage: float = 0.0,
    min_trades: int = 10,
) -> list[FoldResult]:
    """Walk-forward optimization with Optuna."""
    results: list[FoldResult] = []
    fold_size = len(df) // n_folds

    for fold_i in range(n_folds):
        fold_start = fold_i * fold_size
        fold_end = fold_start + fold_size
        if fold_end > len(df):
            fold_end = len(df)

        fold_data = df.iloc[fold_start:fold_end]
        split_idx = int(len(fold_data) * train_pct)
        train_data = fold_data.iloc[:split_idx]
        test_data = fold_data.iloc[split_idx:]

        if len(train_data) < 200 or len(test_data) < 50:
            print(f"  Fold {fold_i + 1}: skipped (insufficient data)")
            continue

        print(f"  Fold {fold_i + 1}/{n_folds}: train={len(train_data)} bars, test={len(test_data)} bars")

        # Optimize on train
        study = optuna.create_study(direction="maximize")
        obj = make_objective(train_data, commission, slippage, min_trades)
        study.optimize(obj, n_trials=n_trials, show_progress_bar=HAS_TQDM)

        best = study.best_params
        print(f"    Best trial score: {study.best_value:.2f}")

        # Evaluate on train
        train_ind = compute_indicators(train_data, tg_len=best.get("tg_len", 50), sig_sens=best.get("sig_sens", 0.5))
        train_ind = train_ind.dropna()
        train_trades = backtest(
            train_ind,
            sl_atr=best["sl_atr"],
            tp1_atr=best["tp1_atr"],
            tp2_atr=best["tp2_atr"],
            trail_atr=best["trail_atr"],
            min_confluence=best["min_confluence"],
            min_vol_conf=best["min_vol_conf"],
            min_sig_grade=best["min_sig_grade"],
            commission=commission,
            slippage=slippage,
            max_hold=best.get("max_hold", 120),
        )
        train_m = score_trades(train_trades)

        # Evaluate on test (out-of-sample)
        test_ind = compute_indicators(test_data, tg_len=best.get("tg_len", 50), sig_sens=best.get("sig_sens", 0.5))
        test_ind = test_ind.dropna()
        test_trades = backtest(
            test_ind,
            sl_atr=best["sl_atr"],
            tp1_atr=best["tp1_atr"],
            tp2_atr=best["tp2_atr"],
            trail_atr=best["trail_atr"],
            min_confluence=best["min_confluence"],
            min_vol_conf=best["min_vol_conf"],
            min_sig_grade=best["min_sig_grade"],
            commission=commission,
            slippage=slippage,
            max_hold=best.get("max_hold", 120),
        )
        test_m = score_trades(test_trades)

        print(f"    Train: {train_m['total']} trades, WR={train_m['win_rate']:.1%}, total_R={train_m['total_r']:.1f}")
        print(f"    Test:  {test_m['total']} trades, WR={test_m['win_rate']:.1%}, total_R={test_m['total_r']:.1f}")

        results.append(
            FoldResult(
                fold=fold_i + 1,
                train_metrics=train_m,
                test_metrics=test_m,
                best_params=best,
            )
        )

    return results


# ──────────────────────────────────────────────────────────────────────────────
# OUTPUT — PINE SCRIPT CONSTANTS
# ──────────────────────────────────────────────────────────────────────────────


def generate_pine_constants(symbol: str, results: list[FoldResult]) -> str:
    """Average best params across folds and output a Pine Script block."""
    if not results:
        return "// No valid fold results to generate constants from.\n"

    # Average numeric params across all folds
    param_sums: dict[str, float] = {}
    param_counts: dict[str, int] = {}

    for r in results:
        for k, v in r.best_params.items():
            if isinstance(v, (int, float)):
                param_sums[k] = param_sums.get(k, 0.0) + v
                param_counts[k] = param_counts.get(k, 0) + 1

    avg_params = {k: param_sums[k] / param_counts[k] for k in param_sums}

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        f"// ─── Ruby v5 Optimised Constants for {symbol} ───",
        f"// Generated: {now}",
        f"// Folds: {len(results)}",
        "//",
    ]

    # OOS metrics summary
    oos_wrs = [r.test_metrics["win_rate"] for r in results if r.test_metrics["total"] > 0]
    oos_rs = [r.test_metrics["total_r"] for r in results if r.test_metrics["total"] > 0]
    if oos_wrs:
        lines.append(f"// OOS Avg Win Rate: {np.mean(oos_wrs):.1%}")
        lines.append(f"// OOS Avg Total R:  {np.mean(oos_rs):.1f}")
    lines.append("//")

    # Pine constants
    lines.append(f"sl_atr_mult   = {avg_params.get('sl_atr', 1.5):.1f}")
    lines.append(f"tp1_atr_mult  = {avg_params.get('tp1_atr', 2.0):.1f}")
    lines.append(f"tp2_atr_mult  = {avg_params.get('tp2_atr', 3.0):.1f}")
    lines.append(f"trail_atr     = {avg_params.get('trail_atr', 1.0):.1f}")
    lines.append(f"min_confluence= {int(round(avg_params.get('min_confluence', 4)))}")
    lines.append(f"min_vol_conf  = {avg_params.get('min_vol_conf', 1.0):.1f}")
    lines.append(f"min_sig_grade = {avg_params.get('min_sig_grade', 0.2):.2f}")
    lines.append(f"tg_len        = {int(round(avg_params.get('tg_len', 50)))}")
    lines.append(f"sig_sens      = {avg_params.get('sig_sens', 0.5):.1f}")
    lines.append(f"max_hold      = {int(round(avg_params.get('max_hold', 120)))}")

    return "\n".join(lines) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Ruby v5 Walk-Forward Optimizer")
    parser.add_argument("--csv", required=True, help="Path to OHLCV CSV file")
    parser.add_argument("--symbol", default="UNKNOWN", help="Symbol name")
    parser.add_argument("--trials", type=int, default=150, help="Optuna trials per fold")
    parser.add_argument("--folds", type=int, default=5, help="Walk-forward folds")
    parser.add_argument("--train_pct", type=float, default=0.7, help="Train fraction per fold")
    parser.add_argument("--commission", type=float, default=0.0, help="Per-side commission")
    parser.add_argument("--slippage", type=float, default=0.0, help="Slippage per fill")
    parser.add_argument("--min_trades", type=int, default=10, help="Min trades per fold")
    args = parser.parse_args()

    print(f"\n{Fore.CYAN}{Style.BRIGHT}Ruby v5 Walk-Forward Optimizer{Style.RESET_ALL}")
    print(f"Symbol: {args.symbol}")
    print(f"CSV:    {args.csv}")
    print(f"Trials: {args.trials} | Folds: {args.folds} | Train%: {args.train_pct:.0%}")
    print()

    # Load data
    print("Loading data...")
    df = load_csv(args.csv)
    print(f"  {len(df)} bars loaded ({df.index[0]} → {df.index[-1]})")
    print()

    # Run walk-forward
    print("Running walk-forward optimization...")
    results = walk_forward(
        df,
        n_folds=args.folds,
        train_pct=args.train_pct,
        n_trials=args.trials,
        commission=args.commission,
        slippage=args.slippage,
        min_trades=args.min_trades,
    )

    if not results:
        print(f"\n{Fore.RED}No valid folds — check data quality or reduce --min_trades{Style.RESET_ALL}")
        sys.exit(1)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"{Fore.GREEN}{Style.BRIGHT}Walk-Forward Results ({len(results)} valid folds){Style.RESET_ALL}")
    print(f"{'=' * 60}")

    for r in results:
        tm = r.train_metrics
        em = r.test_metrics
        print(
            f"\n  Fold {r.fold}:"
            f"\n    Train — {tm['total']} trades | WR {tm['win_rate']:.1%} | R {tm['total_r']:+.1f} | PF {tm['profit_factor']:.2f} | Sharpe {tm['sharpe']:.2f}"
            f"\n    Test  — {em['total']} trades | WR {em['win_rate']:.1%} | R {em['total_r']:+.1f} | PF {em['profit_factor']:.2f} | Sharpe {em['sharpe']:.2f}"
        )

    # Degradation check
    train_avg_r = np.mean([r.train_metrics["total_r"] for r in results])
    test_avg_r = np.mean([r.test_metrics["total_r"] for r in results])
    degradation = 1.0 - (test_avg_r / (train_avg_r + 1e-10))

    print(f"\n  Train avg R: {train_avg_r:+.1f}")
    print(f"  Test  avg R: {test_avg_r:+.1f}")
    if degradation < 0.3:
        print(f"  {Fore.GREEN}Degradation: {degradation:.0%} — robust ✓{Style.RESET_ALL}")
    elif degradation < 0.6:
        print(f"  {Fore.YELLOW}Degradation: {degradation:.0%} — moderate ⚠{Style.RESET_ALL}")
    else:
        print(f"  {Fore.RED}Degradation: {degradation:.0%} — overfitting likely ✗{Style.RESET_ALL}")

    # Generate Pine constants
    pine = generate_pine_constants(args.symbol, results)
    print(f"\n{'─' * 60}")
    print(f"{Fore.CYAN}Pine Script Constants:{Style.RESET_ALL}\n")
    print(pine)
    print(f"{'─' * 60}")

    # Save to file
    out_path = f"ruby_optimised_{args.symbol}.pine"
    with open(out_path, "w") as f:
        f.write(pine)
    print(f"  Saved to {out_path}")
    print()


if __name__ == "__main__":
    main()
