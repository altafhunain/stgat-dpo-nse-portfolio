"""
NSE Portfolio GNN — Phase 2: Benchmark Strategies
===================================================
Computes all 7 baseline portfolio strategies under 5-fold
walk-forward validation + 2025-2026 blind holdout.

Run:
    python training/run_benchmarks.py

Outputs:
    outputs/results/benchmark_fold_results.csv   ← per-fold metrics
    outputs/results/benchmark_summary.csv        ← mean ± std across folds
    outputs/results/benchmark_returns.parquet    ← daily return series
    outputs/results/nifty50_returns.csv          ← benchmark index returns
    outputs/figures/cumulative_wealth.png        ← wealth curves
"""

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
DATA_DIR   = ROOT / "data" / "raw"
OUT_DIR    = ROOT / "outputs" / "results"
FIG_DIR    = ROOT / "outputs" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
PRICES_FILE    = DATA_DIR / "prices_51.parquet"
TRAIN_START    = "2018-01-01"
BLIND_START    = "2026-01-01"
BLIND_END      = "2026-08-28"
RF_ANNUAL      = 0.065          # 6.5% RBI repo rate proxy
TC_BPS         = 10             # transaction cost per leg (basis points)
MAX_WEIGHT     = 0.15           # 15% max per stock
MOM_TOP_N      = 15             # momentum: top-N stocks
MOM_LOOKBACK   = 252            # 12-month lookback
MOM_SKIP       = 21             # skip last month
SEED           = 42

# Walk-forward folds: (train_end, test_start, test_end)
FOLDS = [
    ("2020-12-31", "2021-01-01", "2021-12-31"),
    ("2021-06-30", "2021-07-01", "2022-06-30"),
    ("2021-12-31", "2022-01-01", "2022-12-31"),
    ("2022-06-30", "2022-07-01", "2023-06-30"),
    ("2022-12-31", "2023-01-01", "2023-12-31"),
    ("2023-06-30", "2023-07-01", "2024-06-30"),
    ("2023-12-31", "2024-01-01", "2024-12-31"),
    ("2024-06-30", "2024-07-01", "2025-06-30"),
    ("2024-12-31", "2025-01-01", "2025-12-31"),
]

np.random.seed(SEED)


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════
def load_prices() -> pd.DataFrame:
    """Load 51-stock close prices from parquet."""
    df = pd.read_parquet(PRICES_FILE)
    tickers = sorted(set(df.columns.get_level_values(0)))
    close = pd.DataFrame(
        {t: df[(t, t, "Close")].squeeze() for t in tickers},
        index=df.index
    )
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    # Forward-fill gaps (max 5 days)
    close = close.ffill(limit=5)
    return close


def fetch_nifty50(start: str, end: str) -> pd.Series:
    """Download Nifty 50 index from Yahoo Finance."""
    try:
        import yfinance as yf
        raw = yf.download("^NSEI", start=start, end=end,
                          auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].squeeze()
        else:
            close = raw["Close"]
        close.index = pd.to_datetime(close.index)
        close.name  = "NIFTY50"
        print(f"  ✓  Nifty 50: {len(close):,} days  "
              f"({close.index.min().date()} → {close.index.max().date()})")
        return close
    except Exception as e:
        print(f"  ⚠  Could not fetch Nifty 50: {e}")
        print("     Using equal-weight portfolio of all 51 stocks as proxy.")
        return None


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns, drop first row."""
    return np.log(prices / prices.shift(1)).iloc[1:]


# ═════════════════════════════════════════════════════════════════════════════
# PORTFOLIO UTILITIES
# ═════════════════════════════════════════════════════════════════════════════
def apply_transaction_costs(weights_prev: np.ndarray,
                             weights_new: np.ndarray,
                             tc_bps: float = TC_BPS) -> float:
    """Return total cost as fraction of portfolio value."""
    turnover = np.abs(weights_new - weights_prev).sum()
    return turnover * (tc_bps / 10_000)


def get_rebalance_dates(index: pd.DatetimeIndex,
                        start: str, end: str) -> pd.DatetimeIndex:
    """First trading day of each month within [start, end]."""
    mask  = (index >= start) & (index <= end)
    sub   = index[mask]
    monthly = sub.to_series().groupby([sub.year, sub.month]).first()
    return pd.DatetimeIndex(monthly.values)


def portfolio_returns(weights_schedule: pd.DataFrame,
                      daily_returns: pd.DataFrame,
                      tc_bps: float = TC_BPS) -> pd.Series:
    """
    Given a DataFrame of weights (rows = rebalance dates, cols = tickers),
    compute daily portfolio log returns with transaction costs.
    """
    port_rets = []
    prev_w    = np.zeros(weights_schedule.shape[1])

    for i, (reb_date, next_reb) in enumerate(
        zip(weights_schedule.index, list(weights_schedule.index[1:]) + [daily_returns.index[-1]])
    ):
        new_w = weights_schedule.loc[reb_date].values
        cost  = apply_transaction_costs(prev_w, new_w, tc_bps)

        # Daily returns in this holding period
        mask   = (daily_returns.index >= reb_date) & (daily_returns.index < next_reb)
        period = daily_returns.loc[mask, weights_schedule.columns]
        if period.empty:
            continue

        daily_p = period.fillna(0).values @ new_w
        # Subtract cost on first day only
        if len(daily_p) > 0:
            daily_p[0] -= cost

        dates = period.index
        for d, r in zip(dates, daily_p):
            port_rets.append((d, r))

        prev_w = new_w.copy()

    if not port_rets:
        return pd.Series(dtype=float)

    idx  = pd.DatetimeIndex([x[0] for x in port_rets])
    vals = np.array([x[1] for x in port_rets])
    return pd.Series(vals, index=idx, name="portfolio")


# ═════════════════════════════════════════════════════════════════════════════
# BENCHMARK STRATEGIES
# ═════════════════════════════════════════════════════════════════════════════
def strategy_equal_weight(tickers, rebalance_dates):
    """B1: 1/N equal weight."""
    n  = len(tickers)
    w  = np.ones(n) / n
    df = pd.DataFrame(
        np.tile(w, (len(rebalance_dates), 1)),
        index=rebalance_dates, columns=tickers
    )
    return df


def strategy_buy_hold(prices: pd.DataFrame, start: str,
                      rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """B2: Equal weight at inception, never rebalance (simulate by keeping initial)."""
    tickers = prices.columns.tolist()
    n       = len(tickers)
    # Buy at start — compute drift weights each rebalance date
    start_prices = prices.loc[prices.index >= start].iloc[0]
    shares       = (1.0 / n) / start_prices   # equal INR allocation

    ws = []
    for d in rebalance_dates:
        p   = prices.loc[d]
        val = (shares * p)
        ws.append(val / val.sum())

    return pd.DataFrame(ws, index=rebalance_dates, columns=tickers)


def strategy_min_variance(returns_train: pd.DataFrame,
                           rebalance_dates: pd.DatetimeIndex,
                           max_w: float = MAX_WEIGHT) -> pd.DataFrame:
    """B3a: Minimum variance portfolio."""
    tickers = returns_train.columns.tolist()
    n       = len(tickers)
    lw      = LedoitWolf().fit(returns_train.dropna())
    cov     = lw.covariance_

    def port_var(w):
        return w @ cov @ w

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds      = [(0, max_w)] * n
    w0          = np.ones(n) / n

    result = minimize(port_var, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-9})

    w_opt = result.x if result.success else w0
    w_opt = np.clip(w_opt, 0, max_w)
    w_opt /= w_opt.sum()

    df = pd.DataFrame(
        np.tile(w_opt, (len(rebalance_dates), 1)),
        index=rebalance_dates, columns=tickers
    )
    return df


def strategy_max_sharpe(returns_train: pd.DataFrame,
                         rebalance_dates: pd.DatetimeIndex,
                         rf_daily: float,
                         max_w: float = MAX_WEIGHT) -> pd.DataFrame:
    """B4: Maximum Sharpe ratio (classical MVO)."""
    tickers = returns_train.columns.tolist()
    n       = len(tickers)
    clean   = returns_train.dropna()
    lw      = LedoitWolf().fit(clean)
    cov     = lw.covariance_
    mu      = clean.mean().values

    def neg_sharpe(w):
        port_ret = w @ mu - rf_daily
        port_vol = np.sqrt(w @ cov @ w)
        return -port_ret / (port_vol + 1e-8)

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds      = [(0, max_w)] * n
    w0          = np.ones(n) / n

    result = minimize(neg_sharpe, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-9})

    w_opt = result.x if result.success else w0
    w_opt = np.clip(w_opt, 0, max_w)
    w_opt /= w_opt.sum()

    df = pd.DataFrame(
        np.tile(w_opt, (len(rebalance_dates), 1)),
        index=rebalance_dates, columns=tickers
    )
    return df


def strategy_risk_parity(returns_train: pd.DataFrame,
                          rebalance_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """B5: Risk parity — weights inversely proportional to volatility."""
    tickers = returns_train.columns.tolist()
    vols    = returns_train.std().values
    vols    = np.where(vols < 1e-8, 1e-8, vols)
    w       = (1.0 / vols)
    w      /= w.sum()
    w       = np.clip(w, 0, MAX_WEIGHT)
    w      /= w.sum()

    df = pd.DataFrame(
        np.tile(w, (len(rebalance_dates), 1)),
        index=rebalance_dates, columns=tickers
    )
    return df


def strategy_momentum(prices: pd.DataFrame,
                       rebalance_dates: pd.DatetimeIndex,
                       lookback: int = MOM_LOOKBACK,
                       skip: int = MOM_SKIP,
                       top_n: int = MOM_TOP_N) -> pd.DataFrame:
    """B6: 12-1 month momentum — equal weight in top-N stocks."""
    tickers  = prices.columns.tolist()
    n        = len(tickers)
    rows     = []

    for d in rebalance_dates:
        # Snap to nearest available trading day
        loc_arr = prices.index.get_indexer([d], method="nearest")
        loc     = int(loc_arr[0])
        end_i   = max(0, loc - skip)
        start_i = max(0, loc - lookback)
        if end_i <= start_i:
            w = np.ones(n) / n
        else:
            window = prices.iloc[start_i:end_i]
            ret    = (window.iloc[-1] / window.iloc[0]) - 1
            ranked = ret.rank(ascending=False)
            top    = ranked[ranked <= top_n].index
            w      = np.zeros(n)
            for t in top:
                w[tickers.index(t)] = 1.0 / len(top)
        rows.append(w)

    return pd.DataFrame(rows, index=rebalance_dates, columns=tickers)


def strategy_inverse_vol(returns_train: pd.DataFrame,
                          rebalance_dates: pd.DatetimeIndex,
                          window: int = 63) -> pd.DataFrame:
    """B7: Inverse volatility with rolling window — a stronger naive baseline."""
    tickers = returns_train.columns.tolist()
    n       = len(tickers)
    rows    = []
    for d in rebalance_dates:
        # Use last `window` days of training data
        sub  = returns_train.loc[:d].tail(window)
        vols = sub.std().values
        vols = np.where(vols < 1e-8, 1e-8, vols)
        w    = 1.0 / vols
        w    = np.clip(w / w.sum(), 0, MAX_WEIGHT)
        w   /= w.sum()
        rows.append(w)
    return pd.DataFrame(rows, index=rebalance_dates, columns=tickers)


# ═════════════════════════════════════════════════════════════════════════════
# PERFORMANCE METRICS
# ═════════════════════════════════════════════════════════════════════════════
def compute_metrics(port_rets: pd.Series,
                    bench_rets: pd.Series,
                    rf_annual: float = RF_ANNUAL) -> dict:
    """Compute full metric set for a return series."""
    if len(port_rets) < 5:
        return {}

    rf_daily   = rf_annual / 252
    excess     = port_rets - rf_daily
    ann_factor = 252

    ann_ret    = float(port_rets.mean() * ann_factor)
    ann_vol    = float(port_rets.std() * np.sqrt(ann_factor))
    sharpe     = float(excess.mean() / (port_rets.std() + 1e-8) * np.sqrt(ann_factor))

    # Sortino
    downside   = port_rets[port_rets < rf_daily]
    sortino    = float(excess.mean() / (downside.std() + 1e-8) * np.sqrt(ann_factor))

    # Max drawdown
    cum        = np.exp(port_rets.cumsum())
    roll_max   = cum.cummax()
    dd         = (cum - roll_max) / roll_max
    max_dd     = float(dd.min())

    # Calmar
    calmar     = ann_ret / (abs(max_dd) + 1e-8)

    # Information ratio vs benchmark
    # Align on common dates
    common     = port_rets.index.intersection(bench_rets.index)
    if len(common) > 5:
        active = port_rets.loc[common] - bench_rets.loc[common]
        ir     = float(active.mean() / (active.std() + 1e-8) * np.sqrt(ann_factor))
        alpha  = float(active.mean() * ann_factor)
    else:
        ir     = np.nan
        alpha  = np.nan

    # Win rate
    win_rate   = float((port_rets > 0).mean())

    # Cumulative return
    cum_ret    = float(np.exp(port_rets.sum()) - 1)

    return {
        "ann_return":   round(ann_ret,   4),
        "ann_vol":      round(ann_vol,   4),
        "sharpe":       round(sharpe,    4),
        "sortino":      round(sortino,   4),
        "max_drawdown": round(max_dd,    4),
        "calmar":       round(calmar,    4),
        "info_ratio":   round(ir,        4),
        "alpha":        round(alpha,     4),
        "win_rate":     round(win_rate,  4),
        "cum_return":   round(cum_ret,   4),
    }


# ═════════════════════════════════════════════════════════════════════════════
# WALK-FORWARD RUNNER
# ═════════════════════════════════════════════════════════════════════════════
def run_walk_forward(prices: pd.DataFrame,
                     nifty: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all benchmarks across 5 folds + blind holdout."""

    rf_daily      = RF_ANNUAL / 252
    daily_returns = compute_returns(prices)

    # Nifty 50 daily returns
    if nifty is not None:
        nifty_rets = compute_returns(nifty.to_frame()).iloc[:, 0]
    else:
        # Fallback: equal-weight proxy
        nifty_rets = daily_returns.mean(axis=1)
        nifty_rets.name = "NIFTY50_proxy"

    tickers   = prices.columns.tolist()
    all_folds = FOLDS + [(FOLDS[-1][0], BLIND_START, BLIND_END)]
    fold_names = [f"Fold{i+1}" for i in range(len(FOLDS))] + ["Blind2025"]

    strategies = {
        "EW":       "Equal Weight",
        "BH":       "Buy & Hold",
        "MinVar":   "Min Variance",
        "MaxSharpe":"Max Sharpe (MVO)",
        "RiskPar":  "Risk Parity",
        "Momentum": "Momentum (12-1)",
        "InvVol":   "Inverse Volatility",
    }

    all_metrics  = []
    all_ret_series = {}  # strategy → combined daily returns

    print(f"\n  Running {len(strategies)} strategies × {len(all_folds)} periods")
    print(f"  {'─'*70}")

    for strat_key in strategies:
        all_ret_series[strat_key] = pd.Series(dtype=float)

    for fold_idx, (train_end, test_start, test_end) in enumerate(all_folds):
        fold_name = fold_names[fold_idx]
        is_blind  = fold_name == "Blind2025"
        tag       = "BLIND" if is_blind else f"Fold {fold_idx+1}"

        # Training data
        train_mask  = (daily_returns.index >= TRAIN_START) & \
                      (daily_returns.index <= train_end)
        train_rets  = daily_returns.loc[train_mask]
        train_prices= prices.loc[prices.index <= train_end]

        # Test period rebalance dates
        test_idx    = prices.index[
            (prices.index >= test_start) & (prices.index <= test_end)
        ]
        if len(test_idx) == 0:
            continue

        reb_dates   = get_rebalance_dates(prices.index, test_start, test_end)
        if len(reb_dates) == 0:
            reb_dates = pd.DatetimeIndex([test_idx[0]])

        # Nifty benchmark for this period
        bench_mask  = (nifty_rets.index >= test_start) & \
                      (nifty_rets.index <= test_end)
        bench_period= nifty_rets.loc[bench_mask]

        print(f"\n  [{tag}]  train→{train_end}  |  test {test_start}→{test_end}"
              f"  |  {len(reb_dates)} rebalances")

        for strat_key, strat_name in strategies.items():
            # ── Build weights schedule ───────────────────────────────────────
            if strat_key == "EW":
                w_df = strategy_equal_weight(tickers, reb_dates)
            elif strat_key == "BH":
                w_df = strategy_buy_hold(prices, test_start, reb_dates)
            elif strat_key == "MinVar":
                w_df = strategy_min_variance(train_rets, reb_dates)
            elif strat_key == "MaxSharpe":
                w_df = strategy_max_sharpe(train_rets, reb_dates, rf_daily)
            elif strat_key == "RiskPar":
                w_df = strategy_risk_parity(train_rets, reb_dates)
            elif strat_key == "Momentum":
                w_df = strategy_momentum(train_prices, reb_dates)
            elif strat_key == "InvVol":
                w_df = strategy_inverse_vol(train_rets, reb_dates)
            else:
                continue

            # ── Compute portfolio returns ────────────────────────────────────
            test_rets = daily_returns.loc[
                (daily_returns.index >= test_start) &
                (daily_returns.index <= test_end)
            ]
            p_rets = portfolio_returns(w_df, test_rets)

            if p_rets.empty:
                continue

            # ── Metrics ──────────────────────────────────────────────────────
            m = compute_metrics(p_rets, bench_period)
            m.update({
                "fold":     fold_name,
                "strategy": strat_key,
                "name":     strat_name,
                "n_days":   len(p_rets),
            })
            all_metrics.append(m)

            # Accumulate return series
            all_ret_series[strat_key] = pd.concat(
                [all_ret_series[strat_key], p_rets]
            )

            sr  = m.get("sharpe",     "—")
            ir  = m.get("info_ratio", "—")
            ret = m.get("ann_return", "—")
            print(f"    {strat_key:<12}  ret={ret:>7.3f}  "
                  f"SR={sr:>6.3f}  IR={ir:>6.3f}")

    # ── Nifty 50 full-period metrics ─────────────────────────────────────────
    nifty_wf = nifty_rets.loc[
        (nifty_rets.index >= "2021-01-01") &
        (nifty_rets.index <= "2023-12-31")
    ]
    nifty_m  = compute_metrics(nifty_wf, nifty_wf)
    nifty_m.update({
        "fold": "WF_Combined", "strategy": "NIFTY50",
        "name": "Nifty 50 Index", "n_days": len(nifty_wf)
    })
    all_metrics.append(nifty_m)

    # ── Build results DataFrames ──────────────────────────────────────────────
    results_df = pd.DataFrame(all_metrics)

    # Summary: mean ± std across walk-forward folds only (exclude blind)
    wf_only = results_df[results_df["fold"].str.startswith("Fold")]
    summary_rows = []
    for strat_key in list(strategies.keys()) + ["NIFTY50"]:
        sub = wf_only[wf_only["strategy"] == strat_key]
        if sub.empty:
            continue
        row = {"strategy": strat_key}
        for col in ["ann_return","ann_vol","sharpe","sortino",
                    "max_drawdown","calmar","info_ratio","alpha","win_rate"]:
            if col in sub.columns:
                row[f"{col}_mean"] = round(sub[col].mean(), 4)
                row[f"{col}_std"]  = round(sub[col].std(),  4)
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    # ── Save returns series ───────────────────────────────────────────────────
    ret_df = pd.DataFrame(all_ret_series)
    ret_df.to_parquet(OUT_DIR / "benchmark_returns.parquet")

    nifty_rets.to_csv(OUT_DIR / "nifty50_returns.csv", header=True)

    return results_df, summary_df


# ═════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═════════════════════════════════════════════════════════════════════════════
def plot_cumulative_wealth(ret_df: pd.DataFrame,
                            nifty_rets: pd.Series):
    """Publication-quality cumulative wealth curves."""
    # Filter to walk-forward period only
    wf_mask = (ret_df.index >= "2021-01-01") & (ret_df.index <= "2023-12-31")
    r       = ret_df.loc[wf_mask]
    n_mask  = (nifty_rets.index >= "2021-01-01") & (nifty_rets.index <= "2023-12-31")
    n       = nifty_rets.loc[n_mask]

    colors = {
        "EW":        "#4878d0",
        "BH":        "#6acc65",
        "MinVar":    "#d65f5f",
        "MaxSharpe": "#ee854a",
        "RiskPar":   "#956cb4",
        "Momentum":  "#8c613c",
        "InvVol":    "#dc7ec0",
    }

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f8f8")

    # Plot Nifty 50 first as thick black reference
    cum_n = np.exp(n.cumsum()) * 100
    ax.plot(cum_n.index, cum_n.values,
            color="black", linewidth=2.5, linestyle="--",
            label="Nifty 50 (benchmark)", zorder=10)

    # Plot each strategy
    for strat, color in colors.items():
        if strat not in r.columns:
            continue
        cum = np.exp(r[strat].fillna(0).cumsum()) * 100
        ax.plot(cum.index, cum.values, color=color,
                linewidth=1.5, label=strat, alpha=0.85)

    ax.set_ylabel("Portfolio Value (base = 100)", fontsize=11)
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9,
              ncol=2, columnspacing=1.0)
    ax.grid(True, alpha=0.4, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    out = FIG_DIR / "cumulative_wealth_benchmarks.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"\n  ✓  Figure saved: {out}")


def plot_metrics_heatmap(summary_df: pd.DataFrame):
    """Heatmap of mean metrics across strategies."""
    cols  = ["ann_return_mean","ann_vol_mean","sharpe_mean",
             "sortino_mean","max_drawdown_mean","info_ratio_mean"]
    labels= ["Ann. Return","Ann. Vol","Sharpe","Sortino","Max DD","Info Ratio"]
    avail = [c for c in cols if c in summary_df.columns]
    avail_labels = [labels[cols.index(c)] for c in avail]

    plot_df = summary_df.set_index("strategy")[avail].copy()

    fig, ax = plt.subplots(figsize=(10, 5))
    data    = plot_df.values.astype(float)

    # Normalise each column for colour mapping
    data_norm = np.zeros_like(data)
    for j in range(data.shape[1]):
        col_vals = data[:, j]
        mn, mx   = col_vals.min(), col_vals.max()
        if mx > mn:
            data_norm[:, j] = (col_vals - mn) / (mx - mn)
        else:
            data_norm[:, j] = 0.5

    # Max drawdown: flip so lower = better
    dd_col = avail.index("max_drawdown_mean") if "max_drawdown_mean" in avail else -1
    if dd_col >= 0:
        data_norm[:, dd_col] = 1 - data_norm[:, dd_col]

    im = ax.imshow(data_norm.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(plot_df.index)))
    ax.set_xticklabels(plot_df.index, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(avail_labels)))
    ax.set_yticklabels(avail_labels, fontsize=10)

    # Annotate cells
    for i in range(len(plot_df.index)):
        for j, c in enumerate(avail):
            val = data[i, j]
            txt = f"{val:.3f}"
            ax.text(i, j, txt, ha="center", va="center", fontsize=8.5,
                    color="black" if 0.2 < data_norm[i, j] < 0.8 else "black")

    plt.colorbar(im, ax=ax, shrink=0.6, label="Relative rank (green = better)")
    plt.tight_layout()

    out = FIG_DIR / "benchmark_metrics_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Figure saved: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  NSE Portfolio GNN — Phase 2: Benchmark Strategies")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n── Loading prices ──")
    prices = load_prices()
    print(f"  ✓  {len(prices.columns)} stocks  |  "
          f"{len(prices):,} days  |  "
          f"{prices.index.min().date()} → {prices.index.max().date()}")

    print("\n── Fetching Nifty 50 index ──")
    nifty = fetch_nifty50(prices.index.min().strftime("%Y-%m-%d"),
                           prices.index.max().strftime("%Y-%m-%d"))

    # Align nifty to same trading calendar
    if nifty is not None:
        nifty = nifty.reindex(prices.index).ffill()

    # ── Run walk-forward ──────────────────────────────────────────────────────
    print("\n── Walk-forward evaluation ──")
    results_df, summary_df = run_walk_forward(prices, nifty if nifty is not None
                                               else compute_returns(prices).mean(axis=1))

    # ── Save results ──────────────────────────────────────────────────────────
    results_df.to_csv(OUT_DIR / "benchmark_fold_results.csv", index=False)
    summary_df.to_csv(OUT_DIR / "benchmark_summary.csv", index=False)
    print(f"\n  ✓  Results saved to {OUT_DIR}")

    # ── Print summary table ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY — Mean across 9 walk-forward folds")
    print("=" * 70)
    cols_show = ["strategy","ann_return_mean","ann_vol_mean",
                 "sharpe_mean","info_ratio_mean","max_drawdown_mean"]
    avail = [c for c in cols_show if c in summary_df.columns]
    print(summary_df[avail].to_string(index=False))

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\n── Generating figures ──")
    ret_df = pd.read_parquet(OUT_DIR / "benchmark_returns.parquet")
    nifty_rets = pd.read_csv(OUT_DIR / "nifty50_returns.csv",
                              index_col=0, parse_dates=True).iloc[:, 0]

    plot_cumulative_wealth(ret_df, nifty_rets)
    plot_metrics_heatmap(summary_df)

    print("\n" + "=" * 70)
    print("  Phase 2 complete.")
    print(f"  Results : {OUT_DIR}")
    print(f"  Figures : {FIG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
