"""
training/train_stgat.py
========================
Walk-forward training and evaluation of STGAT-DPO.

Folds (expanding window):
  Fold 1: train 2018-01-01→2020-12-31, test 2021-01-01→2021-12-31
  Fold 2: train 2018-01-01→2021-06-30, test 2021-07-01→2022-06-30
  Fold 3: train 2018-01-01→2021-12-31, test 2022-01-01→2022-12-31
  Fold 4: train 2018-01-01→2022-06-30, test 2022-07-01→2023-06-30
  Fold 5: train 2018-01-01→2022-12-31, test 2023-01-01→2023-12-31
  Blind:  model frozen, test 2025-01-01→2026-03-30

Run:
    python training/train_stgat.py
"""

import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from copy import deepcopy

# Local imports
from utils.features   import build_feature_tensor, get_close_returns, N_FEATURES
from models.gnn.stgat_dpo import (
    STGATDPOModel, HMMRegimeDetector,
    build_sector_adj, build_dynamic_adj,
    combined_loss, information_ratio_loss, REGIME_LAMBDA
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"  / "raw"
OUT_DIR     = ROOT / "outputs" / "results"
FIG_DIR     = ROOT / "outputs" / "figures"
CKPT_DIR    = ROOT / "outputs" / "checkpoints"
for d in [OUT_DIR, FIG_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
PRICES_FILE   = DATA_DIR / "prices_51.parquet"
LOOKBACK      = 20       # trading days for LSTM window
CORR_WINDOW   = 63       # days for rolling correlation graph
BATCH_SIZE    = 32
MAX_EPOCHS    = 150
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
PATIENCE      = 20       # early stopping
GRAD_CLIP     = 1.0
RF_DAILY      = 0.065 / 252
MAX_WEIGHT    = 0.15
SEED          = 42
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FOLDS = [
    ("2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31"),
    ("2018-01-01", "2021-06-30", "2021-07-01", "2022-06-30"),
    ("2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31"),
    ("2018-01-01", "2022-06-30", "2022-07-01", "2023-06-30"),
    ("2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2018-01-01", "2023-06-30", "2023-07-01", "2024-06-30"),
    ("2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2018-01-01", "2024-06-30", "2024-07-01", "2025-06-30"),
    ("2018-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
]
BLIND = ("2018-01-01", "2025-12-31", "2026-01-01", "2026-08-28")

torch.manual_seed(SEED)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
torch.use_deterministic_algorithms(True, warn_only=False)
import random, numpy as np
random.seed(SEED)
np.random.seed(SEED)
np.random.seed(SEED)


# ═════════════════════════════════════════════════════════════════════════════
# DATASET
# ═════════════════════════════════════════════════════════════════════════════
class STGATDataset(Dataset):
    """
    Sliding-window dataset for STGAT training.

    Each sample:
      x         : [N, T, F]   feature window
      adj       : [N, N]      adjacency at the last day of window
      ret_target: [N]         actual returns on day t+1 (supervision)
      bench_ret : float       Nifty 50 return on day t+1
      regime    : int         HMM state on day t
    """
    def __init__(self,
                 feat_arr:     np.ndarray,     # [T_total, N, F]
                 ret_arr:      np.ndarray,     # [T_total-1, N]  (1-day lag)
                 nifty_rets:   np.ndarray,     # [T_total-1]
                 regimes:      np.ndarray,     # [T_total]
                 sector_adj:   np.ndarray,     # [N, N]
                 tickers:      list,
                 lookback:     int = LOOKBACK,
                 corr_window:  int = CORR_WINDOW,
                 eligible_mask: np.ndarray = None):

        self.feat_arr    = feat_arr
        self.ret_arr     = ret_arr
        self.nifty_rets  = nifty_rets
        self.regimes     = regimes
        self.sector_adj  = sector_adj
        self.tickers     = tickers
        self.lookback    = lookback
        self.corr_window = corr_window
        self.eligible_mask = eligible_mask

        # Precompute adjacency matrices for each valid timestep
        T = feat_arr.shape[0]
        self.valid_idx = []
        for t in range(max(lookback, corr_window), T - 1):
            self.valid_idx.append(t)

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        t = self.valid_idx[idx]

        # Feature window: [N, T, F] → [N, lookback, F]
        x = self.feat_arr[t - self.lookback:t]     # [T, N, F]
        x = torch.tensor(x, dtype=torch.float32).permute(1, 0, 2)  # [N, T, F]

        # Adjacency matrix on day t
        ret_window = self.ret_arr[
            max(0, t - self.corr_window):t
        ]                                           # [corr_window, N]
        regime = int(self.regimes[t])
        mask_t = self.eligible_mask[t] if self.eligible_mask is not None else None
        adj = build_dynamic_adj(ret_window, regime, self.sector_adj, eligible_mask=mask_t)
        adj = torch.tensor(adj, dtype=torch.float32)  # [N, N]

        # Target: actual returns on day t+1
        ret_target = torch.tensor(
            self.ret_arr[t], dtype=torch.float32)   # [N]

        # Nifty 50 return on day t+1
        bench_ret = float(self.nifty_rets[t])

        return x, adj, ret_target, bench_ret, regime


def collate_fn(batch):
    xs, adjs, rets, bench, regimes = zip(*batch)
    return (torch.stack(xs),
            torch.stack(adjs),
            torch.stack(rets),
            torch.tensor(bench, dtype=torch.float32),
            list(regimes))


# ═════════════════════════════════════════════════════════════════════════════
# TRAINER
# ═════════════════════════════════════════════════════════════════════════════
class STGATTrainer:
    def __init__(self, n_stocks: int, n_features: int = N_FEATURES):
        self.n_stocks  = n_stocks
        self.model     = STGATDPOModel(
            n_stocks    = n_stocks,
            n_features  = n_features,
            lookback    = LOOKBACK,
            lstm_hidden = 64,
            lstm_layers = 2,
            gat_hidden  = 64,
            gat_heads   = 2,
            dropout     = 0.2,
            max_weight  = MAX_WEIGHT,
        ).to(DEVICE)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=LR, weight_decay=WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=MAX_EPOCHS, eta_min=1e-5
        )
        print(f"  Model parameters: {self.model.count_parameters():,}")
        print(f"  Device: {DEVICE}")

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0

        for x, adj, ret_target, bench_ret, regimes in loader:
            x          = x.to(DEVICE)
            adj        = adj.to(DEVICE)
            ret_target = ret_target.to(DEVICE)
            bench_ret  = bench_ret.to(DEVICE)

            # Use mode regime for the batch
            regime_mode = max(set(regimes), key=regimes.count)
            lam = REGIME_LAMBDA[regime_mode]

            w, mu, cov = self.model(x, adj, regime_lambda=lam)

            # Realized portfolio return: w · actual_returns (per sample in batch)
            port_rets   = (w * ret_target).sum(dim=-1)     # [B]
            loss = combined_loss(port_rets, bench_ret,
                                  mu, ret_target,
                                  alpha=0.7, beta=0.3)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches  += 1

        self.scheduler.step()
        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches  = 0

        for x, adj, ret_target, bench_ret, regimes in loader:
            x          = x.to(DEVICE)
            adj        = adj.to(DEVICE)
            ret_target = ret_target.to(DEVICE)
            bench_ret  = bench_ret.to(DEVICE)

            regime_mode = max(set(regimes), key=regimes.count)
            lam = REGIME_LAMBDA[regime_mode]

            w, mu, cov = self.model(x, adj, regime_lambda=lam)
            port_rets  = (w * ret_target).sum(dim=-1)
            loss = information_ratio_loss(port_rets, bench_ret)

            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    def train(self,
              train_loader: DataLoader,
              val_loader:   DataLoader,
              fold_name:    str) -> dict:
        """Train with early stopping. Returns training history."""
        best_val  = float("inf")
        best_state= None
        no_improve= 0
        history   = {"train_loss": [], "val_loss": []}

        print(f"\n  Training {fold_name}  "
              f"({len(train_loader.dataset)} train  |  "
              f"{len(val_loader.dataset)} val)")
        print(f"  {'Epoch':>6}  {'Train':>8}  {'Val':>8}  {'Best':>8}")
        print(f"  {'─'*38}")

        for epoch in range(1, MAX_EPOCHS + 1):
            tr_loss  = self.train_epoch(train_loader)
            val_loss = self.eval_epoch(val_loader)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val   = val_loss
                best_state = deepcopy(self.model.state_dict())
                no_improve = 0
                tag        = "★"
            else:
                no_improve += 1
                tag         = ""

            if epoch % 10 == 0 or epoch == 1:
                print(f"  {epoch:>6}  {tr_loss:>8.4f}  "
                      f"{val_loss:>8.4f}  {best_val:>8.4f}  {tag}")

            if no_improve >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    def save(self, path: Path):
        torch.save(self.model.state_dict(), path)
        print(f"  ✓  Checkpoint saved: {path}")

    def load(self, path: Path):
        self.model.load_state_dict(
            torch.load(path, map_location=DEVICE))
        print(f"  ✓  Checkpoint loaded: {path}")


# ═════════════════════════════════════════════════════════════════════════════
# PORTFOLIO EVALUATION
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_portfolio(model:       STGATDPOModel,
                        feat_arr:    np.ndarray,
                        ret_arr:     np.ndarray,
                        nifty_rets:  np.ndarray,
                        regimes:     np.ndarray,
                        sector_adj:  np.ndarray,
                        dates:       pd.DatetimeIndex,
                        test_start:  str,
                        test_end:    str,
                        tc_bps:      float = 10.0
                        ) -> tuple[pd.Series, pd.DataFrame]:
    """
    Monthly rebalancing evaluation on the test period.
    Returns (daily portfolio return series, weights DataFrame).
    """
    model.eval()

    # Get rebalancing dates (first trading day of each month)
    test_mask = (dates >= test_start) & (dates <= test_end)
    test_dates = dates[test_mask]
    if len(test_dates) == 0:
        return pd.Series(dtype=float), pd.DataFrame()

    # Monthly rebalance dates
    reb_dates = (test_dates.to_series()
                 .groupby([test_dates.year, test_dates.month])
                 .first().values)
    reb_dates = pd.DatetimeIndex(reb_dates)

    all_weights  = {}
    port_ret_map = {}
    prev_w       = None
    N            = feat_arr.shape[1]
    tc           = tc_bps / 10_000

    for i, reb_d in enumerate(reb_dates):
        # Find index of rebalancing date
        t_idx_arr = np.where(dates == reb_d)[0]
        if len(t_idx_arr) == 0:
            continue
        t = int(t_idx_arr[0])

        if t < max(LOOKBACK, CORR_WINDOW):
            continue

        # Build features window
        x = feat_arr[t - LOOKBACK:t]                    # [T, N, F]
        x = torch.tensor(x, dtype=torch.float32
                         ).permute(1, 0, 2).unsqueeze(0).to(DEVICE)  # [1, N, T, F]

        # Build adjacency
        ret_window = ret_arr[max(0, t - CORR_WINDOW):t]
        regime     = int(regimes[t])
        lam        = REGIME_LAMBDA[regime]
        adj        = build_dynamic_adj(ret_window, regime, sector_adj)
        adj_t      = torch.tensor(adj, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # Model inference
        w, _, _ = model(x, adj_t, regime_lambda=lam)
        w_np     = w.squeeze(0).cpu().numpy()

        # Transaction cost
        if prev_w is not None:
            cost = np.abs(w_np - prev_w).sum() * tc
        else:
            cost = w_np.sum() * tc   # initial purchase

        all_weights[reb_d] = w_np
        prev_w = w_np.copy()

        # Holding period: current reb_date to next reb_date (or test_end)
        if i + 1 < len(reb_dates):
            next_reb = reb_dates[i + 1]
        else:
            next_reb = test_dates[-1] + pd.Timedelta(days=1)

        hold_mask = (dates >= reb_d) & (dates < next_reb)
        hold_dates = dates[hold_mask]

        for d_idx, d in enumerate(hold_dates):
            t_d = np.where(dates == d)[0]
            if len(t_d) == 0:
                continue
            ti = int(t_d[0])
            if ti >= len(ret_arr):
                continue
            r   = float(ret_arr[ti] @ w_np)
            # Subtract transaction cost on first day of holding period
            if d_idx == 0:
                r -= cost
            port_ret_map[d] = r

    if not port_ret_map:
        return pd.Series(dtype=float), pd.DataFrame()

    port_rets  = pd.Series(port_ret_map).sort_index()
    weights_df = pd.DataFrame(all_weights).T

    return port_rets, weights_df


def compute_metrics(port_rets: pd.Series,
                    nifty_ser: pd.Series) -> dict:
    """Compute full metric set."""
    if len(port_rets) < 5:
        return {}

    ann = 252
    rf  = RF_DAILY
    excess  = port_rets - rf
    ann_ret = float(port_rets.mean() * ann)
    ann_vol = float(port_rets.std() * ann ** 0.5)
    sharpe  = float(excess.mean() / (port_rets.std() + 1e-8) * ann ** 0.5)

    down    = port_rets[port_rets < rf]
    sortino = float(excess.mean() / (down.std() + 1e-8) * ann ** 0.5)

    cum     = np.exp(port_rets.cumsum())
    max_dd  = float(((cum - cum.cummax()) / cum.cummax()).min())
    calmar  = ann_ret / (abs(max_dd) + 1e-8)

    common  = port_rets.index.intersection(nifty_ser.index)
    if len(common) > 5:
        active = port_rets.loc[common] - nifty_ser.loc[common]
        ir     = float(active.mean() / (active.std() + 1e-8) * ann ** 0.5)
        alpha  = float(active.mean() * ann)
    else:
        ir, alpha = float("nan"), float("nan")

    cum_ret  = float(np.exp(port_rets.sum()) - 1)
    win_rate = float((port_rets > 0).mean())

    return dict(
        ann_return   = round(ann_ret, 4),
        ann_vol      = round(ann_vol, 4),
        sharpe       = round(sharpe,  4),
        sortino      = round(sortino, 4),
        max_drawdown = round(max_dd,  4),
        calmar       = round(calmar,  4),
        info_ratio   = round(ir,      4),
        alpha        = round(alpha,   4),
        win_rate     = round(win_rate, 4),
        cum_return   = round(cum_ret,  4),
    )


# ═════════════════════════════════════════════════════════════════════════════
# PLOT UTILITIES
# ═════════════════════════════════════════════════════════════════════════════
def plot_results(stgat_rets: dict, bench_rets_csv: Path, fig_dir: Path):
    """Cumulative wealth + attention weight heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick

    nifty = pd.read_csv(bench_rets_csv, index_col=0, parse_dates=True).iloc[:, 0]
    bench_returns = pd.read_parquet(
        bench_rets_csv.parent / "benchmark_returns.parquet")

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#f8f8f8")

    # Nifty 50
    wf_start, wf_end = "2021-01-01", "2023-12-31"
    nm = nifty.loc[wf_start:wf_end]
    ax.plot(np.exp(nm.cumsum()) * 100,
            color="black", lw=2.5, ls="--", label="Nifty 50", zorder=10)

    # Benchmarks (muted)
    for col, color, lw in [
        ("EW",      "#4878d0", 1.2),
        ("MinVar",  "#d65f5f", 1.2),
        ("Momentum","#956cb4", 1.2),
    ]:
        if col in bench_returns.columns:
            s = bench_returns[col].loc[wf_start:wf_end].fillna(0)
            ax.plot(np.exp(s.cumsum()) * 100,
                    color=color, lw=lw, alpha=0.5, label=col)

    # STGAT-DPO
    if stgat_rets:
        combined = pd.concat(stgat_rets.values()).sort_index()
        combined = combined.loc[wf_start:wf_end]
        ax.plot(np.exp(combined.cumsum()) * 100,
                color="#e74c3c", lw=2.5, label="STGAT-DPO (ours)", zorder=9)

    ax.set_ylabel("Portfolio Value (base = 100)", fontsize=11)
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.0f"))
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    ax.grid(True, alpha=0.35, ls="--")
    ax.spines[["top", "right"]].set_visible(False)

    out = fig_dir / "cumulative_wealth_stgat.png"
    plt.tight_layout()
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓  Figure: {out}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN — WALK-FORWARD
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  NSE Portfolio GNN — Phase 3: STGAT-DPO Walk-Forward Training")
    print("=" * 70)

    # ── Load & compute features ───────────────────────────────────────────
    print("\n── Building feature tensor ──")
    feat_arr, tickers, dates = build_feature_tensor(PRICES_FILE, verbose=True)
    N   = len(tickers)
    T   = len(dates)
    print(f"  Tickers : {N}")
    print(f"  Days    : {T}  ({dates[0].date()} → {dates[-1].date()})")

    # ── Close returns (1-day) ─────────────────────────────────────────────
    print("\n── Computing return matrix ──")
    # Get aligned returns from parquet (1-step forward)
    df_raw = pd.read_parquet(PRICES_FILE)
    close_df = pd.DataFrame(
        {t: df_raw[(t, t, "Close")].squeeze() for t in tickers}
    ).reindex(dates).ffill(limit=5)
    ret_df   = np.log(close_df / close_df.shift(1)).iloc[1:]   # [T-1, N]
    ret_arr  = ret_df.fillna(0).values.astype(np.float32)
    # Point-in-time eligibility mask: True once a stock has CORR_WINDOW (63)
    # real (non-pre-listing) trading days of history, avoiding contamination
    # of the correlation graph by pre-listing zero-filled returns.
    first_valid_idx = close_df.notna().values.argmax(axis=0)
    never_valid     = ~close_df.notna().values.any(axis=0)
    eligible_from   = first_valid_idx + 63
    eligible_from[never_valid] = 10**9
    idx_grid        = np.arange(T)[:, None]
    eligible_mask_full = idx_grid >= eligible_from[None, :]

    # Nifty 50 returns
    nifty_csv = OUT_DIR / "nifty50_returns.csv"
    if nifty_csv.exists():
        nifty_ser = pd.read_csv(nifty_csv, index_col=0,
                                 parse_dates=True).iloc[:, 0]
        nifty_ser = nifty_ser.reindex(dates[1:]).ffill().fillna(0)
        nifty_arr = nifty_ser.values.astype(np.float32)
    else:
        print("  ⚠  nifty50_returns.csv not found — using EW proxy")
        nifty_arr = ret_arr.mean(axis=1)
        nifty_ser = pd.Series(nifty_arr, index=dates[1:])

    # Sector adjacency (static, precomputed once)
    sector_adj = build_sector_adj(tickers)

    # ── Walk-forward loop ─────────────────────────────────────────────────
    all_fold_metrics = []
    all_stgat_rets   = {}

    folds_to_run = FOLDS + [BLIND[:4]]
    fold_names   = [f"Fold{i+1}" for i in range(len(FOLDS))] + ["Blind2025"]

    for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(folds_to_run):
        fold_name = fold_names[fold_idx]
        is_blind  = fold_name == "Blind2025"

        print(f"\n{'='*70}")
        print(f"  {fold_name}  |  train→{tr_end}  |  test {te_start}→{te_end}")
        print(f"{'='*70}")

        # ── Date indices ──────────────────────────────────────────────────
        tr_mask = (dates >= tr_start) & (dates <= tr_end)
        te_mask = (dates >= te_start) & (dates <= te_end)

        tr_dates = dates[tr_mask]
        te_dates = dates[te_mask]

        if len(tr_dates) < 300:
            print(f"  ⚠  Skipping — insufficient training data")
            continue

        # ── Align ret_arr with feat_arr (feat_arr starts 1 day later) ────
        # feat_arr[i] corresponds to dates[i]
        # ret_arr[i]  is the RETURN ON dates[i] (from dates[i-1])
        # Alignment: for sample at t, target is ret_arr[t]
        # We need ret_arr indices matching dates
        # ret_arr has T-1 rows, corresponding to dates[1:]
        dates_ret    = dates[1:]      # dates for return rows
        tr_ret_mask  = (dates_ret >= tr_start) & (dates_ret <= tr_end)
        te_ret_mask  = (dates_ret >= te_start) & (dates_ret <= te_end)

        tr_ret_arr   = ret_arr[tr_ret_mask]
        nifty_tr     = nifty_arr[tr_ret_mask]

        # Regime: fit HMM on training Nifty returns
        print("\n── Fitting HMM ──")
        hmm_det = HMMRegimeDetector(n_states=3, seed=SEED)
        hmm_det.fit(nifty_tr)
        regimes_full = hmm_det.predict_causal(nifty_arr)   # [T-1] regime per day; causal (no look-ahead)
        regimes_full_aligned = np.zeros(T, dtype=np.int32)
        regimes_full_aligned[1:] = regimes_full
        regime_counts = np.bincount(regimes_full, minlength=3)
        labels = ["bear", "sideways", "bull"]
        print(f"  Regime distribution: "
              + "  ".join(f"{labels[i]}={regime_counts[i]}" for i in range(3)))

        # ── Build datasets ────────────────────────────────────────────────
        # Training set: all samples within training period
        # For dataset we need indices into feat_arr
        # A sample at position t uses feat_arr[t-lookback:t]
        # and ret_arr[t] as target (needs dates_ret[t] in [tr_start, tr_end])

        # Convert date masks to feat_arr index masks
        # feat_arr index t corresponds to dates[t]
        # ret_arr index t corresponds to dates_ret[t] = dates[t+1]
        # ... so feat_arr[t] target is ret_arr[t] (dates[t+1])
        # We use: for t in [lookback, T-1], if dates[t+1] in train_range

        tr_feat_arr   = feat_arr
        tr_regimes_fa = regimes_full_aligned

        # Compute valid training sample indices
        tr_start_ts = pd.Timestamp(tr_start)
        tr_end_ts   = pd.Timestamp(tr_end)
        tr_sample_dates_mask = np.array([
            (dates[t] >= tr_start_ts) and (dates[t] <= tr_end_ts)
            for t in range(T)
        ])

        # Build ret_arr indexed by dates (not dates_ret)
        # ret_for_feat[t] = log-return on day dates[t] (if t>=1)
        ret_for_feat = np.zeros((T, N), dtype=np.float32)
        ret_for_feat[1:] = ret_arr    # ret_arr[0] = return on dates[1], etc.

        nifty_for_feat = np.zeros(T, dtype=np.float32)
        nifty_for_feat[1:] = nifty_arr

        # Build train/val datasets
        train_dataset = STGATDataset(
            feat_arr    = feat_arr,
            ret_arr     = ret_for_feat,
            nifty_rets  = nifty_for_feat,
            regimes     = tr_regimes_fa,
            sector_adj  = sector_adj,
            tickers     = tickers,
            lookback    = LOOKBACK,
            corr_window = 63,
            eligible_mask = eligible_mask_full,
        )

        # Filter to only training period samples
        train_indices = [i for i, t in enumerate(train_dataset.valid_idx)
                         if tr_sample_dates_mask[t]]
        val_split     = int(0.85 * len(train_indices))
        val_indices   = train_indices[val_split:]
        train_indices = train_indices[:val_split]

        from torch.utils.data import Subset
        train_sub = Subset(train_dataset, train_indices)
        val_sub   = Subset(train_dataset, val_indices)

        print(f"\n  Train samples: {len(train_sub)}"
              f"  |  Val samples: {len(val_sub)}")

        if len(train_sub) < 50:
            print("  ⚠  Too few training samples — skipping")
            continue

        train_loader = DataLoader(train_sub,   batch_size=BATCH_SIZE,
                                   shuffle=True,  collate_fn=collate_fn,
                                   num_workers=0)
        val_loader   = DataLoader(val_sub,     batch_size=BATCH_SIZE,
                                   shuffle=False, collate_fn=collate_fn,
                                   num_workers=0)

        # ── Train (skip for blind — use Fold 5 model) ─────────────────────
        trainer = STGATTrainer(n_stocks=N, n_features=N_FEATURES)

        if is_blind:
            ckpt_path = CKPT_DIR / "stgat_blind.pt"
            if ckpt_path.exists():
                trainer.load(ckpt_path)
                print(f"  Loaded blind model (trained through 2024-09-30)")
            else:
                print(f"  Training blind model on 2018-01-01 to 2024-09-30")
                trainer.train(train_loader, val_loader, fold_name)
                trainer.save(ckpt_path)
        else:
            ckpt_path = CKPT_DIR / f"stgat_{fold_name.lower()}.pt"
            if ckpt_path.exists():
                print(f"  ✓  Checkpoint found — skipping to evaluation")
                trainer.load(ckpt_path)
            else:
                trainer.train(train_loader, val_loader, fold_name)
                trainer.save(ckpt_path)

        # ── Evaluate on test period ───────────────────────────────────────
        print(f"\n── Evaluating {fold_name} ──")
        port_rets, weights_df = evaluate_portfolio(
            model       = trainer.model,
            feat_arr    = feat_arr,
            ret_arr     = ret_for_feat,
            nifty_rets  = nifty_for_feat,
            regimes     = tr_regimes_fa,
            sector_adj  = sector_adj,
            dates       = dates,
            test_start  = te_start,
            test_end    = te_end,
        )

        if port_rets.empty:
            print(f"  ⚠  No portfolio returns generated")
            continue

        nifty_te = nifty_ser.reindex(port_rets.index).ffill().fillna(0)
        metrics  = compute_metrics(port_rets, nifty_te)
        metrics.update({"fold": fold_name, "strategy": "STGAT-DPO",
                         "n_days": len(port_rets)})

        all_fold_metrics.append(metrics)
        all_stgat_rets[fold_name] = port_rets

        print(f"\n  ── {fold_name} Results ──")
        print(f"  Ann. Return : {metrics.get('ann_return', 'nan'):.4f}")
        print(f"  Sharpe      : {metrics.get('sharpe', 'nan'):.4f}")
        print(f"  Info Ratio  : {metrics.get('info_ratio', 'nan'):.4f}")
        print(f"  Max DD      : {metrics.get('max_drawdown', 'nan'):.4f}")
        print(f"  Cum. Return : {metrics.get('cum_return', 'nan'):.4f}")

        # Save weights
        if not weights_df.empty:
            weights_df.columns = tickers[:weights_df.shape[1]] \
                if weights_df.shape[1] <= N else tickers
            wt_path = OUT_DIR / f"weights_{fold_name}.csv"
            weights_df.to_csv(wt_path)

    # ── Save all results ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  STGAT-DPO SUMMARY")
    print(f"{'='*70}")

    if all_fold_metrics:
        results_df = pd.DataFrame(all_fold_metrics)
        results_df.to_csv(OUT_DIR / "stgat_fold_results.csv", index=False)

        # Mean across WF folds only
        wf = results_df[results_df["fold"].str.startswith("Fold")]
        cols = ["ann_return","ann_vol","sharpe","sortino",
                "max_drawdown","calmar","info_ratio","win_rate"]
        print(f"\n  {'Metric':<20}  {'Mean':>8}  {'Std':>8}")
        print(f"  {'─'*38}")
        for c in cols:
            if c in wf.columns:
                print(f"  {c:<20}  "
                      f"{wf[c].mean():>8.4f}  "
                      f"{wf[c].std():>8.4f}")

        # Save combined returns
        if all_stgat_rets:
            combined = pd.concat(all_stgat_rets.values()).sort_index()
            combined.to_csv(OUT_DIR / "stgat_returns.csv", header=True)
            print(f"\n  ✓  Returns saved: {OUT_DIR / 'stgat_returns.csv'}")

        # Plot
        bench_csv = OUT_DIR / "nifty50_returns.csv"
        if bench_csv.exists():
            try:
                plot_results(all_stgat_rets, bench_csv, FIG_DIR)
            except Exception as e:
                print(f"  ⚠  Plot failed: {e}")

    print(f"\n{'='*70}")
    print("  Phase 3 complete.")
    print(f"  Results    : {OUT_DIR}")
    print(f"  Checkpoints: {CKPT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
