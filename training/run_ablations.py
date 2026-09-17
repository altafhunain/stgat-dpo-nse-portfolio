"""
training/run_ablations.py
==========================
Trains and evaluates two ablation variants of STGAT-DPO:

  STGAT-NoHMM  — HMM regime gating removed.
                 Constant λ=4.0 (neutral), static graph threshold=0.25.
                 Isolates the contribution of HMM regime conditioning.

  STGAT-NoGAT  — GAT attention replaced with simple mean aggregation.
                 Keeps LSTM encoder + portfolio layer identical.
                 Isolates the contribution of graph attention.

Results appended to:
  outputs/results/ablation_results.csv
  outputs/results/ablation_summary.csv

Run:
    python training/run_ablations.py
"""

import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pathlib import Path
from copy import deepcopy

from utils.features   import build_feature_tensor, N_FEATURES
from models.gnn.stgat_dpo import (
    STGATDPOModel, HMMRegimeDetector,
    build_sector_adj, build_dynamic_adj,
    combined_loss, information_ratio_loss,
    REGIME_LAMBDA, TemporalEncoder, PortfolioLayer
)
from training.train_stgat import (
    STGATDataset, collate_fn, evaluate_portfolio,
    compute_metrics, FOLDS, BLIND,
    LOOKBACK, BATCH_SIZE, MAX_EPOCHS, LR, WEIGHT_DECAY,
    PATIENCE, GRAD_CLIP, RF_DAILY, MAX_WEIGHT, SEED,
    DEVICE, PRICES_FILE
)

torch.manual_seed(SEED)
np.random.seed(SEED)

ROOT     = Path(__file__).resolve().parent.parent
OUT_DIR  = ROOT / "outputs" / "results"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
FIG_DIR  = ROOT / "outputs" / "figures"
for d in [OUT_DIR, CKPT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
# ABLATION A — STGAT-NoHMM
# Constant λ, static graph threshold. HMM still fitted but ignored at
# allocation time — we pass regime_lambda=4.0 always and use a fixed
# threshold regardless of regime.
# ═════════════════════════════════════════════════════════════════════════════
CONSTANT_LAMBDA    = 4.0        # neutral/sideways always
STATIC_THRESHOLD   = 0.25       # fixed correlation threshold


def build_static_adj(returns_window: np.ndarray,
                      sector_adj: np.ndarray,
                      threshold: float = STATIC_THRESHOLD,
                      sector_weight: float = 0.3) -> np.ndarray:
    """Static adjacency — no regime conditioning."""
    corr = np.corrcoef(returns_window.T)
    corr = np.nan_to_num(corr, nan=0.0)
    dyn  = (np.abs(corr) > threshold).astype(np.float32)
    np.fill_diagonal(dyn, 0.0)
    dyn  = dyn * np.abs(corr).clip(0, 1)
    adj  = (1 - sector_weight) * dyn + sector_weight * sector_adj
    adj  = adj.clip(0, 1)
    np.fill_diagonal(adj, 0.0)
    return adj.astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# ABLATION B — STGAT-NoGAT
# Replaces GraphAttentionLayer with simple mean neighbour aggregation.
# ═════════════════════════════════════════════════════════════════════════════
class MeanAggregationLayer(nn.Module):
    """
    Drop-in replacement for GraphAttentionLayer.
    Aggregates neighbour embeddings by simple (uniform) mean — no attention.
    Followed by a linear projection + LayerNorm to keep the same output dim.
    """
    def __init__(self, in_dim: int, out_dim: int,
                 n_heads: int = 2, dropout: float = 0.2):
        super().__init__()
        self.proj    = nn.Linear(in_dim, out_dim)
        self.norm    = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        """
        h   : [B, N, in_dim]
        adj : [B, N, N]  or  [N, N]
        """
        B, N, _ = h.shape
        if adj.dim() == 2:
            adj = adj.unsqueeze(0).expand(B, -1, -1)  # [B, N, N]

        # Count neighbours per node (avoid division by zero)
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, N, 1]

        # Mean neighbour aggregation
        # neigh_sum[b,i,:] = sum_j adj[b,i,j] * h[b,j,:]
        neigh_sum = torch.bmm(adj, h)                # [B, N, in_dim]
        neigh_avg = neigh_sum / deg                  # [B, N, in_dim]

        # Combine self + neighbour (residual style)
        out = self.proj(h + neigh_avg)
        out = F.elu(out)
        out = self.norm(out)
        return out


class STGATNoGATModel(STGATDPOModel):
    """
    STGAT-NoGAT: identical to STGATDPOModel but with GAT replaced
    by simple mean aggregation. Inherits all other components unchanged.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override both GAT layers with mean aggregation
        in_dim  = kwargs.get('lstm_hidden', 64)
        out_dim = kwargs.get('gat_hidden',  64)
        heads   = kwargs.get('gat_heads',   2)
        drop    = kwargs.get('dropout',     0.2)

        self.gat1 = MeanAggregationLayer(in_dim,  out_dim, heads, drop)
        self.gat2 = MeanAggregationLayer(out_dim, out_dim, heads, drop)


# ═════════════════════════════════════════════════════════════════════════════
# GENERIC TRAINER (reuses train_stgat logic)
# ═════════════════════════════════════════════════════════════════════════════
class AblationTrainer:
    def __init__(self, n_stocks: int, variant: str):
        """
        variant: 'NoHMM' | 'NoGAT'
        """
        self.variant = variant
        self.n_stocks = n_stocks

        if variant == 'NoGAT':
            self.model = STGATNoGATModel(
                n_stocks    = n_stocks,
                n_features  = N_FEATURES,
                lookback    = LOOKBACK,
                lstm_hidden = 64,
                lstm_layers = 2,
                gat_hidden  = 64,
                gat_heads   = 2,
                dropout     = 0.2,
                max_weight  = MAX_WEIGHT,
            ).to(DEVICE)
        else:
            # NoHMM uses identical architecture — only inference differs
            self.model = STGATDPOModel(
                n_stocks    = n_stocks,
                n_features  = N_FEATURES,
                lookback    = LOOKBACK,
                lstm_hidden = 64,
                lstm_layers = 2,
                gat_hidden  = 64,
                gat_heads   = 2,
                dropout     = 0.2,
                max_weight  = MAX_WEIGHT,
            ).to(DEVICE)

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=MAX_EPOCHS, eta_min=1e-5)
        print(f"  [{variant}] Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches  = 0
        for x, adj, ret_target, bench_ret, regimes in loader:
            x          = x.to(DEVICE)
            ret_target = ret_target.to(DEVICE)
            bench_ret  = bench_ret.to(DEVICE)

            if self.variant == 'NoHMM':
                # Always use static adjacency and constant lambda
                adj_use = build_static_adj(
                    ret_target.cpu().numpy(), # placeholder — adj already built
                    np.zeros((self.n_stocks, self.n_stocks))
                )
                adj_t = torch.tensor(adj_use).unsqueeze(0).expand(
                    x.shape[0], -1, -1).to(DEVICE)
                lam = CONSTANT_LAMBDA
            else:
                adj_t = adj.to(DEVICE)
                regime_mode = max(set(regimes), key=regimes.count)
                lam = REGIME_LAMBDA[regime_mode]

            w, mu, cov = self.model(x, adj_t, regime_lambda=lam)
            port_rets  = (w * ret_target).sum(dim=-1)
            loss = combined_loss(port_rets, bench_ret, mu, ret_target)

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
            ret_target = ret_target.to(DEVICE)
            bench_ret  = bench_ret.to(DEVICE)

            if self.variant == 'NoHMM':
                adj_use = build_static_adj(
                    np.zeros((1, self.n_stocks)),
                    np.zeros((self.n_stocks, self.n_stocks))
                )
                adj_t = torch.tensor(adj_use).unsqueeze(0).expand(
                    x.shape[0], -1, -1).to(DEVICE)
                lam = CONSTANT_LAMBDA
            else:
                adj_t = adj.to(DEVICE)
                regime_mode = max(set(regimes), key=regimes.count)
                lam = REGIME_LAMBDA[regime_mode]

            w, mu, cov = self.model(x, adj_t, regime_lambda=lam)
            port_rets  = (w * ret_target).sum(dim=-1)
            loss = information_ratio_loss(port_rets, bench_ret)
            total_loss += loss.item()
            n_batches  += 1

        return total_loss / max(n_batches, 1)

    def train(self, train_loader, val_loader, fold_name) -> None:
        best_val   = float('inf')
        best_state = None
        no_improve = 0

        print(f"\n  Training {self.variant} — {fold_name}  "
              f"({len(train_loader.dataset)} train | "
              f"{len(val_loader.dataset)} val)")
        print(f"  {'Epoch':>6}  {'Train':>8}  {'Val':>8}  {'Best':>8}")
        print(f"  {'─'*36}")

        for epoch in range(1, MAX_EPOCHS + 1):
            tr  = self.train_epoch(train_loader)
            val = self.eval_epoch(val_loader)

            if val < best_val:
                best_val   = val
                best_state = deepcopy(self.model.state_dict())
                no_improve = 0
                tag        = '★'
            else:
                no_improve += 1
                tag         = ''

            if epoch % 10 == 0 or epoch == 1:
                print(f"  {epoch:>6}  {tr:>8.4f}  {val:>8.4f}  {best_val:>8.4f}  {tag}")

            if no_improve >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

        if best_state:
            self.model.load_state_dict(best_state)

    def save(self, path: Path):
        torch.save(self.model.state_dict(), path)
        print(f"  ✓  Saved: {path.name}")

    def load(self, path: Path):
        self.model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(f"  ✓  Loaded: {path.name}")


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION (reuses train_stgat.evaluate_portfolio with variant tweaks)
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate_ablation(model, feat_arr, ret_for_feat, nifty_for_feat,
                       regimes, sector_adj, dates, test_start, test_end,
                       variant: str, tc_bps: float = 10.0):
    """
    Monthly rebalancing evaluation.
    NoHMM: use constant lambda + static adjacency.
    NoGAT: use regime-conditioned lambda + dynamic adjacency (as normal).
    """
    model.eval()
    N   = feat_arr.shape[1]
    tc  = tc_bps / 10_000

    test_mask  = (dates >= test_start) & (dates <= test_end)
    test_dates = dates[test_mask]
    if len(test_dates) == 0:
        return pd.Series(dtype=float)

    reb_dates = (test_dates.to_series()
                 .groupby([test_dates.year, test_dates.month])
                 .first().values)
    reb_dates = pd.DatetimeIndex(reb_dates)

    port_ret_map = {}
    prev_w       = None
    CORR_WINDOW  = 63

    for i, reb_d in enumerate(reb_dates):
        t_arr = np.where(dates == reb_d)[0]
        if len(t_arr) == 0:
            continue
        t = int(t_arr[0])
        if t < max(LOOKBACK, CORR_WINDOW):
            continue

        x = feat_arr[t - LOOKBACK:t]
        x = torch.tensor(x, dtype=torch.float32
                         ).permute(1, 0, 2).unsqueeze(0).to(DEVICE)

        ret_window = ret_for_feat[max(0, t - CORR_WINDOW):t]
        regime     = int(regimes[t])

        if variant == 'NoHMM':
            adj = build_static_adj(ret_window, sector_adj)
            lam = CONSTANT_LAMBDA
        else:
            adj = build_dynamic_adj(ret_window, regime, sector_adj)
            lam = REGIME_LAMBDA[regime]

        adj_t = torch.tensor(adj, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        w, _, _ = model(x, adj_t, regime_lambda=lam)
        w_np     = w.squeeze(0).cpu().numpy()

        cost = (np.abs(w_np - prev_w).sum() * tc
                if prev_w is not None else w_np.sum() * tc)
        prev_w = w_np.copy()

        if i + 1 < len(reb_dates):
            next_reb = reb_dates[i + 1]
        else:
            next_reb = test_dates[-1] + pd.Timedelta(days=1)

        hold_mask  = (dates >= reb_d) & (dates < next_reb)
        hold_dates = dates[hold_mask]

        for d_idx, d in enumerate(hold_dates):
            ti_arr = np.where(dates == d)[0]
            if len(ti_arr) == 0:
                continue
            ti = int(ti_arr[0])
            if ti >= len(ret_for_feat):
                continue
            r = float(ret_for_feat[ti] @ w_np)
            if d_idx == 0:
                r -= cost
            port_ret_map[d] = r

    if not port_ret_map:
        return pd.Series(dtype=float)

    return pd.Series(port_ret_map).sort_index()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN WALK-FORWARD LOOP
# ═════════════════════════════════════════════════════════════════════════════
def run_ablation(variant: str,
                 feat_arr, tickers, dates,
                 ret_for_feat, nifty_arr, nifty_ser,
                 sector_adj, eligible_mask_full):
    """Run one ablation variant through all 6 folds + blind."""
    N = len(tickers)
    all_metrics = []
    all_rets    = {}

    folds_all  = FOLDS + [BLIND[:4]]
    fold_names = [f'Fold{i+1}' for i in range(len(FOLDS))] + ['Blind2025']

    print(f"\n{'='*65}")
    print(f"  ABLATION: STGAT-{variant}")
    print(f"{'='*65}")

    for fold_idx, (tr_start, tr_end, te_start, te_end) in enumerate(folds_all):
        fold_name = fold_names[fold_idx]
        is_blind  = fold_name == 'Blind2025'

        print(f"\n── {fold_name}  train→{tr_end}  test {te_start}→{te_end} ──")

        # Regime fitting (always — even NoHMM fits HMM for dataset construction)
        tr_start_ts = pd.Timestamp(tr_start)
        tr_end_ts   = pd.Timestamp(tr_end)

        dates_ret   = dates[1:]
        tr_ret_mask = (dates_ret >= tr_start_ts) & (dates_ret <= tr_end_ts)
        nifty_tr    = nifty_arr[tr_ret_mask]

        hmm_det = HMMRegimeDetector(n_states=3, seed=SEED)
        hmm_det.fit(nifty_tr)
        regimes_full = hmm_det.predict_causal(nifty_arr)
        regimes_aligned = np.zeros(len(dates), dtype=np.int32)
        regimes_aligned[1:] = regimes_full

        rc = np.bincount(regimes_full, minlength=3)
        labels = ['bear', 'sideways', 'bull']
        print(f"  HMM: {' '.join(f'{labels[i]}={rc[i]}' for i in range(3))}")

        # Build dataset (same as train_stgat.py)
        tr_sample_mask = np.array([
            (dates[t] >= tr_start_ts) and (dates[t] <= tr_end_ts)
            for t in range(len(dates))
        ])

        ret_for_feat_full = np.zeros((len(dates), N), dtype=np.float32)
        ret_for_feat_full[1:] = ret_for_feat
        nifty_full = np.zeros(len(dates), dtype=np.float32)
        nifty_full[1:] = nifty_arr

        dataset = STGATDataset(
            feat_arr   = feat_arr,
            ret_arr    = ret_for_feat_full,
            nifty_rets = nifty_full,
            regimes    = regimes_aligned,
            sector_adj = sector_adj,
            tickers    = tickers,
            lookback   = LOOKBACK,
            corr_window= 63,
            eligible_mask = eligible_mask_full,
        )

        train_indices = [i for i, t in enumerate(dataset.valid_idx)
                         if tr_sample_mask[t]]
        if len(train_indices) < 50:
            print(f"  ⚠  Too few samples — skipping")
            continue

        val_split     = int(0.85 * len(train_indices))
        val_indices   = train_indices[val_split:]
        train_indices = train_indices[:val_split]

        train_loader = DataLoader(Subset(dataset, train_indices),
                                   batch_size=BATCH_SIZE, shuffle=True,
                                   collate_fn=collate_fn, num_workers=0)
        val_loader   = DataLoader(Subset(dataset, val_indices),
                                   batch_size=BATCH_SIZE, shuffle=False,
                                   collate_fn=collate_fn, num_workers=0)

        # Train or load
        trainer   = AblationTrainer(n_stocks=N, variant=variant)
        ckpt_name = f'stgat_{variant.lower()}_{fold_name.lower()}.pt'
        ckpt_path = CKPT_DIR / ckpt_name

        if is_blind:
            last_ckpt = CKPT_DIR / f'stgat_{variant.lower()}_fold9.pt'
            if last_ckpt.exists():
                trainer.load(last_ckpt)
            else:
                trainer.train(train_loader, val_loader, fold_name)
                trainer.save(ckpt_path)
        elif ckpt_path.exists():
            trainer.load(ckpt_path)
        else:
            trainer.train(train_loader, val_loader, fold_name)
            trainer.save(ckpt_path)

        # Evaluate
        port_rets = evaluate_ablation(
            model         = trainer.model,
            feat_arr      = feat_arr,
            ret_for_feat  = ret_for_feat_full,
            nifty_for_feat= nifty_full,
            regimes       = regimes_aligned,
            sector_adj    = sector_adj,
            dates         = dates,
            test_start    = te_start,
            test_end      = te_end,
            variant       = variant,
        )

        if port_rets.empty:
            print(f"  ⚠  No returns generated")
            continue

        nifty_te = nifty_ser.reindex(port_rets.index).ffill().fillna(0)
        m = compute_metrics(port_rets, nifty_te)
        m.update({'fold': fold_name, 'strategy': f'STGAT-{variant}',
                   'n_days': len(port_rets)})
        all_metrics.append(m)
        all_rets[fold_name] = port_rets

        print(f"  Return={m.get('ann_return','?'):.4f}  "
              f"Sharpe={m.get('sharpe','?'):.4f}  "
              f"IR={m.get('info_ratio','?'):.4f}  "
              f"MaxDD={m.get('max_drawdown','?'):.4f}")

    return all_metrics, all_rets


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 65)
    print("  NSE Portfolio GNN — Ablation Study")
    print("=" * 65)

    # ── Load features ────────────────────────────────────────────────────
    print("\n── Building feature tensor ──")
    feat_arr, tickers, dates = build_feature_tensor(PRICES_FILE, verbose=True)
    N = len(tickers)

    # ── Returns ──────────────────────────────────────────────────────────
    import pandas as pd
    df_raw   = pd.read_parquet(PRICES_FILE)
    close_df = pd.DataFrame(
        {t: df_raw[(t, t, 'Close')].squeeze() for t in tickers}
    ).reindex(dates).ffill(limit=5)
    ret_df   = np.log(close_df / close_df.shift(1)).iloc[1:]
    ret_arr  = ret_df.fillna(0).values.astype(np.float32)
    # Point-in-time eligibility mask (mirrors train_stgat.py's fix): True
    # once a stock has CORR_WINDOW (63) real trading days of history,
    # avoiding contamination of the correlation graph by pre-listing
    # zero-filled returns.
    T_mask          = len(dates)
    first_valid_idx = close_df.notna().values.argmax(axis=0)
    never_valid     = ~close_df.notna().values.any(axis=0)
    eligible_from   = first_valid_idx + 63
    eligible_from[never_valid] = 10**9
    idx_grid        = np.arange(T_mask)[:, None]
    eligible_mask_full = idx_grid >= eligible_from[None, :]

    nifty_csv = OUT_DIR / 'nifty50_returns.csv'
    nifty_ser = pd.read_csv(nifty_csv, index_col=0, parse_dates=True
                             ).iloc[:, 0].reindex(dates[1:]).ffill().fillna(0)
    nifty_arr = nifty_ser.values.astype(np.float32)

    sector_adj = build_sector_adj(tickers)

    # ── Run both ablations ───────────────────────────────────────────────
    all_results = []

    for variant in ['NoHMM', 'NoGAT']:
        metrics, rets = run_ablation(
            variant     = variant,
            feat_arr    = feat_arr,
            tickers     = tickers,
            dates       = dates,
            ret_for_feat= ret_arr,
            nifty_arr   = nifty_arr,
            nifty_ser   = nifty_ser,
            sector_adj  = sector_adj,
            eligible_mask_full = eligible_mask_full,
        )
        all_results.extend(metrics)

        # Save per-variant returns
        if rets:
            combined = pd.concat(rets.values()).sort_index()
            combined.to_csv(OUT_DIR / f'ablation_{variant}_returns.csv',
                             header=True)

    # ── Save consolidated results ────────────────────────────────────────
    ablation_df = pd.DataFrame(all_results)
    ablation_df.to_csv(OUT_DIR / 'ablation_results.csv', index=False)

    # ── Load full STGAT results for comparison ────────────────────────────
    stgat_df = pd.read_csv(OUT_DIR / 'stgat_fold_results.csv')

    # ── Combined summary ──────────────────────────────────────────────────
    combined_df = pd.concat([stgat_df, ablation_df], ignore_index=True)

    print(f"\n{'='*65}")
    print("  ABLATION SUMMARY — Mean across 6 walk-forward folds")
    print("=" * 65)
    print(f"\n  {'Model':<18} {'Ann.Ret':>8} {'Sharpe':>8} "
          f"{'IR':>8} {'MaxDD':>8} {'WinRate':>8}")
    print(f"  {'─'*58}")

    for strat in ['STGAT-DPO', 'STGAT-NoHMM', 'STGAT-NoGAT']:
        sub = combined_df[
            (combined_df['strategy'] == strat) &
            (combined_df['fold'].str.startswith('Fold'))
        ]
        if sub.empty:
            continue
        marker = '  ← full model' if strat == 'STGAT-DPO' else ''
        print(f"  {strat:<18} {sub['ann_return'].mean():>8.4f} "
              f"{sub['sharpe'].mean():>8.4f} "
              f"{sub['info_ratio'].mean():>8.4f} "
              f"{sub['max_drawdown'].mean():>8.4f} "
              f"{sub['win_rate'].mean():>8.4f}{marker}")

    # Contribution delta
    print(f"\n  {'─'*58}")
    print("  IR contribution of each component:")
    full_ir  = combined_df[(combined_df['strategy']=='STGAT-DPO') &
                            combined_df['fold'].str.startswith('Fold')]['info_ratio'].mean()
    nohmm_ir = combined_df[(combined_df['strategy']=='STGAT-NoHMM') &
                            combined_df['fold'].str.startswith('Fold')]['info_ratio'].mean()
    nogat_ir = combined_df[(combined_df['strategy']=='STGAT-NoGAT') &
                            combined_df['fold'].str.startswith('Fold')]['info_ratio'].mean()

    if not pd.isna(nohmm_ir):
        print(f"  HMM regime gating contribution : {full_ir - nohmm_ir:+.4f} IR")
    if not pd.isna(nogat_ir):
        print(f"  GAT attention contribution     : {full_ir - nogat_ir:+.4f} IR")

    # Save summary
    summary_rows = []
    for strat in ['STGAT-DPO', 'STGAT-NoHMM', 'STGAT-NoGAT']:
        sub = combined_df[
            (combined_df['strategy'] == strat) &
            (combined_df['fold'].str.startswith('Fold'))
        ]
        if sub.empty:
            continue
        row = {'strategy': strat}
        for col in ['ann_return','ann_vol','sharpe','sortino',
                    'max_drawdown','calmar','info_ratio','win_rate']:
            if col in sub.columns:
                row[f'{col}_mean'] = round(sub[col].mean(), 4)
                row[f'{col}_std']  = round(sub[col].std(),  4)
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(
        OUT_DIR / 'ablation_summary.csv', index=False)

    print(f"\n  Results: {OUT_DIR / 'ablation_results.csv'}")
    print(f"  Summary: {OUT_DIR / 'ablation_summary.csv'}")
    print(f"\n{'='*65}")
    print("  Ablation study complete.")
    print(f"{'='*65}\n")


if __name__ == '__main__':
    main()
