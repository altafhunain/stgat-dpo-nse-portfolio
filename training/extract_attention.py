"""
training/extract_attention.py
==============================
Extracts GAT attention weights from saved STGAT-DPO fold checkpoints
and generates publication-quality attention heatmaps per HMM regime.

Figures produced:
  fig9_attention_heatmap.png    — 3-panel 51×51 attention per regime
  fig10_attention_top_pairs.png — top attending stock pairs per regime

Run:
    python training/extract_attention.py
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

from utils.features import build_feature_tensor, N_FEATURES
from models.gnn.stgat_dpo import (
    STGATDPOModel, HMMRegimeDetector,
    build_sector_adj, build_dynamic_adj, REGIME_LAMBDA
)
from training.train_stgat import (
    FOLDS, LOOKBACK, MAX_WEIGHT, SEED, DEVICE, PRICES_FILE
)

torch.manual_seed(SEED)
np.random.seed(SEED)

ROOT     = Path(__file__).resolve().parent.parent
OUT_DIR  = ROOT / "outputs" / "results"
FIG_DIR  = ROOT / "outputs" / "figures"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Sector membership for axis grouping
SECTOR_MAP = {
    "IT":           ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS","MPHASIS.NS"],
    "Banking":      ["HDFCBANK.NS","ICICIBANK.NS","AXISBANK.NS","INDUSINDBK.NS","BAJFINANCE.NS","SBIN.NS"],
    "FMCG":         ["HINDUNILVR.NS","ITC.NS","NESTLEIND.NS","BRITANNIA.NS","DABUR.NS","GODREJCP.NS","TATACONSUM.NS","UBL.NS"],
    "Auto":         ["MARUTI.NS","BAJAJFINSV.NS","M&M.NS"],
    "Pharma":       ["SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS"],
    "Energy":       ["RELIANCE.NS","NTPC.NS","COALINDIA.NS","BPCL.NS","ONGC.NS","TATAPOWER.NS"],
    "Infrastructure":["LT.NS","ADANIPORTS.NS","ULTRACEMCO.NS"],
    "Consumer":     ["ASIANPAINT.NS","TITAN.NS","BERGEPAINT.NS","HAVELLS.NS","VOLTAS.NS"],
    "Telecom":      ["BHARTIARTL.NS"],
    "Metals":       ["HINDALCO.NS","JSWSTEEL.NS"],
    "Insurance":    ["HDFCLIFE.NS","SBILIFE.NS","ICICIPRULI.NS"],
    "Power":        ["POWERGRID.NS"],
    "Chemicals":    ["PIDILITIND.NS"],
    "Cement":       ["GRASIM.NS","SHREECEM.NS","AMBUJACEM.NS"],
}

SECTOR_COLORS = {
    "IT":           "#2196F3",
    "Banking":      "#9C27B0",
    "FMCG":         "#4CAF50",
    "Auto":         "#FF9800",
    "Pharma":       "#F44336",
    "Energy":       "#795548",
    "Infrastructure":"#607D8B",
    "Consumer":     "#00BCD4",
    "Telecom":      "#FF5722",
    "Metals":       "#9E9E9E",
    "Insurance":    "#3F51B5",
    "Power":        "#FFEB3B",
    "Chemicals":    "#8BC34A",
    "Cement":       "#FF80AB",
}


# ═════════════════════════════════════════════════════════════════════════════
# ATTENTION HOOK — intercept GAT attention weights
# ═════════════════════════════════════════════════════════════════════════════
class AttentionCapture:
    """Context manager that hooks into the GAT layer to capture α weights."""

    def __init__(self):
        self.alpha_gat1 = None   # [B, N, N, H] from gat1
        self.alpha_gat2 = None   # [B, N, N, H] from gat2
        self._hooks     = []

    def register(self, model: STGATDPOModel):
        """Attach forward hooks to both GAT layers."""

        def hook_gat1(module, input, output):
            # Recompute alpha from the stored logit + adj
            # We monkey-patch the forward to return alpha alongside output
            pass

        # Patch the GAT forward to expose alpha
        original_gat1_forward = model.gat1.forward
        original_gat2_forward = model.gat2.forward
        capture = self

        def patched_gat1(h, adj):
            B, N, _ = h.shape
            Wh = model.gat1.W(h).view(B, N, model.gat1.n_heads,
                                        model.gat1.head_dim)
            e_src = (Wh * model.gat1.a_src).sum(-1)
            e_dst = (Wh * model.gat1.a_dst).sum(-1)
            e = model.gat1.leaky(e_src.unsqueeze(2) + e_dst.unsqueeze(1))
            if adj.dim() == 2:
                adj_b = adj.unsqueeze(0).unsqueeze(-1)
            else:
                adj_b = adj.unsqueeze(-1)
            e = e.masked_fill(adj_b == 0, float("-inf"))
            alpha = F.softmax(e, dim=2)
            alpha = torch.nan_to_num(alpha, nan=0.0)
            capture.alpha_gat1 = alpha.detach().cpu()
            out = torch.einsum("bijh,bjhd->bihd", alpha, Wh)
            out = out.reshape(B, N, model.gat1.out_dim)
            out = F.elu(out)
            out = model.gat1.norm(out)
            return out

        def patched_gat2(h, adj):
            B, N, _ = h.shape
            Wh = model.gat2.W(h).view(B, N, model.gat2.n_heads,
                                        model.gat2.head_dim)
            e_src = (Wh * model.gat2.a_src).sum(-1)
            e_dst = (Wh * model.gat2.a_dst).sum(-1)
            e = model.gat2.leaky(e_src.unsqueeze(2) + e_dst.unsqueeze(1))
            if adj.dim() == 2:
                adj_b = adj.unsqueeze(0).unsqueeze(-1)
            else:
                adj_b = adj.unsqueeze(-1)
            e = e.masked_fill(adj_b == 0, float("-inf"))
            alpha = F.softmax(e, dim=2)
            alpha = torch.nan_to_num(alpha, nan=0.0)
            capture.alpha_gat2 = alpha.detach().cpu()
            out = torch.einsum("bijh,bjhd->bihd", alpha, Wh)
            out = out.reshape(B, N, model.gat2.out_dim)
            out = F.elu(out)
            out = model.gat2.norm(out)
            return out

        model.gat1.forward = patched_gat1
        model.gat2.forward = patched_gat2
        return model


# ═════════════════════════════════════════════════════════════════════════════
# EXTRACT ATTENTION WEIGHTS ACROSS ALL FOLDS AND REGIMES
# ═════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def extract_attention_by_regime(model, feat_arr, ret_for_feat,
                                  nifty_arr, regimes, reg_post,
                                  sector_adj, dates,
                                  test_start, test_end,
                                  capture: AttentionCapture):
    """
    Run forward passes on test period days, grouped by HMM regime.
    Returns dict: regime → mean attention matrix [N, N]
    """
    model.eval()
    N           = feat_arr.shape[1]
    CORR_WINDOW = 63
    regime_attn = {0: [], 1: [], 2: []}  # bear/sideways/bull

    test_mask  = (dates >= test_start) & (dates <= test_end)
    test_dates = dates[test_mask]

    for d in test_dates[::5]:   # sample every 5 days for speed
        t_arr = np.where(dates == d)[0]
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
        lam        = REGIME_LAMBDA[regime]
        adj        = build_dynamic_adj(ret_window, regime, sector_adj)
        adj_t      = torch.tensor(adj, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        _ = model(x, adj_t, regime_lambda=lam)

        if capture.alpha_gat1 is not None:
            # Mean across heads and batch: [N, N]
            attn = capture.alpha_gat1[0].mean(-1).numpy()  # [N, N]
            regime_attn[regime].append(attn)

    # Average within each regime
    result = {}
    for reg, attn_list in regime_attn.items():
        if attn_list:
            result[reg] = np.stack(attn_list).mean(axis=0)  # [N, N]
        else:
            result[reg] = np.zeros((N, N), dtype=np.float32)

    return result


# ═════════════════════════════════════════════════════════════════════════════
# SORT TICKERS BY SECTOR
# ═════════════════════════════════════════════════════════════════════════════
def get_sector_order(tickers):
    """Return tickers sorted by sector, with sector boundaries."""
    ticker_to_sector = {}
    for sector, members in SECTOR_MAP.items():
        for t in members:
            if t in tickers:
                ticker_to_sector[t] = sector

    # Order: group by sector
    sector_order = list(SECTOR_MAP.keys())
    sorted_tickers = []
    sector_boundaries = []
    for sector in sector_order:
        group = [t for t in tickers
                 if ticker_to_sector.get(t) == sector]
        if group:
            sector_boundaries.append((len(sorted_tickers),
                                       len(sorted_tickers) + len(group),
                                       sector))
            sorted_tickers.extend(group)

    # Add any unlisted tickers
    for t in tickers:
        if t not in sorted_tickers:
            sorted_tickers.append(t)

    return sorted_tickers, sector_boundaries


# ═════════════════════════════════════════════════════════════════════════════
# FIGURES
# ═════════════════════════════════════════════════════════════════════════════
def plot_attention_heatmaps(regime_attn_all: dict,
                             tickers: list):
    """
    fig9: 3-panel attention heatmap (bear / sideways / bull).
    Stocks sorted by sector. Sector color bars on axes.
    """
    sorted_tickers, sector_bounds = get_sector_order(tickers)
    idx_map = {t: i for i, t in enumerate(tickers)}
    reorder = [idx_map[t] for t in sorted_tickers]

    short_labels = [t.replace(".NS", "") for t in sorted_tickers]

    regime_labels = {0: "Bear regime", 1: "Sideways regime", 2: "Bull regime"}
    regime_cmaps  = {0: "Reds", 1: "YlOrBr", 2: "Greens"}

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    fig.patch.set_facecolor("white")

    for ax, (regime, label) in zip(axes, regime_labels.items()):
        attn = regime_attn_all.get(regime,
                np.zeros((len(tickers), len(tickers))))

        # Reorder by sector
        attn_reord = attn[np.ix_(reorder, reorder)]

        # Normalise per row (attention is already row-normalised but
        # averaging across time/days may shift it slightly)
        row_sum = attn_reord.sum(axis=1, keepdims=True).clip(min=1e-8)
        attn_norm = attn_reord / row_sum

        N = len(sorted_tickers)
        im = ax.imshow(attn_norm, cmap=regime_cmaps[regime],
                       aspect="auto", interpolation="nearest",
                       vmin=0, vmax=attn_norm.max())

        # Sector boundary lines
        for start, end, sector in sector_bounds:
            ax.axhline(start - 0.5, color="white", lw=0.8, alpha=0.7)
            ax.axvline(start - 0.5, color="white", lw=0.8, alpha=0.7)

        # Axis labels — show every 3rd ticker to avoid crowding
        tick_pos   = list(range(0, N, 3))
        tick_labels= [short_labels[i] for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=6.5)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_labels, fontsize=6.5)

        # Sector color bar on left margin
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks([])
        for start, end, sector in sector_bounds:
            mid = (start + end) / 2
            color = SECTOR_COLORS.get(sector, "#888888")
            ax2.add_patch(plt.Rectangle(
                (1.01, start / N), 0.025, (end - start) / N,
                transform=ax2.transAxes,
                color=color, clip_on=False))
            if end - start >= 3:
                ax2.text(1.045, (start + end) / 2 / N, sector,
                          transform=ax2.transAxes,
                          fontsize=6, va="center", ha="left",
                          color=color)

        ax.set_xlabel(label, fontsize=10, fontweight="500", labelpad=8)
        plt.colorbar(im, ax=ax, shrink=0.75, pad=0.08,
                     label="Attention weight")

    plt.tight_layout(pad=1.5)
    out = FIG_DIR / "fig9_attention_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {out.name}")


def plot_top_attention_pairs(regime_attn_all: dict,
                              tickers: list):
    """
    fig10: Top-15 attending stock pairs per regime.
    Shows which cross-stock relationships the GAT learns.
    """
    regime_labels = {0: "Bear", 1: "Sideways", 2: "Bull"}
    short         = {t: t.replace(".NS", "") for t in tickers}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.patch.set_facecolor("white")

    for ax, (regime, label) in zip(axes, regime_labels.items()):
        attn = regime_attn_all.get(regime,
                np.zeros((len(tickers), len(tickers))))
        np.fill_diagonal(attn, 0)   # exclude self-attention

        # Get top-15 pairs
        flat_idx = np.argsort(attn.flatten())[::-1][:15]
        rows, cols = np.unravel_index(flat_idx, attn.shape)

        pair_labels = [f"{short[tickers[r]]} → {short[tickers[c]]}"
                       for r, c in zip(rows, cols)]
        pair_vals   = [attn[r, c] for r, c in zip(rows, cols)]

        # Colour by sector relationship
        def same_sector(t1, t2):
            for members in SECTOR_MAP.values():
                if t1 in members and t2 in members:
                    return True
            return False

        colors = ["#2196F3" if same_sector(tickers[r], tickers[c])
                  else "#FF9800"
                  for r, c in zip(rows, cols)]

        y = np.arange(len(pair_labels))
        ax.barh(y[::-1], pair_vals, color=colors[::-1],
                zorder=3, edgecolor="white", linewidth=0.3)
        ax.set_yticks(y)
        ax.set_yticklabels(pair_labels[::-1], fontsize=7.5)
        ax.set_xlabel("Attention weight", fontsize=9)
        ax.set_title(f"{label} regime", fontsize=10,
                     fontweight="500", pad=6)
        ax.grid(True, alpha=0.3, axis="x")
        ax.spines[["top", "right"]].set_visible(False)

        # Legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="#2196F3", label="Same sector"),
            Patch(color="#FF9800", label="Cross-sector"),
        ], fontsize=7.5, loc="lower right")

    plt.tight_layout(pad=1.5)
    out = FIG_DIR / "fig10_attention_top_pairs.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✓  {out.name}")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Extracting GAT attention weights")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────
    feat_arr, tickers, dates = build_feature_tensor(PRICES_FILE, verbose=True)
    N = len(tickers)

    df_raw   = pd.read_parquet(PRICES_FILE)
    close_df = pd.DataFrame(
        {t: df_raw[(t, t, "Close")].squeeze() for t in tickers}
    ).reindex(dates).ffill(limit=5)
    ret_df   = np.log(close_df / close_df.shift(1)).iloc[1:]
    ret_arr  = ret_df.fillna(0).values.astype(np.float32)
    ret_full = np.zeros((len(dates), N), dtype=np.float32)
    ret_full[1:] = ret_arr

    nifty_ser = pd.read_csv(OUT_DIR / "nifty50_returns.csv",
                             index_col=0, parse_dates=True
                             ).iloc[:, 0].reindex(dates[1:]).ffill().fillna(0)
    nifty_full = np.zeros(len(dates), dtype=np.float32)
    nifty_full[1:] = nifty_ser.values

    sector_adj = build_sector_adj(tickers)

    # ── Accumulate attention across all folds ────────────────────────────
    # Combine folds 4, 5, 6 (most data, most recent, covers both bull+bear)
    use_folds = [
        ("Fold4", "2018-01-01", "2022-06-30", "2022-07-01", "2023-06-30"),
        ("Fold5", "2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
        ("Fold6", "2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ]

    accumulated = {0: [], 1: [], 2: []}   # regime → list of [N,N] matrices

    for fold_name, tr_start, tr_end, te_start, te_end in use_folds:
        ckpt_path = CKPT_DIR / f"stgat_{fold_name.lower()}.pt"
        if not ckpt_path.exists():
            print(f"  ⚠  Checkpoint not found: {ckpt_path.name} — skipping")
            continue

        print(f"\n── Processing {fold_name} ──")

        # Fit HMM for this fold
        tr_start_ts = pd.Timestamp(tr_start)
        tr_end_ts   = pd.Timestamp(tr_end)
        dates_ret   = dates[1:]
        tr_mask     = (dates_ret >= tr_start_ts) & (dates_ret <= tr_end_ts)
        nifty_tr    = nifty_ser.reindex(dates_ret[tr_mask]).fillna(0).values

        hmm_det = HMMRegimeDetector(n_states=3, seed=SEED)
        hmm_det.fit(nifty_tr)

        nifty_all    = nifty_ser.reindex(dates[1:]).fillna(0).values
        regimes_raw  = hmm_det.predict_causal(nifty_all)
        regimes_full = np.zeros(len(dates), dtype=np.int32)
        regimes_full[1:] = regimes_raw

        rc = np.bincount(regimes_raw, minlength=3)
        print(f"  HMM: bear={rc[0]} sideways={rc[1]} bull={rc[2]}")

        # Load model + attach capture
        model = STGATDPOModel(
            n_stocks=N, n_features=N_FEATURES, lookback=LOOKBACK,
            lstm_hidden=64, lstm_layers=2, gat_hidden=64, gat_heads=2,
            dropout=0.2, max_weight=MAX_WEIGHT,
        ).to(DEVICE)
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()

        capture = AttentionCapture()
        model   = capture.register(model)

        # Extract
        regime_attn = extract_attention_by_regime(
            model        = model,
            feat_arr     = feat_arr,
            ret_for_feat = ret_full,
            nifty_arr    = nifty_full,
            regimes      = regimes_full,
            reg_post     = None,
            sector_adj   = sector_adj,
            dates        = dates,
            test_start   = te_start,
            test_end     = te_end,
            capture      = capture,
        )

        for reg, attn in regime_attn.items():
            if attn.sum() > 0:
                accumulated[reg].append(attn)
                print(f"  Regime {['bear','sideways','bull'][reg]}: "
                      f"{attn.shape} attention matrix captured")

    # ── Average across folds ──────────────────────────────────────────────
    regime_attn_final = {}
    for reg, attn_list in accumulated.items():
        if attn_list:
            regime_attn_final[reg] = np.stack(attn_list).mean(axis=0)
            print(f"  Regime {['bear','sideways','bull'][reg]}: "
                  f"averaged {len(attn_list)} fold(s)")
        else:
            regime_attn_final[reg] = np.ones((N, N), dtype=np.float32) / N

    # ── Save raw attention matrices ───────────────────────────────────────
    for reg, attn in regime_attn_final.items():
        label = ["bear", "sideways", "bull"][reg]
        df_attn = pd.DataFrame(attn, index=tickers, columns=tickers)
        df_attn.to_csv(OUT_DIR / f"attention_{label}.csv")
        print(f"  ✓  Saved: attention_{label}.csv")

    # ── Generate figures ──────────────────────────────────────────────────
    print("\n── Generating attention figures ──")
    plot_attention_heatmaps(regime_attn_final, tickers)
    plot_top_attention_pairs(regime_attn_final, tickers)

    # ── Summary statistics ────────────────────────────────────────────────
    print("\n── Attention pattern analysis ──")
    for reg in [0, 1, 2]:
        label = ["Bear", "Sideways", "Bull"][reg]
        attn  = regime_attn_final[reg].copy()
        np.fill_diagonal(attn, 0)

        # Same-sector vs cross-sector attention
        same_mask  = np.zeros((N, N), dtype=bool)
        for members in SECTOR_MAP.values():
            idxs = [tickers.index(t) for t in members if t in tickers]
            for i in idxs:
                for j in idxs:
                    if i != j:
                        same_mask[i, j] = True

        same_attn  = attn[same_mask].mean()  if same_mask.any()  else 0
        cross_attn = attn[~same_mask & (attn > 0)].mean() if (~same_mask).any() else 0

        print(f"\n  {label} regime:")
        print(f"    Same-sector attention  : {same_attn:.5f}")
        print(f"    Cross-sector attention : {cross_attn:.5f}")
        print(f"    Same/cross ratio       : {same_attn/(cross_attn+1e-8):.2f}x")

    print(f"\n  Figures saved to: {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
