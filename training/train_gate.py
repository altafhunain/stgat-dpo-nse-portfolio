"""
training/train_gate.py
========================
Trains a small MLP gating network that blends STGAT-DPO and BL-ML daily
returns: blended_return[t] = g(t) * stgat_ret[t] + (1-g(t)) * blml_ret[t],
where g(t) = MLP(gate_features[t-1]) (lagged by 1 day to avoid using
same-day closing data to decide that day's allocation).

Trained ONLY on the 2021-2024 walk-forward period (2021-2023 fit /
2024 early-stop validation). The 2025-2026 blind holdout is NEVER seen
during training or model selection -- evaluated exactly once, separately,
in evaluate_gate.py.

Loss: negative Information Ratio of the blended return series vs Nifty 50,
computed over randomly sampled 63-day training windows per batch.
"""
import sys, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from copy import deepcopy

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "results"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_START, TRAIN_END = "2021-01-01", "2023-12-31"
VAL_START, VAL_END     = "2024-01-01", "2024-12-31"
WINDOW = 63
N_EPOCHS = 200
PATIENCE = 12
LR = 1e-3
BATCHES_PER_EPOCH = 20

class GateNetwork(nn.Module):
    def __init__(self, in_dim=12, hidden=()):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(0.35)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x)).squeeze(-1)

def information_ratio_loss(port_rets, bench_rets):
    active = port_rets - bench_rets
    mean = active.mean()
    std = active.std()
    if torch.isnan(std) or std < 1e-6:
        return -mean * (252 ** 0.5)
    ir = mean / std * (252 ** 0.5)
    return -ir

print("=" * 70)
print("  Training gate network: STGAT-DPO / BL-ML blend")
print("=" * 70)

# ── Load and align data ──────────────────────────────────────────────────
gate_feat = pd.read_csv(OUT_DIR / "gate_features.csv", index_col=0, parse_dates=True)
stgat_ret = pd.read_csv(OUT_DIR / "stgat_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
blml_ret = pd.read_csv(OUT_DIR / "bl_ml_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
nifty_ret = pd.read_csv(OUT_DIR / "nifty50_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]

stgat_ret = stgat_ret[~stgat_ret.index.duplicated(keep="first")].sort_index()
blml_ret = blml_ret[~blml_ret.index.duplicated(keep="first")].sort_index()
nifty_ret = nifty_ret[~nifty_ret.index.duplicated(keep="first")].sort_index()

# Lag gate features by 1 day (yesterday's info decides today's blend)
gate_feat = gate_feat.drop(columns=["regime_bear", "vol_surge"])
gate_feat_lagged = gate_feat.shift(1)

common = gate_feat_lagged.index.intersection(stgat_ret.index).intersection(blml_ret.index).intersection(nifty_ret.index)
common = common.sort_values()
gate_feat_lagged = gate_feat_lagged.loc[common].fillna(0)
stgat_ret = stgat_ret.loc[common]
blml_ret = blml_ret.loc[common]
nifty_ret = nifty_ret.loc[common]

print(f"  Aligned dataset: {len(common)} days, {gate_feat_lagged.shape[1]} features")

# Standardize features using TRAIN period stats only (no leakage into val/holdout)
train_mask = (common >= TRAIN_START) & (common <= TRAIN_END)
feat_mean = gate_feat_lagged.loc[train_mask].mean()
feat_std = gate_feat_lagged.loc[train_mask].std().replace(0, 1)
gate_feat_norm = (gate_feat_lagged - feat_mean) / feat_std
gate_feat_norm = gate_feat_norm.clip(-5, 5)

X = torch.tensor(gate_feat_norm.values, dtype=torch.float32)
stgat_t = torch.tensor(stgat_ret.values, dtype=torch.float32)
blml_t = torch.tensor(blml_ret.values, dtype=torch.float32)
nifty_t = torch.tensor(nifty_ret.values, dtype=torch.float32)

val_mask = (common >= VAL_START) & (common <= VAL_END)
train_idx = np.where(train_mask)[0]
val_idx = np.where(val_mask)[0]
print(f"  Train days: {len(train_idx)}  ({TRAIN_START} to {TRAIN_END})")
print(f"  Val days:   {len(val_idx)}  ({VAL_START} to {VAL_END})")

model = GateNetwork(in_dim=X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

def blended_ir(idx_range, model, eval_mode=True):
    if eval_mode:
        model.eval()
        with torch.no_grad():
            g = model(X[idx_range])
    else:
        g = model(X[idx_range])
    blended = g * stgat_t[idx_range] + (1 - g) * blml_t[idx_range]
    return blended, nifty_t[idx_range]

def compute_ir(port_rets, bench_rets):
    active = port_rets - bench_rets
    mean = active.mean().item()
    std = active.std().item()
    if std < 1e-8:
        return 0.0
    return mean / std * (252 ** 0.5)

best_val_ir = -1e9
best_state = None
no_improve = 0
print(f"\n  {'Epoch':>6}  {'TrainIR':>9}  {'ValIR':>9}  {'Best':>9}")
print("  " + "-" * 42)

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for _ in range(BATCHES_PER_EPOCH):
        if len(train_idx) <= WINDOW:
            start = 0
        else:
            start = np.random.randint(0, len(train_idx) - WINDOW)
        window_idx = train_idx[start:start + WINDOW]
        blended, bench = blended_ir(window_idx, model, eval_mode=False)
        loss = information_ratio_loss(blended, bench)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    # Full-period train IR (eval mode) and val IR
    train_blended, train_bench = blended_ir(train_idx, model, eval_mode=True)
    train_ir = compute_ir(train_blended, train_bench)
    val_blended, val_bench = blended_ir(val_idx, model, eval_mode=True)
    val_ir = compute_ir(val_blended, val_bench)

    if val_ir > best_val_ir:
        best_val_ir = val_ir
        best_state = deepcopy(model.state_dict())
        no_improve = 0
        tag = "*"
    else:
        no_improve += 1
        tag = ""

    if epoch % 10 == 0 or epoch == 1:
        print(f"  {epoch:>6}  {train_ir:>9.4f}  {val_ir:>9.4f}  {best_val_ir:>9.4f}  {tag}")

    if no_improve >= PATIENCE:
        print(f"  Early stop at epoch {epoch}")
        break

if best_state is not None:
    model.load_state_dict(best_state)

torch.save({
    "model_state": model.state_dict(),
    "feat_mean": feat_mean,
    "feat_std": feat_std,
    "in_dim": X.shape[1],
}, CKPT_DIR / "gate_network.pt")
print(f"\n  Saved: {CKPT_DIR / 'gate_network.pt'}")
print(f"  Best validation IR (2024): {best_val_ir:.4f}")

# Report train/val gate statistics for sanity
model.eval()
with torch.no_grad():
    g_train = model(X[train_idx])
    g_val = model(X[val_idx])
print(f"\n  Gate value stats -- train: mean={g_train.mean():.3f} std={g_train.std():.3f} "
      f"min={g_train.min():.3f} max={g_train.max():.3f}")
print(f"  Gate value stats -- val:   mean={g_val.mean():.3f} std={g_val.std():.3f} "
      f"min={g_val.min():.3f} max={g_val.max():.3f}")
print("\nDone. Blind holdout (2025-2026) NOT touched -- see evaluate_gate.py for that.")
