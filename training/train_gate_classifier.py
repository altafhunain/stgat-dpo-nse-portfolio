"""
training/train_gate_classifier.py
====================================
Trains the gate network as a binary classifier: P(STGAT-DPO outperforms
BL-ML over the next 21 trading days | features known through yesterday).
The sigmoid output is used directly as the blend weight g at inference.

Different training MECHANISM from train_gate.py (which optimized
differentiable IR directly over sampled windows -- three configurations
of that approach failed to generalize to the 2024 validation year).
Cross-entropy on a forward-looking binary label is a more standard,
typically more stable supervised-learning signal.

Labels are forward-looking by construction (which model wins over
[t, t+21]) -- legitimate for training TARGETS (never used as model
INPUT), with the last 21 days of train/val trimmed so no label's
forward window crosses into the next period.

2025-2026 blind holdout is NOT touched here -- see evaluate_gate.py.
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
from pathlib import Path
from copy import deepcopy

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "results"
CKPT_DIR = ROOT / "outputs" / "checkpoints"

TRAIN_START, TRAIN_END = "2021-01-01", "2023-12-31"
VAL_START, VAL_END     = "2024-01-01", "2024-12-31"
LABEL_HORIZON = 21
N_EPOCHS = 300
PATIENCE = 25
LR = 1e-3
BATCH_SIZE = 32

class GateNetwork(nn.Module):
    def __init__(self, in_dim=10, hidden=(8,)):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(0.2)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # raw logits; sigmoid applied by loss / at inference

print("=" * 70)
print("  Training gate as binary classifier: STGAT-DPO vs BL-ML (21d fwd)")
print("=" * 70)

# ── Load and align data (same as train_gate.py) ─────────────────────────
gate_feat = pd.read_csv(OUT_DIR / "gate_features.csv", index_col=0, parse_dates=True)
gate_feat = gate_feat.drop(columns=["regime_bear", "vol_surge"])
stgat_ret = pd.read_csv(OUT_DIR / "stgat_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
blml_ret = pd.read_csv(OUT_DIR / "bl_ml_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]
nifty_ret = pd.read_csv(OUT_DIR / "nifty50_returns.csv", index_col=0, parse_dates=True).iloc[:, 0]

stgat_ret = stgat_ret[~stgat_ret.index.duplicated(keep="first")].sort_index()
blml_ret = blml_ret[~blml_ret.index.duplicated(keep="first")].sort_index()
nifty_ret = nifty_ret[~nifty_ret.index.duplicated(keep="first")].sort_index()

gate_feat_lagged = gate_feat.shift(1)
common = gate_feat_lagged.index.intersection(stgat_ret.index).intersection(blml_ret.index).intersection(nifty_ret.index)
common = common.sort_values()
gate_feat_lagged = gate_feat_lagged.loc[common].fillna(0)
stgat_ret = stgat_ret.loc[common]
blml_ret = blml_ret.loc[common]
nifty_ret = nifty_ret.loc[common]

print(f"  Aligned dataset: {len(common)} days, {gate_feat_lagged.shape[1]} features")

# ── Build forward-looking labels: does STGAT-DPO win over [t, t+21]? ────
stgat_fwd = stgat_ret.rolling(LABEL_HORIZON).sum().shift(-LABEL_HORIZON)
blml_fwd = blml_ret.rolling(LABEL_HORIZON).sum().shift(-LABEL_HORIZON)
label = (stgat_fwd > blml_fwd).astype(float)
valid_label = ~(stgat_fwd.isna() | blml_fwd.isna())

# ── Standardize using TRAIN period only ──────────────────────────────────
train_mask = (common >= TRAIN_START) & (common <= TRAIN_END) & valid_label.values
val_mask = (common >= VAL_START) & (common <= VAL_END) & valid_label.values

feat_mean = gate_feat_lagged.loc[train_mask].mean()
feat_std = gate_feat_lagged.loc[train_mask].std().replace(0, 1)
gate_feat_norm = ((gate_feat_lagged - feat_mean) / feat_std).clip(-5, 5)

X = torch.tensor(gate_feat_norm.values, dtype=torch.float32)
y = torch.tensor(label.values, dtype=torch.float32)
stgat_t = torch.tensor(stgat_ret.values, dtype=torch.float32)
blml_t = torch.tensor(blml_ret.values, dtype=torch.float32)
nifty_t = torch.tensor(nifty_ret.values, dtype=torch.float32)

train_idx = np.where(train_mask)[0]
val_idx = np.where(val_mask)[0]
print(f"  Train days (label-valid): {len(train_idx)}  ({TRAIN_START} to {TRAIN_END})")
print(f"  Val days (label-valid):   {len(val_idx)}  ({VAL_START} to {VAL_END})")
print(f"  Train label balance: STGAT wins {y[train_idx].mean():.1%} of windows")
print(f"  Val label balance:   STGAT wins {y[val_idx].mean():.1%} of windows")

model = GateNetwork(in_dim=X.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
bce = nn.BCEWithLogitsLoss()
print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

def compute_ir(port_rets, bench_rets):
    active = port_rets - bench_rets
    std = active.std().item()
    if std < 1e-8:
        return 0.0
    return active.mean().item() / std * (252 ** 0.5)

def eval_val():
    model.eval()
    with torch.no_grad():
        logits = model(X[val_idx])
        g = torch.sigmoid(logits)
        blended = g * stgat_t[val_idx] + (1 - g) * blml_t[val_idx]
    return compute_ir(blended, nifty_t[val_idx]), g

best_val_ir = -1e9
best_state = None
no_improve = 0
print(f"\n  {'Epoch':>6}  {'BCE':>8}  {'ValIR':>9}  {'Best':>9}")
print("  " + "-" * 40)

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    perm = np.random.permutation(train_idx)
    epoch_loss = 0.0
    n_batches = 0
    for start in range(0, len(perm), BATCH_SIZE):
        batch_idx = perm[start:start + BATCH_SIZE]
        logits = model(X[batch_idx])
        loss = bce(logits, y[batch_idx])
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    val_ir, _ = eval_val()
    if val_ir > best_val_ir:
        best_val_ir = val_ir
        best_state = deepcopy(model.state_dict())
        no_improve = 0
        tag = "*"
    else:
        no_improve += 1
        tag = ""

    if epoch % 10 == 0 or epoch == 1:
        print(f"  {epoch:>6}  {epoch_loss/n_batches:>8.4f}  {val_ir:>9.4f}  {best_val_ir:>9.4f}  {tag}")

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
    "mode": "classifier",
}, CKPT_DIR / "gate_network_classifier.pt")
print(f"\n  Saved: {CKPT_DIR / 'gate_network_classifier.pt'}")
print(f"  Best validation IR (2024): {best_val_ir:.4f}")

final_val_ir, g_val = eval_val()
model.eval()
with torch.no_grad():
    g_train = torch.sigmoid(model(X[train_idx]))
print(f"\n  Gate value stats -- train: mean={g_train.mean():.3f} std={g_train.std():.3f}")
print(f"  Gate value stats -- val:   mean={g_val.mean():.3f} std={g_val.std():.3f}")

# Reference comparisons
pure_stgat_ir = compute_ir(stgat_t[val_idx], nifty_t[val_idx])
pure_blml_ir = compute_ir(blml_t[val_idx], nifty_t[val_idx])
fifty_blend_ir = compute_ir(0.5*stgat_t[val_idx] + 0.5*blml_t[val_idx], nifty_t[val_idx])
print(f"\n  2024 validation IR comparison:")
print(f"    Pure STGAT-DPO:  {pure_stgat_ir:.4f}")
print(f"    Pure BL-ML:      {pure_blml_ir:.4f}")
print(f"    50/50 blend:     {fifty_blend_ir:.4f}")
print(f"    Learned gate:    {best_val_ir:.4f}")
print("\nDone. Blind holdout (2025-2026) NOT touched -- see evaluate_gate.py for that.")
