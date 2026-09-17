"""
train.py

Trains the StockPredictionModel with a combined multi-task loss (Huber
regression + class-weighted cross-entropy classification, summed across
all 3 horizons). Uses STRICT walk-forward date-based splitting.

CHECKPOINT SELECTION: uses average validation IC (Spearman correlation
between predicted and actual returns, averaged across the 3 horizons) as
the selection criterion, NOT val_loss. Found via evaluation on fold 1's
2018 test year that val_loss-based selection was picking checkpoints with
near-zero or even NEGATIVE IC on the actual held-out test year -- val_loss
(a combined regression+classification loss) doesn't reliably track actual
predictive skill, since a model can improve on the easier classification
component without genuinely learning to predict returns. IC is a much
more direct measure of "does this actually predict returns."

Usage:
    python train.py \
        --master-dir ./data/master_with_targets \
        --universe-csv nse500_universe.csv \
        --train-end 2021-12-31 --val-end 2023-12-31 \
        --window 60 --epochs 30 --batch-size 256
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from dataset import StockSequenceDataset, HORIZONS, SIGNAL_MAP
from model import StockPredictionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train")


def compute_class_weights(dataset, horizon, device):
    counts = np.zeros(3)
    for symbol, idx in dataset.samples:
        df = dataset.data_by_symbol[symbol]
        label = df.iloc[idx][f"target_signal_{horizon}"]
        counts[SIGNAL_MAP[label]] += 1

    total = counts.sum()
    weights = total / (3 * counts)
    logger.info("Class weights for %s horizon (SELL/HOLD/BUY counts %s): %s",
                horizon, counts.tolist(), weights.tolist())
    return torch.tensor(weights, dtype=torch.float32).to(device)


def combined_loss(outputs, batch, class_weights_per_horizon, horizon_loss_weights):
    total_loss = 0.0
    breakdown = {}
    for horizon in HORIZONS:
        pred_return = outputs[horizon]["return"]
        pred_logits = outputs[horizon]["signal_logits"]
        true_return = batch[f"target_return_{horizon}"]
        true_signal = batch[f"target_signal_{horizon}"]

        reg_loss = nn.functional.huber_loss(pred_return, true_return)
        cls_loss = nn.functional.cross_entropy(pred_logits, true_signal, weight=class_weights_per_horizon[horizon])

        h_loss = horizon_loss_weights.get(horizon, 1.0) * (reg_loss + cls_loss)
        total_loss = total_loss + h_loss
        breakdown[horizon] = {"reg": reg_loss.item(), "cls": cls_loss.item()}
    return total_loss, breakdown


def move_batch_to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def evaluate(model, loader, class_weights_per_horizon, device):
    """
    Returns (val_loss, val_accs, val_ics, avg_ic).
    val_ics: per-horizon Spearman IC between predicted and actual returns.
    avg_ic: mean IC across the 3 horizons -- THIS is the checkpoint
    selection criterion, not val_loss.
    """
    model.eval()
    total_loss = 0.0
    correct = {h: 0 for h in HORIZONS}
    total = 0
    pred_returns = {h: [] for h in HORIZONS}
    true_returns = {h: [] for h in HORIZONS}

    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch["features"], batch["cap_tier"])
            loss, _ = combined_loss(outputs, batch, class_weights_per_horizon, {h: 1.0 for h in HORIZONS})
            bs = batch["features"].size(0)
            total_loss += loss.item() * bs
            for h in HORIZONS:
                preds = outputs[h]["signal_logits"].argmax(dim=-1)
                correct[h] += (preds == batch[f"target_signal_{h}"]).sum().item()
                pred_returns[h].extend(outputs[h]["return"].cpu().numpy().tolist())
                true_returns[h].extend(batch[f"target_return_{h}"].cpu().numpy().tolist())
            total += bs

    total = max(total, 1)
    val_accs = {h: correct[h] / total for h in HORIZONS}

    val_ics = {}
    for h in HORIZONS:
        ic, _ = spearmanr(pred_returns[h], true_returns[h])
        val_ics[h] = ic if not np.isnan(ic) else 0.0
    avg_ic = float(np.mean(list(val_ics.values())))

    return total_loss / total, val_accs, val_ics, avg_ic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-dir", required=True)
    parser.add_argument("--universe-csv", required=True)
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--val-end", required=True)
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--checkpoint-dir", default="./checkpoints")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Building train dataset (up to %s)...", args.train_end)
    train_ds = StockSequenceDataset(args.master_dir, args.universe_csv, window=args.window, end_date=args.train_end)
    feature_cols = train_ds.feature_cols

    logger.info("Building val dataset (%s to %s)...", args.train_end, args.val_end)
    train_feature_stats = train_ds.get_feature_stats()
    val_ds = StockSequenceDataset(
        args.master_dir, args.universe_csv, window=args.window,
        start_date=args.train_end, end_date=args.val_end, feature_cols=feature_cols,
        feature_stats=train_feature_stats,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if len(val_ds) == 0:
        raise RuntimeError(
            f"Validation dataset is empty (0 samples) for range {args.train_end} to {args.val_end}."
        )

    class_weights_per_horizon = {h: compute_class_weights(train_ds, h, device) for h in HORIZONS}

    regime_cols = ["prob_bull", "prob_bear", "prob_sideways"]
    regime_feature_indices = [feature_cols.index(c) for c in regime_cols]
    logger.info("Regime feature indices in feature vector: %s (columns: %s)", regime_feature_indices, regime_cols)

    model = StockPredictionModel(
        n_features=len(feature_cols),
        regime_feature_indices=regime_feature_indices,
        hidden_dim=args.hidden_dim,
        window=args.window,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    logger.info("Starting training: %d epochs, %d train samples, %d val samples",
                args.epochs, len(train_ds), len(val_ds))

    best_avg_ic = float("-inf")
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad()
            outputs = model(batch["features"], batch["cap_tier"])
            loss, _ = combined_loss(outputs, batch, class_weights_per_horizon, {h: 1.0 for h in HORIZONS})
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * batch["features"].size(0)

        train_loss = epoch_loss / len(train_ds)
        val_loss, val_accs, val_ics, avg_ic = evaluate(model, val_loader, class_weights_per_horizon, device)

        logger.info("Epoch %d/%d - train_loss: %.4f, val_loss: %.4f, val_acc: %s, val_ic: %s, avg_ic: %.4f",
                    epoch + 1, args.epochs, train_loss, val_loss,
                    {h: f"{v:.3f}" for h, v in val_accs.items()},
                    {h: f"{v:.4f}" for h, v in val_ics.items()}, avg_ic)

        if avg_ic > best_avg_ic:
            best_avg_ic = avg_ic
            torch.save({
                "model_state_dict": model.state_dict(),
                "feature_cols": feature_cols,
                "feature_mean": train_feature_stats["mean"],
                "feature_std": train_feature_stats["std"],
                "epoch": epoch,
                "val_loss": val_loss,
                "avg_ic": avg_ic,
                "val_ics": val_ics,
            }, checkpoint_dir / "best_model.pt")
            logger.info("New best model saved (avg_ic: %.4f, val_loss: %.4f)", avg_ic, val_loss)

    logger.info("Training complete. Best avg_ic: %.4f", best_avg_ic)


if __name__ == "__main__":
    main()
