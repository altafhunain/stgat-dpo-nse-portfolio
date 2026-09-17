"""
ensemble_holdout.py

Tests ensembling across all 8 walk-forward fold checkpoints, evaluated on
the TRUE holdout period (2026 YTD) that NO fold has ever touched. Serves
two purposes: (1) tests whether soft-voting ensembling improves over any
single fold, (2) provides the ultimate out-of-sample validation number.

Each fold has its OWN feature normalization stats, so this builds a
separate normalized dataset per fold, runs each fold's model, then
averages predicted returns (regression) and softmax probabilities
(classification, soft voting) across folds.

Usage:
    python ensemble_holdout.py \
        --checkpoint-root ./walk_forward_results_tb \
        --master-dir ./data/master_with_targets_tb \
        --universe-csv nse500_universe.csv \
        --holdout-start 2026-01-01 --holdout-end 2026-07-29 \
        --output-csv ensemble_holdout_results.csv
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from dataset import StockSequenceDataset, HORIZONS
from model import StockPredictionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ensemble_holdout")

HORIZON_TRADING_DAYS = {"1m": 21, "2m": 42, "3m": 63}


def load_fold_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_cols = ckpt["feature_cols"]
    feature_stats = {"mean": ckpt["feature_mean"], "std": ckpt["feature_std"]}

    regime_cols = ["prob_bull", "prob_bear", "prob_sideways"]
    regime_feature_indices = [feature_cols.index(c) for c in regime_cols]

    model = StockPredictionModel(n_features=len(feature_cols), regime_feature_indices=regime_feature_indices).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, feature_cols, feature_stats


def run_fold_inference(model, master_dir, universe_csv, feature_cols, feature_stats,
                        holdout_start, holdout_end, window, batch_size, device):
    ds = StockSequenceDataset(
        master_dir, universe_csv, window=window,
        start_date=holdout_start, end_date=holdout_end,
        feature_cols=feature_cols, feature_stats=feature_stats,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    keys = ds.samples

    results = {h: {"pred_return": [], "pred_probs": [], "true_return": [], "true_signal": []} for h in HORIZONS}

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            cap_tier = batch["cap_tier"].to(device)
            outputs = model(features, cap_tier)

            for h in HORIZONS:
                results[h]["pred_return"].extend(outputs[h]["return"].cpu().numpy().tolist())
                probs = F.softmax(outputs[h]["signal_logits"], dim=-1).cpu().numpy()
                results[h]["pred_probs"].extend(probs.tolist())
                results[h]["true_return"].extend(batch[f"target_return_{h}"].numpy().tolist())
                results[h]["true_signal"].extend(batch[f"target_signal_{h}"].numpy().tolist())

    return keys, results


def compute_metrics(pred_ret, true_ret, pred_sig, true_sig, horizon):
    pred_ret = np.array(pred_ret)
    true_ret = np.array(true_ret)
    pred_sig = np.array(pred_sig)
    true_sig = np.array(true_sig)

    ic, ic_pvalue = spearmanr(pred_ret, true_ret)
    directional_acc = np.mean(np.sign(pred_ret) == np.sign(true_ret))

    precision, recall, f1, support = precision_recall_fscore_support(
        true_sig, pred_sig, labels=[0, 1, 2], zero_division=0
    )

    position = np.where(pred_sig == 2, 1, np.where(pred_sig == 0, -1, 0))
    strategy_returns = position * true_ret
    active_returns = strategy_returns[position != 0]
    if len(active_returns) > 1 and active_returns.std() > 0:
        sharpe = (active_returns.mean() / active_returns.std()) * np.sqrt(252 / HORIZON_TRADING_DAYS[horizon])
    else:
        sharpe = float("nan")

    return {
        "ic": ic, "ic_pvalue": ic_pvalue, "directional_accuracy": directional_acc,
        "buy_precision": precision[2], "buy_recall": recall[2],
        "sell_precision": precision[0], "sell_recall": recall[0],
        "sharpe": sharpe, "n_active": int((position != 0).sum()), "n_total": len(position),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--master-dir", required=True)
    parser.add_argument("--universe-csv", required=True)
    parser.add_argument("--holdout-start", default="2026-01-01")
    parser.add_argument("--holdout-end", default="2026-07-29")
    parser.add_argument("--window", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--output-csv", default="ensemble_holdout_results.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Using device: %s", device)
    logger.info("TRUE HOLDOUT PERIOD: %s to %s -- untouched by any fold", args.holdout_start, args.holdout_end)

    checkpoint_root = Path(args.checkpoint_root)
    fold_results = {}

    for fold_num in range(1, 9):
        checkpoint_path = checkpoint_root / f"fold_{fold_num}" / "best_model.pt"
        if not checkpoint_path.exists():
            logger.warning("Fold %d checkpoint not found -- skipping", fold_num)
            continue

        logger.info("Running fold %d on holdout period...", fold_num)
        model, feature_cols, feature_stats = load_fold_model(checkpoint_path, device)
        keys, results = run_fold_inference(
            model, args.master_dir, args.universe_csv, feature_cols, feature_stats,
            args.holdout_start, args.holdout_end, args.window, args.batch_size, device
        )
        fold_results[fold_num] = (keys, results)
        logger.info("Fold %d: %d samples on holdout", fold_num, len(keys))

    key_sets = [set(keys) for keys, _ in fold_results.values()]
    if not all(ks == key_sets[0] for ks in key_sets):
        logger.warning("Fold sample keys differ across folds -- using INTERSECTION only")
        common_keys = set.intersection(*key_sets)
    else:
        common_keys = key_sets[0]
    logger.info("Common samples across all folds: %d", len(common_keys))

    all_rows = []
    fold_key_to_idx = {
        fold_num: {k: i for i, k in enumerate(keys)}
        for fold_num, (keys, _) in fold_results.items()
    }
    common_keys_list = list(common_keys)
    fold_nums = list(fold_results.keys())

    for horizon in HORIZONS:
        for fold_num, (keys, results) in fold_results.items():
            key_to_idx = fold_key_to_idx[fold_num]
            common_idx = [key_to_idx[k] for k in common_keys_list]

            pred_ret = [results[horizon]["pred_return"][i] for i in common_idx]
            true_ret = [results[horizon]["true_return"][i] for i in common_idx]
            true_sig = [results[horizon]["true_signal"][i] for i in common_idx]
            pred_sig = [int(np.argmax(results[horizon]["pred_probs"][i])) for i in common_idx]

            m = compute_metrics(pred_ret, true_ret, pred_sig, true_sig, horizon)
            m.update({"horizon": horizon, "model": f"fold_{fold_num}"})
            all_rows.append(m)

        # VECTORIZED ensemble computation: stack all folds' predictions
        # into single numpy arrays ONCE, then use axis-wise mean -- avoids
        # the per-sample Python loop + repeated small np.mean() calls
        # that made the original implementation take 20+ minutes for
        # ~36K samples. This should complete in well under a second.
        returns_matrix = np.array([
            [fold_results[fn][1][horizon]["pred_return"][fold_key_to_idx[fn][k]] for k in common_keys_list]
            for fn in fold_nums
        ])  # shape: (n_folds, n_samples)

        probs_matrix = np.array([
            [fold_results[fn][1][horizon]["pred_probs"][fold_key_to_idx[fn][k]] for k in common_keys_list]
            for fn in fold_nums
        ])  # shape: (n_folds, n_samples, 3)

        ensemble_pred_ret = returns_matrix.mean(axis=0)  # (n_samples,)
        ensemble_probs = probs_matrix.mean(axis=0)  # (n_samples, 3)
        ensemble_pred_sig = ensemble_probs.argmax(axis=1)

        first_fold = fold_nums[0]
        keys0, results0 = fold_results[first_fold]
        key_to_idx0 = fold_key_to_idx[first_fold]
        idx0 = [key_to_idx0[k] for k in common_keys_list]
        ref_true_ret = [results0[horizon]["true_return"][i] for i in idx0]
        ref_true_sig = [results0[horizon]["true_signal"][i] for i in idx0]

        m_ensemble = compute_metrics(ensemble_pred_ret, ref_true_ret, ensemble_pred_sig, ref_true_sig, horizon)
        m_ensemble.update({"horizon": horizon, "model": "ENSEMBLE_ALL_8"})
        all_rows.append(m_ensemble)

        best_single = max(r["ic"] for r in all_rows if r["horizon"] == horizon and r["model"] != "ENSEMBLE_ALL_8")
        logger.info("Horizon %s -- Ensemble IC: %.4f, Sharpe: %.3f | Best single fold IC: %.4f",
                    horizon, m_ensemble["ic"], m_ensemble["sharpe"], best_single)

    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(args.output_csv, index=False)
    logger.info("\nFull results:\n%s", summary_df.to_string(index=False))

    logger.info("\n=== TRUE HOLDOUT (2026 YTD) SUMMARY: Ensemble vs. average of individual folds ===")
    for horizon in HORIZONS:
        h_df = summary_df[summary_df["horizon"] == horizon]
        ensemble_row = h_df[h_df["model"] == "ENSEMBLE_ALL_8"].iloc[0]
        individual_avg = h_df[h_df["model"] != "ENSEMBLE_ALL_8"]["ic"].mean()
        logger.info("%s: Ensemble IC=%.4f | Individual-fold avg IC=%.4f | Ensemble Sharpe=%.3f",
                    horizon, ensemble_row["ic"], individual_avg, ensemble_row["sharpe"])


if __name__ == "__main__":
    main()
