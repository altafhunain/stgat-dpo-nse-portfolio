"""
Sensitivity analysis v2 (all 9 walk-forward folds, checkpoint-only, no retraining).
Baseline = trained regime schedule; threshold sweep = additive shift; cash start;
no silent fallback; main-metric conventions; records traded notional / cost drag.
Writes *_v2.csv only; nothing existing is overwritten.
"""
import sys, os
os.environ["OMP_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch
from pathlib import Path
from utils.features import build_feature_tensor, N_FEATURES
from models.gnn.stgat_dpo import (STGATDPOModel, HMMRegimeDetector, build_sector_adj,
                                  REGIME_LAMBDA, REGIME_THRESHOLDS)
from training.train_stgat import FOLDS, LOOKBACK, MAX_WEIGHT, SEED, DEVICE, PRICES_FILE

torch.manual_seed(SEED); np.random.seed(SEED)
ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "results"; CKPT_DIR = ROOT / "outputs" / "checkpoints"
CORR_WINDOW = 63; RF_DAILY = 0.065 / 252


@torch.no_grad()
def evaluate_v2(model, feat_arr, ret_full, nifty_ser, regimes, sector_adj, dates,
                test_start, test_end, tc_bps=10, thr_shift=0.0, lookback=20, max_weight=0.15):
    model.eval()
    N = feat_arr.shape[1]; tc = tc_bps / 10_000
    test_dates = dates[(dates >= test_start) & (dates <= test_end)]
    if len(test_dates) == 0:
        return {}
    reb_dates = pd.DatetimeIndex(test_dates.to_series()
                                 .groupby([test_dates.year, test_dates.month]).first().values)
    port, prev_w, traded = {}, None, []
    for i, reb_d in enumerate(reb_dates):
        t_arr = np.where(dates == reb_d)[0]
        if len(t_arr) == 0:
            continue
        t = int(t_arr[0])
        if t < max(lookback, CORR_WINDOW):
            continue
        x = feat_arr[max(0, t - lookback):t]
        if len(x) < lookback:
            pad = np.zeros((lookback - len(x), N, feat_arr.shape[2]), dtype=np.float32)
            x = np.concatenate([pad, x], axis=0)
        x_t = torch.tensor(x, dtype=torch.float32).permute(1, 0, 2).unsqueeze(0).to(DEVICE)

        regime = int(regimes[t]); lam = REGIME_LAMBDA[regime]
        thr = REGIME_THRESHOLDS[regime] + thr_shift
        corr = np.nan_to_num(np.corrcoef(ret_full[max(0, t - CORR_WINDOW):t].T), nan=0.0)
        dyn = (np.abs(corr) > thr).astype(np.float32)
        np.fill_diagonal(dyn, 0.0)
        dyn = dyn * np.abs(corr).clip(0, 1)
        adj = (0.7 * dyn + 0.3 * sector_adj).clip(0, 1)
        np.fill_diagonal(adj, 0.0)
        adj_t = torch.tensor(adj.astype(np.float32)).unsqueeze(0).to(DEVICE)

        w, _, _ = model(x_t, adj_t, regime_lambda=lam)          # no silent fallback
        w_np = w.squeeze(0).cpu().numpy()
        if max_weight != 0.15:                                   # inference-time cap perturbation
            w_np = np.clip(w_np, 0, max_weight)
            w_np = w_np / (w_np.sum() + 1e-8)

        if prev_w is None:
            cost = w_np.sum() * tc                               # initial buy-in from cash
        else:
            cost = np.abs(w_np - prev_w).sum() * tc
            traded.append(np.abs(w_np - prev_w).sum())
        prev_w = w_np.copy()

        next_reb = reb_dates[i + 1] if i + 1 < len(reb_dates) else test_dates[-1] + pd.Timedelta(days=1)
        hold = dates[(dates >= reb_d) & (dates < next_reb)]
        for d_idx, d in enumerate(hold):
            ti = int(np.where(dates == d)[0][0])
            if ti >= len(ret_full):
                continue
            r = float(ret_full[ti] @ w_np)
            if d_idx == 0:
                r -= cost
            port[d] = r

    rets = pd.Series(port).sort_index()
    if len(rets) < 5:
        return {}
    nifty_te = nifty_ser.reindex(rets.index).ffill().fillna(0)
    ann = 252
    excess = rets - RF_DAILY
    active = rets - nifty_te
    cum = np.exp(rets.cumsum())
    return {
        "ann_return": round(float(rets.mean() * ann), 4),
        "ann_vol": round(float(rets.std() * ann ** 0.5), 4),
        "sharpe": round(float(excess.mean() / (rets.std() + 1e-8) * ann ** 0.5), 4),
        "info_ratio": round(float(active.mean() / (active.std() + 1e-8) * ann ** 0.5), 4),
        "max_drawdown": round(float(((cum - cum.cummax()) / cum.cummax()).min()), 4),
        "mean_traded": round(float(np.mean(traded)) if traded else np.nan, 4),
    }


def main():
    print("=" * 66); print("  Sensitivity v2 - STGAT-DPO, all folds, trained configuration"); print("=" * 66)
    feat_arr, tickers, dates = build_feature_tensor(PRICES_FILE, verbose=False)
    N = len(tickers)
    df_raw = pd.read_parquet(PRICES_FILE)
    close_df = pd.DataFrame({t: df_raw[(t, t, "Close")].squeeze() for t in tickers}).reindex(dates).ffill(limit=5)
    ret_df = np.log(close_df / close_df.shift(1)).iloc[1:]
    ret_full = np.zeros((len(dates), N), dtype=np.float32)
    ret_full[1:] = ret_df.fillna(0).values.astype(np.float32)
    nifty_ser = (pd.read_csv(OUT_DIR / "nifty50_returns.csv", index_col=0, parse_dates=True)
                 .iloc[:, 0].reindex(dates[1:]).ffill().fillna(0))
    sector_adj = build_sector_adj(tickers); dates_ret = dates[1:]

    main_ir = {}
    try:
        m = pd.read_csv(OUT_DIR / "stgat_fold_results.csv")
        if "strategy" in m.columns:
            m = m[m["strategy"] == "STGAT-DPO"]
        main_ir = dict(zip(m["fold"], m["info_ratio"]))
    except Exception as e:
        print("  (could not read stgat_fold_results.csv for validation:", e, ")")

    sweeps = [
        ("transaction_cost", [0, 5, 10, 20, 30, 50], lambda v: dict(tc_bps=v)),
        ("threshold_shift", [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10], lambda v: dict(thr_shift=v)),
        ("lookback", [5, 10, 15, 20, 25, 30], lambda v: dict(lookback=v)),
        ("max_weight", [0.02, 0.03, 0.04, 0.05, 0.10, 0.15], lambda v: dict(max_weight=v)),
    ]
    rows, check = [], []
    for k, (tr_s, tr_e, te_s, te_e) in enumerate(FOLDS, start=1):
        fold = f"Fold{k}"
        tr_mask = (dates_ret >= pd.Timestamp(tr_s)) & (dates_ret <= pd.Timestamp(tr_e))
        hmm = HMMRegimeDetector(n_states=3, seed=SEED)
        hmm.fit(nifty_ser.reindex(dates_ret[tr_mask]).fillna(0).values)
        reg = np.zeros(len(dates), dtype=np.int32)
        reg[1:] = hmm.predict_causal(nifty_ser.reindex(dates_ret).fillna(0).values)
        ckpt = CKPT_DIR / f"stgat_fold{k}.pt"
        if not ckpt.exists():
            print(f"  SKIP {fold}: no checkpoint"); continue
        model = STGATDPOModel(n_stocks=N, n_features=N_FEATURES, lookback=LOOKBACK, lstm_hidden=64,
                              lstm_layers=2, gat_hidden=64, gat_heads=2, dropout=0.0,
                              max_weight=MAX_WEIGHT).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE)); model.eval()
        base = evaluate_v2(model, feat_arr, ret_full, nifty_ser, reg, sector_adj, dates, te_s, te_e)
        check.append((fold, main_ir.get(fold, np.nan), base["info_ratio"]))
        print(f"  {fold}: baseline IR v2 = {base['info_ratio']:.4f}   main = {main_ir.get(fold, float('nan')):.4f}")
        for name, vals, fn in sweeps:
            for v in vals:
                met = evaluate_v2(model, feat_arr, ret_full, nifty_ser, reg, sector_adj, dates, te_s, te_e, **fn(v))
                met.update({"sensitivity": name, "param": v, "fold": fold}); rows.append(met)

    df = pd.DataFrame(rows); df.to_csv(OUT_DIR / "sensitivity_results_allfolds_v2.csv", index=False)
    agg = (df.groupby(["sensitivity", "param"])
             .agg(mean_IR=("info_ratio", "mean"), std_IR=("info_ratio", "std"),
                  min_IR=("info_ratio", "min"), max_IR=("info_ratio", "max"),
                  mean_traded=("mean_traded", "mean")).reset_index())
    agg["cost_drag_bps_per_year"] = agg["mean_traded"] * 10 * 12
    agg.to_csv(OUT_DIR / "sensitivity_crossfold_summary_v2.csv", index=False)
    print("\nBASELINE VALIDATION (v2 trained configuration vs main fold results)")
    cdf = pd.DataFrame(check, columns=["fold", "main_IR", "v2_IR"]); cdf["diff"] = cdf.v2_IR - cdf.main_IR
    print(cdf.round(4).to_string(index=False))
    print(f"  mean main = {cdf.main_IR.mean():.4f}   mean v2 = {cdf.v2_IR.mean():.4f}   max |diff| = {cdf['diff'].abs().max():.4f}")
    print("\nCROSS-FOLD SUMMARY"); print(agg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
