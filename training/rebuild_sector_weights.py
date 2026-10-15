"""
training/rebuild_sector_weights.py  (nine-fold version)
Sector allocation of STGAT-DPO per walk-forward fold (mean weight across rebalance dates), using the model's SECTOR_MAP.
Writes outputs/results/sector_weights_rebuilt.csv (9 sectors x 9 folds). Run from the repository root.
"""
import ast, pandas as pd
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent; R = ROOT / "outputs" / "results"
SM = None
for node in ast.parse((ROOT / "models" / "gnn" / "stgat_dpo.py").read_text()).body:     # parse without importing torch
    if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "SECTOR_MAP" for t in node.targets): SM = ast.literal_eval(node.value)
MAIN = ["IT", "Banking", "FMCG", "Auto", "Pharma", "Energy", "Infrastructure", "Consumer"]
t2g = {t: (s if s in MAIN else "Other") for s, m in SM.items() for t in m}
groups = MAIN + ["Other"]; tab = {}
for i in range(1, 10):
    w = pd.read_csv(R / f"weights_Fold{i}.csv", index_col=0).mean(axis=0)
    tab[f"Fold{i}"] = {g: w[[t for t in w.index if t2g.get(t) == g]].sum() for g in groups}
out = pd.DataFrame(tab); out.to_csv(R / "sector_weights_rebuilt.csv"); print((out * 100).round(2).to_string())
