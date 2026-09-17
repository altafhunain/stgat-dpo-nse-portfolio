"""
analyze_regime_by_year.py

Diagnostic analysis: for each test year (2018-2025), computes the HMM
regime distribution, average India VIX, and regime transition frequency,
correlated against the ACTUAL test IC read directly from the fold
evaluation results CSV -- no manual transcription of numbers, everything
sourced from the real output file.

Usage:
    python analyze_regime_by_year.py \
        --regime-file regime_labels.parquet \
        --index-vix-csv index_vix.csv \
        --fold-summary fold_evaluation_summary_tb_v2.csv
"""

import argparse
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("analyze_regime")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime-file", required=True)
    parser.add_argument("--index-vix-csv", required=True)
    parser.add_argument("--fold-summary", required=True,
                         help="fold_evaluation_summary_tb_v2.csv -- test_ic read from HERE, not hardcoded")
    args = parser.parse_args()

    fold_summary = pd.read_csv(args.fold_summary)
    test_ic_by_year = fold_summary.groupby("test_year")["test_ic"].mean().to_dict()
    logger.info("Test IC by year (loaded directly from %s): %s", args.fold_summary, test_ic_by_year)

    regime = pd.read_parquet(args.regime_file).reset_index().rename(columns={"index": "date"})
    regime["date"] = pd.to_datetime(regime["date"])
    regime["year"] = regime["date"].dt.year

    vix = pd.read_csv(args.index_vix_csv, parse_dates=["date"])
    vix["year"] = vix["date"].dt.year

    rows = []
    for year in range(2018, 2026):
        year_regime = regime[regime["year"] == year]
        year_vix = vix[vix["year"] == year]

        if len(year_regime) == 0:
            continue

        regime_pct = year_regime["regime_name"].value_counts(normalize=True) * 100
        transitions = (year_regime["regime_name"] != year_regime["regime_name"].shift(1)).sum()

        rows.append({
            "year": year,
            "test_ic": test_ic_by_year.get(year),
            "pct_bull": regime_pct.get("Bull", 0),
            "pct_bear": regime_pct.get("Bear", 0),
            "pct_sideways": regime_pct.get("Sideways", 0),
            "n_regime_transitions": transitions,
            "avg_vix": year_vix["india_vix"].mean() if len(year_vix) else None,
            "max_vix": year_vix["india_vix"].max() if len(year_vix) else None,
            "vix_std": year_vix["india_vix"].std() if len(year_vix) else None,
        })

    df = pd.DataFrame(rows)
    logger.info("\n%s", df.to_string(index=False))

    logger.info("\nCorrelation of test_ic with regime/VIX characteristics:")
    corr_cols = ["pct_bull", "pct_bear", "pct_sideways", "n_regime_transitions", "avg_vix", "max_vix", "vix_std"]
    correlations = df[["test_ic"] + corr_cols].corr()["test_ic"].drop("test_ic")
    logger.info("\n%s", correlations.sort_values().to_string())

    df.to_csv("regime_analysis_by_year.csv", index=False)
    logger.info("\nSaved to regime_analysis_by_year.csv")


if __name__ == "__main__":
    main()
