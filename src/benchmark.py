"""Benchmark baselines for the utilization-risk model (Team Update 4.1 step 1).

The benchmark must be SIMPLE - a heuristic a care analyst could replicate:
  B1  majority-class predictor (predict the train-split majority for everyone)
  B2  1-feature rule: history-window total_visits terciles -> Low/Med/High
      (rank-based, same tie-safe approach as the labeling rule)

Both are evaluated on the held-out TEST split with the same metric set as the
real models (src/model_common.multiclass_metrics), so notebook 04's
model-vs-benchmark comparison is apples-to-apples.

Outputs (default data-dir sibling artifacts/evaluation/):
  benchmark_metrics.json           both baselines' test metrics
  benchmark_test_predictions.csv   per-member B2 predictions for error analysis

Usage:
  .venv/bin/python src/benchmark.py --table data/processed/training_table_1k.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from model_common import (
        LABEL_COLUMN,
        RISK_CLASSES,
        encode_features,
        load_table,
        multiclass_metrics,
        split_mask,
    )
except ImportError:
    from src.model_common import (
        LABEL_COLUMN,
        RISK_CLASSES,
        encode_features,
        load_table,
        multiclass_metrics,
        split_mask,
    )


def visits_tercile_rule(history_visits: pd.Series) -> pd.Series:
    """B2: rank-terciles of history total_visits mapped to Low/Medium/High."""
    pct = history_visits.rank(pct=True)
    return pd.Series(
        np.where(pct <= 1 / 3, "Low", np.where(pct <= 2 / 3, "Medium", "High")),
        index=history_visits.index,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default="data/processed/training_table.csv")
    parser.add_argument("--output-dir",
                        default=None,
                        help="default: <table dir>/../../artifacts/evaluation")
    args = parser.parse_args()

    table_path = Path(args.table)
    output_dir = (Path(args.output_dir) if args.output_dir
                  else table_path.parent.parent.parent / "artifacts" / "evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    table = load_table(table_path)
    _, _, audit = encode_features(table)  # encoding unused by baselines; validates schema
    test_idx = split_mask(audit, "test")
    y_test = audit.loc[test_idx, LABEL_COLUMN].reset_index(drop=True)
    person_ids = audit.loc[test_idx, "person_id"].reset_index(drop=True)

    y_train = audit.loc[split_mask(audit, "train"), LABEL_COLUMN]

    metrics = {"table": str(table_path), "test_size": int(test_idx.sum())}

    # --- B1: majority class ---
    majority = y_train.value_counts().idxmax()
    b1_pred = pd.Series(majority, index=y_test.index)
    metrics["b1_majority_class"] = {
        "predicted_class": majority,
        **multiclass_metrics(y_test, b1_pred.to_numpy(), _one_hot(b1_pred)),
    }

    # --- B2: history total_visits rank-terciles (whole eligible population) ---
    b2_all = visits_tercile_rule(table["total_visits"])
    b2_pred = b2_all.iloc[test_idx].reset_index(drop=True)
    metrics["b2_visits_tercile_rule"] = multiclass_metrics(
        y_test, b2_pred.to_numpy(), _one_hot(b2_pred))

    metrics_path = output_dir / "benchmark_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"wrote {metrics_path}")
    print(json.dumps({k: v for k, v in metrics.items()
                      if k in ("b1_majority_class", "b2_visits_tercile_rule")},
                     indent=2)[:1200])

    pred_path = output_dir / "benchmark_test_predictions.csv"
    pd.DataFrame({
        "person_id": person_ids,
        "y_true": y_test,
        "y_pred_b1_majority": majority,
        "y_pred_b2_visits_rule": b2_pred,
    }).to_csv(pred_path, index=False)
    print(f"wrote {pred_path}")


def _one_hot(pred: pd.Series) -> np.ndarray:
    proba = np.zeros((len(pred), len(RISK_CLASSES)))
    for i, cls in enumerate(RISK_CLASSES):
        proba[:, i] = (pred.to_numpy() == cls).astype(float)
    return proba


if __name__ == "__main__":
    main()
