"""Train a utilization-risk classifier (Task 10) - runs locally AND in SageMaker.

Models (Team Update 4.1 step 2 / design doc Model Training):
  logreg  multinomial LogisticRegression (saga, L2) - interpretable baseline
  rf      RandomForest (class_weight=balanced)      - nonlinear comparison
  xgb     XGBoost gradient-boosted trees (PRIMARY)  - multi:softprob, num_class=3

Data layout: a single training_table.csv containing the course-mandated
``split`` column (train/validation/test/production). Local runs point
``--data-dir`` at ``data/processed``; in SageMaker the same file is provided
through the ``train`` channel (i.e. /opt/ml/input/data/train/training_table.csv)
and outputs land in /opt/ml/model with metrics/predictions alongside.

Preprocessing is identical for every model (src/model_common.encode_features):
one-hot nominals fixed on the TRAIN split, race excluded from features
(fairness-audit slice only). XGBoost early-stops on the validation split.

Outputs (in --output-dir):
  <model>_metrics.json            validation + test metrics (macro F1 primary,
                                  High-Risk recall headline)
  <model>_test_predictions.csv    person_id, y_true, y_pred, proba_{Low,Med,High}
  model.<bin|joblib>              fitted model artifact

Usage (local smoke):
  .venv/bin/python src/train.py --model xgb --data-dir data/processed \
      --table training_table_1k.csv --output-dir artifacts/models/xgb_1k
"""

import argparse
import json
import os
from pathlib import Path
from typing import Tuple

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


def _resolve_paths(args) -> Tuple[Path, Path]:
    data_dir = Path(args.data_dir)
    candidates = [
        data_dir / args.table,
        Path("/opt/ml/input/data/train") / args.table,  # SageMaker channel
        data_dir / "training_table.csv",
    ]
    table_path = next((p for p in candidates if p.is_file()), None)
    if table_path is None:
        raise FileNotFoundError(f"training table not found in: {candidates}")
    output_dir = Path(os.getenv("SM_MODEL_DIR", args.output_dir or "artifacts/models/run"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return table_path, output_dir


def _build_estimator(args, seed: int):
    if args.model == "logreg":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(
            solver="saga", C=args.C,
            max_iter=5000, random_state=seed,
        )
    if args.model == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth or None,
            min_samples_leaf=args.min_samples_leaf, class_weight="balanced",
            n_jobs=-1, random_state=seed,
        )
    if args.model == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            objective="multi:softprob", num_class=3,
            eta=args.eta, max_depth=args.max_depth or 6,
            min_child_weight=args.min_child_weight,
            subsample=args.subsample, colsample_bytree=args.colsample_bytree,
            eval_metric="mlogloss", tree_method="hist",
            random_state=seed, n_jobs=-1,
        )
    raise ValueError(f"unknown --model {args.model!r}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["logreg", "rf", "xgb"], required=True)
    p.add_argument("--data-dir", default="data/processed")
    p.add_argument("--table", default="training_table.csv")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    # design-doc hyperparameter grid (single values here; the grid sweep runs
    # via the notebook / HyperparameterTuner)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--max-depth", type=int, default=6)
    p.add_argument("--min-samples-leaf", type=int, default=5)
    p.add_argument("--eta", type=float, default=0.1)
    p.add_argument("--min-child-weight", type=float, default=5.0)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    args = p.parse_args()

    table_path, output_dir = _resolve_paths(args)
    print(f"table: {table_path} -> output: {output_dir}")
    table = load_table(table_path)

    X_all, feature_names, audit = encode_features(table)
    train_idx = split_mask(audit, "train")
    val_idx = split_mask(audit, "validation")
    test_idx = split_mask(audit, "test")

    X_train, y_train = X_all[train_idx], audit.loc[train_idx, LABEL_COLUMN]
    X_val, y_val = X_all[val_idx], audit.loc[val_idx, LABEL_COLUMN]
    X_test, y_test = X_all[test_idx], audit.loc[test_idx, LABEL_COLUMN]

    # Fit on integer-coded labels (Low=0, Medium=1, High=2): XGBoost requires
    # 0..num_class-1 when num_class is set explicitly, and sklearn accepts the
    # same encoding - one convention for every model. Metrics still use the
    # original string labels via RISK_CLASSES indexing.
    y_train_fit = pd.Categorical(y_train, categories=RISK_CLASSES).codes
    y_val_fit = pd.Categorical(y_val, categories=RISK_CLASSES).codes

    model = _build_estimator(args, args.seed)
    fit_kwargs = {}
    if args.model == "xgb":
        fit_kwargs = dict(eval_set=[(X_val, y_val_fit)], verbose=False)
    model.fit(X_train, y_train_fit, **fit_kwargs)

    # With coded labels, predict_proba columns follow sorted-class order
    # (0, 1, 2) - which IS the RISK_CLASSES order; keep a defensive reorder
    # through model.classes_ so a string-labelled estimator would also work.
    class_order = [int(list(model.classes_).index(i))
                   if i in list(model.classes_) else int(i)
                   for i in range(len(RISK_CLASSES))]

    def predict(X):
        proba = model.predict_proba(X)[:, class_order]
        pred = np.array(RISK_CLASSES)[proba.argmax(axis=1)]
        return pred, proba

    val_pred, val_proba = predict(X_val)
    test_pred, test_proba = predict(X_test)

    metrics = {
        "model": args.model,
        "seed": args.seed,
        "hyperparameters": {k: getattr(args, k) for k in
                            ["C", "n_estimators", "max_depth", "min_samples_leaf",
                             "eta", "min_child_weight", "subsample",
                             "colsample_bytree"]},
        "n_features": int(X_all.shape[1]),
        "n_train": int(train_idx.sum()),
        "n_validation": int(val_idx.sum()),
        "n_test": int(test_idx.sum()),
        "validation": multiclass_metrics(y_val.reset_index(drop=True), val_pred, val_proba),
        "test": multiclass_metrics(y_test.reset_index(drop=True), test_pred, test_proba),
    }

    metrics_path = output_dir / f"{args.model}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(json.dumps({"validation": metrics["validation"], "test": metrics["test"]},
                     indent=2)[:1500])
    print(f"wrote {metrics_path}")

    pred_path = output_dir / f"{args.model}_test_predictions.csv"
    pd.DataFrame({
        "person_id": audit.loc[test_idx, "person_id"].reset_index(drop=True),
        "y_true": y_test.reset_index(drop=True),
        "y_pred": test_pred,
        **{f"proba_{cls}": test_proba[:, i] for i, cls in enumerate(RISK_CLASSES)},
    }).to_csv(pred_path, index=False)
    print(f"wrote {pred_path}")

    artifact = output_dir / ("model.bin" if args.model == "xgb" else "model.joblib")
    if args.model == "xgb":
        model.save_model(str(artifact))
    else:
        import joblib  # lazy: the XGBoost container ships without joblib

        joblib.dump(model, artifact)
    print(f"wrote {artifact}")


if __name__ == "__main__":
    main()
