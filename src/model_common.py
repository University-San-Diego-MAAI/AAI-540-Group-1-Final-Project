"""Shared preprocessing + evaluation helpers for the modeling phase.

Used by ``src/benchmark.py`` (Task 9) and ``src/train.py`` (Task 10) so the
benchmark and the real models apply IDENTICAL feature transformations —
the fairness of the model-vs-benchmark comparison depends on it.

Encoding contract (inherited from build_features.py / build_training_data.py):
- row key ``person_id`` is never a feature;
- ``race`` is EXCLUDED from features (fairness-audit slice only, per the
  design doc) but returned alongside for sliced evaluation;
- nominal columns (age_band, gender, state) are one-hot encoded with
  categories fixed on the TRAIN split (unseen categories at scoring become
  all-zero columns, never a crash);
- all remaining feature columns are int counts/flags, used as-is;
- the label is ``risk_class`` (Low / Medium / High) and the split selector is
  ``split`` (train / validation / test / production).

Expects ``src/`` on sys.path (same pattern as the other src modules).
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

RISK_CLASSES = ["Low", "Medium", "High"]
ID_COLUMN = "person_id"
LABEL_COLUMN = "risk_class"
SPLIT_COLUMN = "split"
AUDIT_COLUMNS = ["gender", "race", "age_band"]  # kept for sliced evaluation
NOMINAL_FEATURES = ["age_band", "gender", "state"]
COUNT_FEATURES = [
    "age",
    "observation_months",
    "inpatient_visits",
    "outpatient_visits",
    "total_visits",
    "distinct_care_sites",
    "diabetes",
    "chf",
    "copd",
    "stroke",
    "cancer",
    "renal_disease",
    "drug_exposures",
    "distinct_drugs",
    "drug_eras",
    "distinct_conditions",
    "distinct_procedures",
]
DROP_COLUMNS = ["label_rule_version", "er_visits", "ethnicity"]


def load_table(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    return df


def encode_features(df: pd.DataFrame, train_columns: Optional[pd.Index] = None):
    """One-hot encode nominals; return (X float32 ndarray, feature_names, audit_df).

    When ``train_columns`` is provided (validation/test/production scoring),
    the frame is reindexed to exactly those columns — scoring cannot produce
    a different schema than training.
    """
    encoded = pd.get_dummies(df[NOMINAL_FEATURES].astype(str), dtype=np.int8)
    features = pd.concat(
        [df[COUNT_FEATURES].reset_index(drop=True), encoded.reset_index(drop=True)],
        axis=1,
    )
    if train_columns is not None:
        features = features.reindex(columns=train_columns, fill_value=0)
    feature_names = features.columns
    audit = df[[ID_COLUMN, LABEL_COLUMN, SPLIT_COLUMN,
                *AUDIT_COLUMNS]].reset_index(drop=True)
    return features.astype("float32").to_numpy(), feature_names, audit


def split_mask(audit: pd.DataFrame, split: str) -> np.ndarray:
    return (audit[SPLIT_COLUMN] == split).to_numpy()


def multiclass_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Macro F1 (primary), per-class precision/recall, confusion matrix, OvR ROC-AUC.

    ``y_proba`` arrives with columns in RISK_CLASSES order (Low/Medium/High),
    but sklearn's OvR ROC-AUC expects columns in sorted-class order — reorder
    via an integer encoding instead of trusting label conventions.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=RISK_CLASSES, zero_division=0
    )
    sorted_classes = sorted(RISK_CLASSES)
    y_true_idx = pd.Categorical(y_true, categories=sorted_classes).codes
    proba_sorted = y_proba[:, [RISK_CLASSES.index(c) for c in sorted_classes]]
    return {
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=RISK_CLASSES,
                                         average="macro")), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "per_class": {
            cls: {"precision": round(float(p), 4), "recall": round(float(r), 4),
                  "f1": round(float(f), 4)}
            for cls, p, r, f in zip(RISK_CLASSES, precision, recall, f1)
        },
        "high_risk_recall": round(float(recall[RISK_CLASSES.index("High")]), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred,
                                             labels=RISK_CLASSES).tolist(),
        "roc_auc_ovr": round(float(roc_auc_score(y_true_idx, proba_sorted,
                                                 multi_class="ovr")), 4),
    }
