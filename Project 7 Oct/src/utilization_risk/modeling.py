from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from .config import CLASSES, RANDOM_STATE, TARGET


class LabelEncodingClassifier(ClassifierMixin, BaseEstimator):
    """Encode string classes for estimators that require integer targets."""

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, features: object, target: pd.Series) -> LabelEncodingClassifier:
        self.label_encoder_ = LabelEncoder()
        encoded_target = self.label_encoder_.fit_transform(target)
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(features, encoded_target)
        self.classes_ = self.label_encoder_.classes_
        return self

    def predict(self, features: object) -> np.ndarray:
        encoded_predictions = self.estimator_.predict(features)
        return self.label_encoder_.inverse_transform(
            np.asarray(encoded_predictions, dtype=int)
        )

    def predict_proba(self, features: object) -> np.ndarray:
        return self.estimator_.predict_proba(features)


try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "XGBoost is required for model comparison. Install project dependencies "
        "with `python -m pip install -r requirements.txt`."
    ) from exc


@dataclass
class DataPartitions:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    production: pd.DataFrame
    class_edges: tuple[float, float]


def _classify_rates(rates: pd.Series, edges: tuple[float, float]) -> pd.Series:
    labels = np.select(
        [rates <= edges[0], rates <= edges[1]],
        [CLASSES[0], CLASSES[1]],
        default=CLASSES[2],
    )
    return pd.Series(labels, index=rates.index, name=TARGET)


def make_partitions(
    features: pd.DataFrame,
    annualized_visits: pd.Series,
    random_state: int = RANDOM_STATE,
) -> DataPartitions:
    """Split members 40/10/10/40 and learn proxy-label edges on train only."""
    if not features.index.is_unique:
        raise ValueError("Member index must be unique to prevent split leakage.")
    if not features.index.equals(annualized_visits.index):
        annualized_visits = annualized_visits.reindex(features.index)
    valid = annualized_visits.notna() & np.isfinite(annualized_visits)
    features = features.loc[valid].copy()
    annualized_visits = annualized_visits.loc[valid].astype(float)
    if len(features) < 20:
        raise ValueError("At least 20 eligible members are required for the splits.")

    rng = np.random.default_rng(random_state)
    shuffled = rng.permutation(features.index.to_numpy())
    total = len(shuffled)
    train_end = int(total * 0.40)
    validation_end = train_end + int(total * 0.10)
    test_end = validation_end + int(total * 0.10)
    train_ids = shuffled[:train_end]
    validation_ids = shuffled[train_end:validation_end]
    test_ids = shuffled[validation_end:test_end]
    production_ids = shuffled[test_end:]

    train_rates = annualized_visits.loc[train_ids]
    q1, q2 = train_rates.quantile([1 / 3, 2 / 3]).to_numpy()
    if q1 >= q2:
        raise ValueError(
            "Training future-visit rates do not yield three distinct risk tiers."
        )
    edges = (float(q1), float(q2))

    def labeled(ids: np.ndarray) -> pd.DataFrame:
        partition = features.loc[ids].copy()
        partition[TARGET] = _classify_rates(annualized_visits.loc[ids], edges)
        return partition

    train = labeled(train_ids)
    validation = labeled(validation_ids)
    test = labeled(test_ids)
    production = features.loc[production_ids].copy()
    production = production.drop(columns=[TARGET], errors="ignore")
    return DataPartitions(train, validation, test, production, edges)


def make_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = list(
        features.select_dtypes(include=["object", "string", "category"]).columns
    )
    numeric = [column for column in features.columns if column not in categorical]
    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def candidate_models() -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "xgboost": LabelEncodingClassifier(XGBClassifier(
            objective="multi:softprob",
            num_class=len(CLASSES),
            eval_metric="mlogloss",
            tree_method="hist",
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    }


def compare_models(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Pipeline]]:
    x_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]
    x_validation = validation.drop(columns=[TARGET])
    y_validation = validation[TARGET]
    pipelines: dict[str, Pipeline] = {}
    rows: list[dict[str, float | str]] = []

    for name, estimator in candidate_models().items():
        pipeline = Pipeline(
            [
                ("preprocessor", make_preprocessor(x_train)),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_validation)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_validation, predictions),
                "f1_macro": f1_score(
                    y_validation, predictions, average="macro", zero_division=0
                ),
                "f1_weighted": f1_score(
                    y_validation, predictions, average="weighted", zero_division=0
                ),
            }
        )
        pipelines[name] = pipeline
    comparison = pd.DataFrame(rows).sort_values(
        ["f1_macro", "accuracy"], ascending=False
    )
    return comparison, pipelines


def tune_best_model(
    model_name: str, train: pd.DataFrame, base_pipeline: Pipeline
) -> Pipeline:
    """Tune a small grid with cross-validation on training members only."""
    grids: dict[str, list[dict[str, list[object]]]] = {
        "logistic_regression": [
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__class_weight": ["balanced", None],
            }
        ],
        "random_forest": [
            {
                "classifier__n_estimators": [100],
                "classifier__max_depth": [None, 20],
                "classifier__min_samples_leaf": [2],
            }
        ],
        "extra_trees": [
            {
                "classifier__n_estimators": [100],
                "classifier__max_depth": [None, 20],
                "classifier__min_samples_leaf": [2],
            }
        ],
        "decision_tree": [
            {
                "classifier__max_depth": [None, 12, 24],
                "classifier__min_samples_leaf": [1, 5],
                "classifier__criterion": ["gini", "entropy"],
            }
        ],
        "xgboost": [
            {
                "classifier__estimator__n_estimators": [100, 200],
                "classifier__estimator__max_depth": [3, 6],
                "classifier__estimator__learning_rate": [0.05, 0.1],
            }
        ],
    }
    search = GridSearchCV(
        base_pipeline,
        grids[model_name],
        scoring="f1_macro",
        cv=3,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    search.fit(train.drop(columns=[TARGET]), train[TARGET])
    return search.best_estimator_


def evaluate(
    model: Pipeline, partition: pd.DataFrame
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    actual = partition[TARGET]
    predicted = model.predict(partition.drop(columns=[TARGET]))
    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "f1_macro": float(
            f1_score(actual, predicted, average="macro", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(actual, predicted, average="weighted", zero_division=0)
        ),
    }
    report = pd.DataFrame(
        classification_report(
            actual,
            predicted,
            labels=list(CLASSES),
            output_dict=True,
            zero_division=0,
        )
    ).T
    matrix = pd.DataFrame(
        confusion_matrix(actual, predicted, labels=list(CLASSES)),
        index=pd.Index(CLASSES, name="actual"),
        columns=pd.Index(CLASSES, name="predicted"),
    )
    return metrics, report, matrix
