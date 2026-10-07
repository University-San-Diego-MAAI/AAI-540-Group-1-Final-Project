from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from .config import OUTPUTS_DIR, RANDOM_STATE, TARGET
from .data import build_feature_table
from .modeling import compare_models, evaluate, make_partitions, tune_best_model


def run_pipeline(
    dataset_dir: Path | None = None,
    outputs_dir: Path = OUTPUTS_DIR,
    chunksize: int | None = None,
) -> dict[str, object]:
    """Run feature construction, member-level splits, training, and evaluation."""
    data_path = dataset_dir
    if data_path is None:
        from .config import DATASET_DIR

        data_path = DATASET_DIR
    if chunksize is None:
        features, target_rates = build_feature_table(data_path)
    else:
        features, target_rates = build_feature_table(data_path, chunksize=chunksize)

    partitions = make_partitions(features, target_rates, random_state=RANDOM_STATE)
    processed_dir = outputs_dir / "processed"
    split_dir = outputs_dir / "splits"
    model_dir = outputs_dir / "model"
    evaluation_dir = outputs_dir / "evaluation"
    prediction_dir = outputs_dir / "predictions"
    for directory in (processed_dir, split_dir, model_dir, evaluation_dir, prediction_dir):
        directory.mkdir(parents=True, exist_ok=True)

    labeled = pd.concat(
        [partitions.train, partitions.validation, partitions.test], axis=0
    ).sort_index()
    labeled.to_csv(processed_dir / "member_features_labeled.csv", index_label="person_id")
    partitions.train.to_csv(split_dir / "train.csv", index_label="person_id")
    partitions.validation.to_csv(split_dir / "validation.csv", index_label="person_id")
    partitions.test.to_csv(split_dir / "test.csv", index_label="person_id")
    partitions.production.to_csv(
        split_dir / "production_features.csv", index_label="person_id"
    )

    comparison, fitted_models = compare_models(
        partitions.train, partitions.validation
    )
    comparison.to_csv(evaluation_dir / "model_comparison.csv", index=False)
    best_name = str(comparison.iloc[0]["model"])
    best_pipeline = tune_best_model(
        best_name, partitions.train, fitted_models[best_name]
    )
    metrics, report, matrix = evaluate(best_pipeline, partitions.test)
    report.to_csv(evaluation_dir / "test_classification_report.csv")
    matrix.to_csv(evaluation_dir / "test_confusion_matrix.csv")
    with (evaluation_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    joblib.dump(best_pipeline, model_dir / "best_model.joblib")

    production_predictions = best_pipeline.predict(partitions.production)
    pd.DataFrame(
        {
            TARGET: production_predictions,
        },
        index=partitions.production.index,
    ).to_csv(prediction_dir / "production_predictions.csv", index_label="person_id")

    metadata = {
        "selected_model": best_name,
        "best_parameters": best_pipeline.named_steps["classifier"].get_params(),
        "target": TARGET,
        "target_definition": (
            "Training-tercile bands of annualized visit counts in the final "
            "third of each member observation period."
        ),
        "training_target_rate_edges": list(partitions.class_edges),
        "random_state": RANDOM_STATE,
        "partition_sizes": {
            "train": len(partitions.train),
            "validation": len(partitions.validation),
            "test": len(partitions.test),
            "production": len(partitions.production),
        },
        "test_metrics": metrics,
    }
    with (model_dir / "model_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2, default=str)
    print("Model comparison (validation):")
    print(comparison.to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Test metrics: {metrics}")
    print(f"Artifacts saved under: {outputs_dir}")
    return {
        "comparison": comparison,
        "metrics": metrics,
        "model_path": model_dir / "best_model.joblib",
        "partitions": partitions,
    }
