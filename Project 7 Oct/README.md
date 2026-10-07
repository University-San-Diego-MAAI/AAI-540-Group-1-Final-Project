# Health Insurance Member Utilization Risk

This project builds a member-level, multiclass classification baseline from the
OMOP-style CSVs in `dataset/`. It uses pandas and scikit-learn only for data
processing and modeling; no database is required.

## Important target definition

The supplied data has **no observed utilization-risk label or paid-claim cost**.
The pipeline therefore creates a clearly marked proxy target:

1. Keep members with at least 365 days in their OMOP observation period.
2. Use the first two-thirds of each member's observation period as the feature
   window and the final third as the outcome window.
3. Count visits in that future window and annualize the count by observed days.
4. Learn the 33rd and 67th percentile cut points from the training partition
   only. Assign `Low`, `Medium`, or `High` risk by future visit-rate tercile.

This is a prediction of **relative future visit intensity**, not insurance
cost, medical necessity, or a validated clinical/actuarial risk score. It is
only a useful supervised-learning prototype if stakeholders accept this proxy.
For a production insurance risk model, replace it with a clinically/business
defined label such as future paid claims or allowed amount over a fixed horizon.

## Project layout

```text
dataset/                         Input CSV tables (unchanged)
notebooks/health_insurance_utilization_risk.ipynb
                                 Complete start-to-finish workflow (recommended)
notebooks/01_eda.ipynb           Focused EDA notebook
notebooks/02_modeling.ipynb      Focused modeling notebook
reports/                         Generated EDA report and dataset inventory
scripts/run_eda.py               Command-line EDA entry point
scripts/run_pipeline.py          Command-line modeling entry point
src/utilization_risk/
  config.py                      Paths, target and reproducibility constants
  data.py                        Cohort creation and chunked CSV aggregation
  modeling.py                    Splitting, preprocessing, model selection, scoring
  pipeline.py                    End-to-end orchestration and saved artifacts
  eda.py                         Chunked raw-data EDA
tests/                            Lightweight unit tests
outputs/                          Generated datasets, model, predictions and metrics
```

## Setup and run

Use Python 3.10 or newer. Install dependencies into the selected environment:

```powershell
python -m pip install -r requirements.txt
python scripts/run_eda.py
python scripts/run_pipeline.py
```

For the complete notebook workflow, open
`notebooks/health_insurance_utilization_risk.ipynb` and run all cells from top
to bottom. The focused notebooks are optional. The full CSVs are several
gigabytes, so EDA and feature creation read event files in chunks and may take a while.
Set a smaller `chunksize` in `src/utilization_risk/config.py` if memory is
limited.

## Merging and feature selection

`person_id` is the member key. Event-level records are never joined directly
to one another: doing so would multiply rows and inflate counts. Instead, the
pipeline creates one member-level row from `person` and `observation_period`,
then left-joins per-member event aggregates computed from:

- `visit_occurrence`
- `condition_occurrence`
- `procedure_occurrence`
- `drug_exposure`
- `measurement`
- `observation`
- `device_exposure`
- `payer_plan_period` (baseline period count)

The feature table includes age at the member-specific prediction date,
demographic concept codes, baseline duration known by that date and counts of
events strictly before that date. It excludes identifiers from the model,
total/future observation-period duration, event rows
after the prediction date, future visit counts, and post-index death. Era tables
are excluded because they duplicate condition/drug history; provider, care
site, location, and drug-strength lookup tables are not member-level predictive
features and would introduce high-cardinality or redundant joins. The EDA still
inventories every CSV in `dataset/`.

Members without events in a history table remain in the cohort and receive a
zero count. Invalid IDs/dates are excluded from event aggregates. Missing
demographics are imputed in the training pipeline rather than dropping large
parts of the cohort.

## Splits, preprocessing, and evaluation

Member IDs are assigned once to disjoint partitions: 40% training, 10%
validation, 10% test, and 40% production. The production export contains
features only. The numeric imputer and `StandardScaler` and categorical
imputer/one-hot encoder are fitted as part of each scikit-learn pipeline on the
training partition only, preventing preprocessing leakage.

Logistic Regression, Decision Tree, Random Forest, Extra Trees, and XGBoost
are compared using validation accuracy and macro-F1. A small cross-validated
grid search tunes the best validation model on training data. The selected
model is then evaluated once on the held-out test partition. Accuracy,
macro/weighted F1, per-class metrics, and a confusion matrix are saved. A
prediction file is also generated for the reserved production members, without
exposing their labels. XGBoost is an explicit project dependency.

Generated artifacts:

- `reports/eda_report.md`, `reports/dataset_inventory.csv`,
  `reports/column_missingness.csv`
- `outputs/processed/member_features_labeled.csv`
- `outputs/splits/train.csv`, `validation.csv`, `test.csv`,
  `production_features.csv`
- `outputs/model/best_model.joblib`, `model_metadata.json`
- `outputs/evaluation/model_comparison.csv`, `test_metrics.json`,
  `test_classification_report.csv`, `test_confusion_matrix.csv`
- `outputs/predictions/production_predictions.csv`

## Limitations

The relative tercile labels change with the training cohort. The observational
periods have different lengths, so outcome visits are annualized; results still
depend on the completeness and coding quality of the source data. This dataset
contains highly incomplete fields and is not a substitute for validation on
real plan claims, a prospective/time-based external test, fairness review, or
clinical/actuarial governance. Never use this proxy model to make individual
coverage or care decisions.
