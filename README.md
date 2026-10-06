# AAI-540 Group 1 Final Project — Member Utilization-Risk Classifier

Multiclass ML system (Low / Medium / High utilization risk) for a hypothetical
UnitedHealthcare care-management scenario, built on the CMS DE-SynPUF synthetic
Medicare claims dataset (OMOP CDM v5.x, public AWS Open Data bucket
`s3://synpuf-omop`). The data is fully synthetic — all findings are
methodological exercises, never clinical conclusions.

## Pipeline (what lives where)

| Stage | Artifact | Notes |
|---|---|---|
| Raw datalake | `scripts/sync_raw_to_s3.py` | mirrors 17 OMOP tables (~915 MB) into `s3://<default-bucket>/aai-540-g1/raw/` |
| Validation + features | `src/load_synpuf.py`, `src/build_features.py` | 39-check gate; 100k-row member feature table |
| Training table | `src/build_training_data.py` | temporal split (2008 features / 2009-10 labels), death censoring, course 40/10/10/40 split |
| Athena | `scripts/setup_athena.py` | workgroup `aai540_g1`, external tables, validation queries |
| Feature Store | `notebooks/02_feature_store.ipynb` | `aai-540-g1-member-features` group, 82,624 records |
| Benchmark + training | `src/benchmark.py`, `src/train.py`, `notebooks/03_training.ipynb` | majority + visits-tercile baselines; LogReg / RF / XGBoost SageMaker jobs |
| Evaluation + registry | `notebooks/04_evaluation_registry.ipynb` | champion selection, fairness slices, model card; group `member-utilization-risk` |
| Batch scoring | `notebooks/05_batch_scoring.ipynb` | Batch Transform over the 40% production cohort → ranked worklist |
| Monitoring | `notebooks/06_monitoring.ipynb` | data/quality/bias monitors + CloudWatch dashboard (cleanup cell stops charges) |
| CI/CD | `src/pipeline.py` | SageMaker Pipeline: train → register (approval gate); `pipeline.upsert()` only |

**Results (held-out test):** benchmark visits-tercile macro F1 0.530; champion
XGBoost (eta 0.05, depth 6) macro F1 **0.774**, High-Risk recall **0.847**,
OvR ROC-AUC 0.917.

## Environment

1. Clone and create the virtualenv:

   ```bash
   git clone <repo-url> && cd final-project-usd
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```

2. Download the CMS DE-SynPUF OMOP CDM data (public buckets, no credentials; ~1 GB):

   ```bash
   aws s3 sync --no-sign-request s3://synpuf-omop/cmsdesynpuf1k/ data/raw/synpuf1k/
   aws s3 sync --no-sign-request s3://synpuf-omop/cmsdesynpuf100k/ data/raw/synpuf100k/
   ```

3. Rebuild the artifacts (each exits non-zero on check failure):

   ```bash
   .venv/bin/python src/load_synpuf.py        # validation gate
   .venv/bin/python src/build_features.py     # member_features.csv
   .venv/bin/python src/build_training_data.py  # training_table.csv (+ 1k)
   ```

4. AWS phase: place credentials in a repo-root `.env` (`AWS_ACCESS_KEY_ID`,
   `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`), verify with
   `scripts/check_aws_env.py`, then run the `scripts/` and `notebooks/`
   above. SageMaker SDK work uses the isolated `.venv-sm` (Python 3.9,
   `sagemaker>=2,<3`).
