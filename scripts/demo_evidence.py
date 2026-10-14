"""Evidence pass for the demo/recording: monitoring reports + screenshots fodder.

Rebuilds the gradeable artifacts (everything is idempotent; ~40-50 min of
mostly-waiting, < $2):

  1. Model Registry: group + package (from the demo model) + model card
  2. Feature group: create + ingest a 2,000-record sample (console screenshot)
  3. DATA monitor report: baseline statistics/constraints job -> S3
  4. MODEL monitor report: one-off Clarify bias job (best effort) + a
     model-quality report computed against the demo endpoint on the test split
  5. CI/CD DAG: pipeline success run + a deliberate-failure run (red DAG)

Run AFTER deploy_demo_stack.py:
  set -a && source .env && set +a && .venv-sm/bin/python scripts/demo_evidence.py
"""

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import boto3
import pandas as pd

from aws_common import check_credentials, region

ENDPOINT = "aai540-g1-demo-endpoint"
MODEL = "aai540-g1-demo-model"
GROUP = "member-utilization-risk"
FG_NAME = "aai-540-g1-member-features"
MON = "aai-540-g1/monitoring"
PREFIX = "aai-540-g1"


def _wait_job(sm, name, kind="training"):
    describe = sm.describe_training_job if kind == "training" else sm.describe_processing_job
    key = "TrainingJobStatus" if kind == "training" else "ProcessingJobStatus"
    while True:
        st = describe(TrainingJobName=name) if kind == "training" else describe(ProcessingJobName=name)
        s = st[key]
        print(f"  {name[:60]}: {s}", flush=True)
        if s in ("Completed", "Failed", "Stopped"):
            if s != "Completed":
                sys.exit(f"job {name} ended {s}: {st.get('FailureReason', '')[:200]}")
            return st


def registry(sm, bucket, role, region_name):
    try:
        sm.create_model_package_group(
            ModelPackageGroupName=GROUP,
            ModelPackageGroupDescription=(
                "Versioned lineage of multiclass classifiers predicting member "
                "utilization risk (Low/Medium/High) from 2008 history-window "
                "features on CMS DE-SynPUF (OMOP)."))
        print("registry group created")
    except sm.exceptions.ClientError as e:
        if "exist" not in str(e).lower():
            raise
    model_data = f"s3://{bucket}/{PREFIX}/artifacts/demo/model-clean.tar.gz"
    image = f"683313688378.dkr.ecr.{region_name}.amazonaws.com/sagemaker-xgboost:1.7-1"
    pkg = sm.create_model_package(
        ModelPackageGroupName=GROUP,
        ModelPackageDescription=("Champion XGBoost multiclass (eta 0.05, depth 6): "
                                 "test macro F1 0.774, High-Risk recall 0.847."),
        InferenceSpecification={
            "Containers": [{"Image": image, "ModelDataUrl": model_data,
                            "Framework": "XGBOOST", "FrameworkVersion": "1.7-1"}],
            "SupportedTransformInstanceTypes": ["ml.m5.xlarge"],
            "SupportedRealtimeInferenceInstanceTypes": ["ml.m5.xlarge"],
            "SupportedContentTypes": ["text/csv"],
            "SupportedResponseMIMETypes": ["text/csv"]},
        ModelApprovalStatus="PendingManualApproval")
    print("model package:", pkg["ModelPackageArn"])
    card = {"model_overview": {
                "model_description": ("XGBoost multiclass utilization-risk classifier "
                                      "(champion config, synthetic DE-SynPUF data)."),
                "model_creator": "Sourangshu Pal (AAI-540 Group 1)",
                "model_owner": "Sourangshu Pal (AAI-540 Group 1)",
                "model_artifact": [model_data],
                "algorithm_type": "XGBoost (gradient-boosted trees)",
                "problem_type": "Supervised multiclass classification"},
            "intended_uses": {
                "purpose_of_model": "Prioritize care-management outreach by predicted risk tier.",
                "intended_uses": ("Course demo pipeline. Not a diagnostic device; synthetic "
                                  "data means performance does not transfer to real members."),
                "risk_rating": "Medium",
                "explanations_for_risk_rating": "Educational use; human approval gates deployment."},
            "training_details": {
                "objective_function": {"function": "Minimize",
                                       "facet": "multiclass log loss (mlogloss)"},
                "training_observations": "SageMaker training job, ml.m5.2xlarge, seed 42."},
            "evaluation_details": [{"name": "holdout-test",
                                    "evaluation_observation": "macro F1 0.774; High recall 0.847."}],
            "additional_information": {
                "ethical_considerations": "Synthetic data; no PHI/PII.",
                "caveats_and_recommendations": "Labels are rule-based terciles (relative ranking).",
                "custom_details": {"split": "40/10/10/40", "champion_run": "xgb-e005-d6-w5"}}}
    try:
        sm.delete_model_card(ModelCardName=f"{GROUP}-card")
    except Exception:
        pass
    try:
        sm.create_model_card(ModelCardName=f"{GROUP}-card", ModelCardStatus="Draft",
                             Content=json.dumps(card))
        print("model card recreated")
    except Exception as exc:
        print("model card skipped (registry group+package are the graded evidence):",
              str(exc)[:140])


def feature_group_sample(sm, fs_runtime, bucket, role):
    try:
        sm.delete_feature_group(FeatureGroupName=FG_NAME)
        time.sleep(20)
    except Exception:
        pass
    from sagemaker.feature_store.feature_group import FeatureGroup
    import sagemaker
    from sagemaker.session import Session
    bs = boto3.Session(region_name=region())
    sess = Session(boto_session=bs,
                   sagemaker_client=sm,
                   sagemaker_featurestore_runtime_client=fs_runtime)
    df = pd.read_csv(ROOT / "data" / "processed" / "training_table.csv").head(2000).copy()
    df["person_id"] = df["person_id"].astype(str)
    df["event_time"] = pd.Timestamp.now(tz="UTC").timestamp()
    fg = FeatureGroup(name=FG_NAME, sagemaker_session=sess)
    fg.load_feature_definitions(data_frame=df)
    fg.create(s3_uri=f"s3://{bucket}/{PREFIX}/features", record_identifier_name="person_id",
              event_time_feature_name="event_time", role_arn=role, enable_online_store=True)
    status = fg.describe().get("FeatureGroupStatus")
    while status == "Creating":
        time.sleep(5)
        status = fg.describe().get("FeatureGroupStatus")
    records = df.astype(object).where(pd.notna(df), None).to_dict("records")
    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda rec: fs_runtime.put_record(
            FeatureGroupName=FG_NAME,
            Record=[{"FeatureName": k, "ValueAsString": str(v) if v is not None else "0"}
                    for k, v in rec.items()]), records, chunksize=64))
    print(f"feature group ready: {FG_NAME} ({len(records):,} records) - console screenshot now")


def data_monitor_report(sm, bucket, role):
    from sagemaker.model_monitor import DefaultModelMonitor
    from sagemaker.model_monitor.dataset_format import DatasetFormat
    import sagemaker
    from sagemaker.session import Session
    bs = boto3.Session(region_name=region())
    sess = Session(boto_session=bs, sagemaker_client=sm)
    monitor = DefaultModelMonitor(role=role, instance_count=1, instance_type="ml.m5.xlarge",
                                  volume_size_in_gb=20, max_runtime_in_seconds=1800)
    name = f"demo-baseline-{int(time.time())}"
    monitor.suggest_baseline(
        baseline_dataset=f"s3://{bucket}/{MON}/baseline/baseline_train.csv",
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=f"s3://{bucket}/{MON}/data-baseline",
        wait=False, logs=False, job_name=name)
    print("baseline job:", name)
    _wait_job(sm, name, kind="processing")


def model_reports(bs, bucket, s3):
    runtime = bs.client("sagemaker-runtime", region_name=region())
    table = pd.read_csv(ROOT / "data" / "processed" / "training_table.csv")
    test = table[table["split"] == "test"].head(400)
    from model_common import encode_features
    X_all, feature_names, audit = encode_features(table)
    mask = (audit["split"] == "test").to_numpy()
    X_test = pd.DataFrame(X_all[mask], columns=feature_names).head(400)

    preds = []
    for i in range(len(X_test)):
        body = ",".join(f"{v:.6g}" for v in X_test.iloc[i].to_numpy())
        resp = runtime.invoke_endpoint(EndpointName=ENDPOINT, ContentType="text/csv",
                                       Accept="application/json", Body=body)
        p = json.loads(resp["Body"].read().decode())["predictions"][0]
        if isinstance(p, dict):
            p = p.get("score") or list(p.values())[0]
        preds.append(["Low", "Medium", "High"][max(range(3), key=lambda j: p[j])])
        if i % 100 == 0:
            print(f"  quality probe {i}/{len(X_test)}", flush=True)

    from sklearn.metrics import classification_report, f1_score, confusion_matrix
    report = {
        "macro_f1": round(float(f1_score(test["risk_class"], preds, average="macro")), 4),
        "classification_report": classification_report(test["risk_class"], preds,
                                                       output_dict=True),
        "confusion_matrix": confusion_matrix(test["risk_class"], preds,
                                             labels=["Low", "Medium", "High"]).tolist(),
        "note": ("Live endpoint predictions vs held-out test labels - the model-quality "
                 "monitoring evidence (classic schedule APIs are in AWS maintenance mode).")}
    s3.put_object(Bucket=bucket, Key=f"{MON}/reports/model_quality_report.json",
                  Body=json.dumps(report, indent=2))
    html = (f"<html><body style='font-family:sans-serif'><h1>Model Quality Report</h1>"
            f"<p>Endpoint: {ENDPOINT} · sample: 400 held-out members · synthetic data</p>"
            f"<p><b>Macro F1: {report['macro_f1']}</b></p>"
            f"<pre>{json.dumps(report['classification_report'], indent=2)[:3000]}</pre>"
            f"</body></html>")
    s3.put_object(Bucket=bucket, Key=f"{MON}/reports/model_quality_report.html",
                  Body=html, ContentType="text/html")
    print("model quality report published to", f"{MON}/reports/")

    # one-off Clarify bias job (best effort - the classic bias monitor is maintenance-mode)
    try:
        from sagemaker.clarify import SageMakerClarifyProcessor, BiasConfig, DataConfig, ModelConfig
        import sagemaker as sm_sdk
        from sagemaker.session import Session
        sess = Session(boto_session=bs, sagemaker_client=sm)
        clarify = SageMakerClarifyProcessor(role=_role(), instance_count=1,
                                            instance_type="ml.m5.xlarge",
                                            sagemaker_session=sess)
        data_config = DataConfig(
            s3_data_input_path=f"s3://{bucket}/{PREFIX}/processed/training/training_table.csv",
            s3_output_path=f"s3://{bucket}/{MON}/bias-report",
            label="risk_class", dataset_type="text/csv",
            headers=list(pd.read_csv(ROOT / "data" / "processed" / "training_table.csv",
                                     nrows=1).columns))
        model_config = ModelConfig(model_name=MODEL, instance_type="ml.m5.xlarge",
                                   instance_count=1, content_type="text/csv",
                                   accept_type="text/csv")
        bias_config = BiasConfig(label_values_or_threshold=["Low", "Medium", "High"],
                                 facet_name="gender_Male", facet_values_or_threshold=[1])
        clarify.run_bias(data_config=data_config, model_config=model_config,
                         bias_config=bias_config, wait=True,
                         job_name=f"demo-bias-{int(time.time())}")
        print("clarify bias report in", f"{MON}/bias-report/")
    except Exception as e:
        s3.put_object(Bucket=bucket, Key=f"{MON}/bias-report/NOTE.txt",
                      Body=(f"One-off Clarify bias job could not run: {str(e)[:300]}\n"
                            f"Model-quality report above serves as the model-monitor evidence."))
        print("clarify bias skipped:", str(e)[:150])


def _role():
    import sagemaker
    try:
        return sagemaker.get_execution_role()
    except Exception:
        account = boto3.client("sts").get_caller_identity()["Account"]
        return f"arn:aws:iam::{account}:role/aai-540-g1-execution-role"


def pipeline_dags(sm, bucket, role):
    """One green + one red pipeline execution for the DAG screenshots."""
    import sagemaker
    from sagemaker import image_uris
    from sagemaker.inputs import TrainingInput
    from sagemaker.estimator import Estimator
    from sagemaker.workflow.pipeline import Pipeline
    from sagemaker.workflow.steps import TrainingStep
    from sagemaker.session import Session
    bs = boto3.Session(region_name=region())
    sess = Session(boto_session=bs, sagemaker_client=sm)
    image = image_uris.retrieve("xgboost", region(), version="1.7-1")
    base = dict(entry_point="train.py", source_dir="src", role=role,
                instance_count=1, instance_type="ml.m5.xlarge",
                output_path=f"s3://{bucket}/{PREFIX}/artifacts/pipeline",
                sagemaker_session=sess)

    def _pipe(name, eta):
        est = Estimator(image_uri=image, hyperparameters={
            "model": "xgb", "eta": eta, "max-depth": "6", "min-child-weight": "5"}, **base)
        step = TrainingStep(name="train", estimator=est,
                            inputs={"train": TrainingInput(
                                f"s3://{bucket}/{PREFIX}/processed/training/training_table.csv",
                                content_type="text/csv")})
        return Pipeline(name=name, steps=[step], sagemaker_session=sess)

    ok = _pipe("aai-540-g1-demo-pipeline-ok", "0.05")
    ok.upsert(role_arn=role)
    ex = ok.start()
    print("pipeline success run started:", ex.arn)
    while True:
        d = sm.describe_pipeline_execution(PipelineExecutionArn=ex.arn)
        print("  ok run:", d["PipelineExecutionStatus"], flush=True)
        if d["PipelineExecutionStatus"] in ("Succeeded", "Failed"):
            break
        time.sleep(30)

    bad = _pipe("aai-540-g1-demo-pipeline-fail", "99")   # invalid eta -> training fails
    bad.upsert(role_arn=role)
    ex2 = bad.start()
    print("pipeline failure run started:", ex2.arn)
    while True:
        d = sm.describe_pipeline_execution(PipelineExecutionArn=ex2.arn)
        print("  fail run:", d["PipelineExecutionStatus"], flush=True)
        if d["PipelineExecutionStatus"] in ("Succeeded", "Failed"):
            break
        time.sleep(30)
    print("screenshot now: SageMaker Studio -> Pipelines -> both DAGs (green + red)")


def main():
    check_credentials()
    r = region()
    bs = boto3.Session(region_name=r)
    sm = bs.client("sagemaker", region_name=r)
    s3 = bs.client("s3", region_name=r)
    fs_runtime = bs.client("sagemaker-featurestore-runtime", region_name=r)
    bucket = f"sagemaker-{r}-{boto3.client('sts').get_caller_identity()['Account']}"
    role = _role()

    # baseline input must exist before the data monitor report
    s3.upload_file(str(ROOT / "data" / "processed" / "training_table.csv"),
                   bucket, f"{PREFIX}/processed/training/training_table.csv")
    import pandas as pd
    t = pd.read_csv(ROOT / "data" / "processed" / "training_table.csv")
    t[t["split"] == "train"].drop(columns=["risk_class", "split", "label_rule_version"]) \
        .to_csv("/tmp/baseline_train.csv", index=False, header=False)
    s3.upload_file("/tmp/baseline_train.csv", bucket, f"{MON}/baseline/baseline_train.csv")

    print("== registry");           registry(sm, bucket, role, r)
    print("== feature group");      feature_group_sample(sm, fs_runtime, bucket, role)
    print("== data monitor report"); data_monitor_report(sm, bucket, role)
    print("== model reports");      model_reports(bs, bucket, s3)
    print("== pipeline DAGs");      pipeline_dags(sm, bucket, role)
    print("\nEVIDENCE PASS COMPLETE - see console/S3 for screenshots.")


if __name__ == "__main__":
    main()
