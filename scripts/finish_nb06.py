"""Completes notebook 06's remaining steps WITHOUT Jupyter (the ipykernel
running notebook 06 kept dying at the monitor-baselining cell in this
environment - see notebooks/06_monitoring.ipynb, whose cells are equivalent).

Steps (idempotent): data-quality baseline job -> poll, three monitoring
schedules, CloudWatch alarms + dashboard, report-artifact listing.
Assumes the monitoring endpoint is InService and capture data exists
(both done by notebook 06's early cells).

Run:  .venv-sm/bin/python scripts/finish_nb06.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import boto3
import sagemaker
from sagemaker.model_monitor import (
    CronExpressionGenerator,
    DefaultModelMonitor,
    ModelBiasMonitor,
    ModelQualityMonitor,
)
from sagemaker.model_monitor.dataset_format import DatasetFormat
from sagemaker.clarify import BiasConfig, ModelConfig
from sagemaker.session import Session

from aws_common import check_credentials, region

ENDPOINT = "member-utilization-risk-monitor"
MODEL = "member-utilization-risk-champion"
MON = "aai-540-g1/monitoring"
SCHEDULES = ["aai540-g1-data-quality", "aai540-g1-model-quality", "aai540-g1-model-bias"]

check_credentials()
region_name = region()
bs = boto3.Session(region_name=region_name)
sm = bs.client("sagemaker", region_name=region_name)
cw = bs.client("cloudwatch", region_name=region_name)
s3 = bs.client("s3", region_name=region_name)
sess = Session(boto_session=bs, sagemaker_client=sm)
bucket = sess.default_bucket()

try:
    role = sagemaker.get_execution_role()
except Exception:
    iam = boto3.client("iam", region_name=region_name)
    by_name = {r["RoleName"]: r for r in iam.list_roles()["Roles"]}
    picked = ("aai-540-g1-execution-role" if "aai-540-g1-execution-role" in by_name
              else sorted((n for n in by_name
                           if n.startswith("AmazonSageMaker-ExecutionRole")),
                          key=lambda n: by_name[n]["CreateDate"])[-1])
    account = boto3.client("sts").get_caller_identity()["Account"]
    role = f"arn:aws:iam::{account}:role/{picked}"

# --- 1. data-quality baseline (processing job), polled with prints -------
existing = {j["ProcessingJobName"]
            for j in sm.list_processing_jobs(SortBy="CreationTime",
                                             SortOrder="Descending")["ProcessingJobSummaries"]}
if any("baseline-suggestion" in n or "data-quality" in n for n in existing):
    print("baseline job already exists - skipping")
else:
    monitor = DefaultModelMonitor(role=role, instance_count=1,
                                  instance_type="ml.m5.xlarge",
                                  volume_size_in_gb=20, max_runtime_in_seconds=1800)
    monitor.suggest_baseline(
        baseline_dataset=f"s3://{bucket}/{MON}/baseline/baseline_train.csv",
        dataset_format=DatasetFormat.csv(header=False),
        output_s3_uri=f"s3://{bucket}/{MON}/data-baseline",
        wait=False, logs=False)
    job = monitor.latest_job.name
    print("baseline job submitted:", job, flush=True)
    while True:
        pj = sm.describe_processing_job(ProcessingJobName=job)
        status = pj["ProcessingJobStatus"]
        print(f"  baseline: {status}", flush=True)
        if status == "Completed":
            break
        if status == "Failed":
            sys.exit(f"baseline failed: {pj.get('FailureReason')}")
        time.sleep(20)

# --- 2. monitoring schedules (idempotent) ----------------------------------
existing_sched = {s["MonitoringScheduleName"]
                  for s in sm.list_monitoring_schedules()["MonitoringScheduleSummaries"]}

# data-quality schedule, wired to the baseline job's statistics/constraints
if "aai540-g1-data-quality" not in existing_sched:
    try:
        stats_obj = next(o["Key"] for o in
                         s3.list_objects_v2(Bucket=bucket, Prefix=f"{MON}/data-baseline/")
                         .get("Contents", []) if o["Key"].endswith("statistics.json"))
        constr_obj = next(o["Key"] for o in
                          s3.list_objects_v2(Bucket=bucket, Prefix=f"{MON}/data-baseline/")
                          .get("Contents", []) if o["Key"].endswith("constraints.json"))
        data_monitor = DefaultModelMonitor(role=role, instance_count=1,
                                           instance_type="ml.m5.xlarge",
                                           volume_size_in_gb=20,
                                           max_runtime_in_seconds=1800)
        data_monitor.create_monitoring_schedule(
            monitor_schedule_name="aai540-g1-data-quality",
            endpoint_input=ENDPOINT,
            output_s3_uri=f"s3://{bucket}/{MON}/data-quality",
            statistics=f"s3://{bucket}/{stats_obj}",
            constraints=f"s3://{bucket}/{constr_obj}",
            schedule_cron_expression=CronExpressionGenerator.hourly(),
            enable_cloudwatch_metrics=True)
        print("data-quality schedule created")
    except Exception as exc:
        print("data-quality schedule:", str(exc)[:150])
else:
    print("data-quality schedule exists")

# quality monitor with production ground truth
if "aai540-g1-model-quality" not in existing_sched:
    try:
        qm = ModelQualityMonitor(role=role, instance_count=1,
                                 instance_type="ml.m5.xlarge",
                                 volume_size_in_gb=20, max_runtime_in_seconds=1800)
        qm.create_monitoring_schedule(
            monitor_schedule_name="aai540-g1-model-quality",
            endpoint_input=sagemaker.model_monitor.EndpointInput(
                endpoint_name=ENDPOINT,
                destination=f"s3://{bucket}/{MON}/quality/input",
                start_time_offset="-PT1H", end_time_offset="-PT0H"),
            ground_truth_input=f"s3://{bucket}/{MON}/ground_truth",
            problem_type="MulticlassClassification",
            inference_attribute="0", probability_attribute="0",
            probability_threshold_attribute="0.5",
            output_s3_uri=f"s3://{bucket}/{MON}/quality",
            schedule_cron_expression=CronExpressionGenerator.hourly())
        print("model-quality schedule created")
    except Exception as exc:
        print("model-quality schedule:", str(exc)[:120])
else:
    print("model-quality schedule exists")

# bias monitor (facet: gender one-hot)
if "aai540-g1-model-bias" not in existing_sched:
    try:
        bm = ModelBiasMonitor(role=role, instance_count=1,
                              instance_type="ml.m5.xlarge",
                              volume_size_in_gb=20, max_runtime_in_seconds=1800)
        bm.create_monitoring_schedule(
            monitor_schedule_name="aai540-g1-model-bias",
            endpoint_input=ENDPOINT,
            bias_config=BiasConfig(label_values_or_threshold=["Low", "Medium", "High"],
                                   facet_name="gender_Male", facet_values_or_threshold=[1]),
            model_config=ModelConfig(model_name=MODEL, instance_type="ml.m5.xlarge",
                                     instance_count=1, content_type="text/csv",
                                     accept_type="text/csv"),
            output_s3_uri=f"s3://{bucket}/{MON}/bias",
            schedule_cron_expression=CronExpressionGenerator.hourly())
        print("model-bias schedule created")
    except Exception as exc:
        print("model-bias schedule:", str(exc)[:120])
else:
    print("model-bias schedule exists")

# --- 3. infra alarms + dashboard -------------------------------------------
for alarm_name, metric_name, stat, threshold in [
        ("aai540-g1-endpoint-5xx", "ModelInvocation5XXErrors", "Sum", 1),
        ("aai540-g1-endpoint-latency", "ModelLatency", "Average", 30000)]:
    cw.put_metric_alarm(
        AlarmName=alarm_name, Namespace="AWS/SageMaker", MetricName=metric_name,
        Dimensions=[{"Name": "EndpointName", "Value": ENDPOINT}],
        Statistic=stat, Period=300, EvaluationPeriods=1, Threshold=threshold,
        ComparisonOperator="GreaterThanOrEqualToThreshold")
print("infra alarms created")

dashboard = {"widgets": [
    {"type": "metric", "x": 0, "y": 0, "width": 12, "height": 6,
     "properties": {"title": "Endpoint invocations + 5XX",
                    "metrics": [["AWS/SageMaker", "Invocations", "EndpointName", ENDPOINT],
                                [".", "ModelInvocation5XXErrors", ".", "."]],
                    "period": 300, "stat": "Sum", "region": region_name}},
    {"type": "metric", "x": 12, "y": 0, "width": 12, "height": 6,
     "properties": {"title": "Endpoint latency (ms)",
                    "metrics": [["AWS/SageMaker", "ModelLatency", "EndpointName", ENDPOINT]],
                    "period": 300, "stat": "Average", "region": region_name}},
    {"type": "metric", "x": 0, "y": 6, "width": 12, "height": 6,
     "properties": {"title": "Data-quality violations",
                    "metrics": [["AWS/SageMaker/ModelMonitoring", "FeaturesViolated",
                                 "MonitoringSchedule", "aai540-g1-data-quality"]],
                    "period": 3600, "stat": "Average", "region": region_name}},
]}
cw.put_dashboard(DashboardName="aai540-g1-monitoring", DashboardBody=json.dumps(dashboard))
print("dashboard aai540-g1-monitoring published")

# --- 4. report artifacts ----------------------------------------------------
objs = s3.list_objects_v2(Bucket=bucket, Prefix=f"{MON}/data-baseline/").get("Contents", [])
print(f"baseline report artifacts: {len(objs)} objects under {MON}/data-baseline/")
print("\nDONE - run scripts/teardown_aws.py after screenshots to stop charges.")
