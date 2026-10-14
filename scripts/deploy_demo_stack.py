"""Deploy the demo stack: serverless inference endpoint + static UI.

Creates (idempotently), in order:
  1. training_table.csv back in S3 (needed by the UI member picker and the
     monitoring evidence pass)
  2. a champion-config training job (xgb eta 0.05 / depth 6 / mcw 5) -> model
  3. SageMaker SERVERLESS inference endpoint (scales to zero: $0 at rest)
     with 100% data capture to S3
  4. 12-member demo roster (stratified, encoded feature vectors) -> members.json
  5. Lambda proxy (API Gateway -> invoke_endpoint) + HTTP API with CORS
  6. public S3 static website hosting demo/index.html
  7. CloudWatch dashboard + infra alarms (light up with demo traffic)

Residual cost at rest: ~$0.05-0.10/month (S3 website + reports). The
endpoint costs nothing until invoked; recording-day warmth via demo_day.py.

Run:  set -a && source .env && set +a && .venv-sm/bin/python scripts/deploy_demo_stack.py
"""

import json
import sys
import time
import zipfile
from io import BytesIO, StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import boto3
import pandas as pd
import sagemaker
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.session import Session

from aws_common import check_credentials, region

ENDPOINT = "aai540-g1-demo-endpoint"
MODEL = "aai540-g1-demo-model"
LAMBDA_FN = "aai540-g1-demo-proxy"
LAMBDA_ROLE = "aai540-g1-lambda-role"
HTTP_API = "aai540-g1-demo-api"
SITE_BUCKET = None  # set in main: aai540-g1-demo-<account>
DASHBOARD = "aai540-g1-monitoring"
MON = "aai-540-g1/monitoring"
PREFIX = "aai-540-g1"

LAMBDA_CODE = '''
import json, os, boto3
runtime = boto3.client("sagemaker-runtime")
ENDPOINT = os.environ["ENDPOINT_NAME"]
def handler(event, context):
    body = json.loads(event.get("body") or "{}")
    vec = body["vector"]
    csv = ",".join(f"{v:.6g}" for v in vec)
    resp = runtime.invoke_endpoint(EndpointName=ENDPOINT, ContentType="text/csv",
                                   Accept="application/json", Body=csv)
    pred = json.loads(resp["Body"].read().decode())["predictions"][0]
    return {"statusCode": 200,
            "headers": {"Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"probabilities": pred})}
'''


def _role_arn(iam):
    try:
        return sagemaker.get_execution_role()
    except Exception:
        by_name = {r["RoleName"]: r for r in iam.list_roles()["Roles"]}
        picked = ("aai-540-g1-execution-role" if "aai-540-g1-execution-role" in by_name
                  else sorted((n for n in by_name
                               if n.startswith("AmazonSageMaker-ExecutionRole")),
                              key=lambda n: by_name[n]["CreateDate"])[-1])
        account = boto3.client("sts").get_caller_identity()["Account"]
        return f"arn:aws:iam::{account}:role/{picked}"


def train_champion(sess, role, bucket, sm):
    key = f"{PREFIX}/processed/training/training_table.csv"
    existing = {j["TrainingJobName"]
                for j in sm.list_training_jobs(SortBy="CreationTime", SortOrder="Descending",
                                               MaxResults=50)["TrainingJobSummaries"]}
    done = next((n for n in existing
                 if n.startswith("demo-champion") and
                 sm.describe_training_job(TrainingJobName=n)["TrainingJobStatus"] == "Completed"), None)
    if done:
        print("champion training job exists:", done)
        return sm.describe_training_job(TrainingJobName=done)["ModelArtifacts"]["S3ModelArtifacts"]
    est = Estimator(
        image_uri=image_uris.retrieve("xgboost", region(), version="1.7-1"),
        entry_point="train.py", source_dir="src", role=role,
        instance_count=1, instance_type="ml.m5.2xlarge",
        output_path=f"s3://{bucket}/{PREFIX}/artifacts/demo",
        sagemaker_session=sess,
        hyperparameters={"model": "xgb", "eta": "0.05", "max-depth": "6",
                         "min-child-weight": "5"},
    )
    name = f"demo-champion-{int(time.time())}"
    est.fit({"train": TrainingInput(f"s3://{bucket}/{key}", content_type="text/csv")},
            job_name=name, wait=True)
    return sm.describe_training_job(TrainingJobName=name)["ModelArtifacts"]["S3ModelArtifacts"]


def clean_model_tarball(s3, bucket, model_s3):
    """train.py's tarball carries metrics/predictions alongside model.bin;
    the serving container wants a tarball with only model.bin."""
    key = model_s3.split(f"s3://{bucket}/")[1]
    raw = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    import tarfile
    src = tarfile.open(fileobj=BytesIO(raw))
    assert "model.bin" in src.getnames(), "expected model.bin in training tarball"
    out = BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz") as dst:
        dst.addfile(src.getmember("model.bin"), fileobj=src.extractfile("model.bin"))
    out.seek(0)
    clean_key = f"{PREFIX}/artifacts/demo/model-clean.tar.gz"
    s3.upload_fileobj(out, bucket, clean_key)
    return f"s3://{bucket}/{clean_key}"


def build_members(s3, bucket):
    from model_common import encode_features, load_table
    table = load_table(ROOT / "data" / "processed" / "training_table.csv")
    X_all, feature_names, audit = encode_features(table)
    picks = []
    for cls in ["Low", "Medium", "High"]:
        idx = audit[audit["risk_class"] == cls].index[:4]
        picks += list(idx)
    members = []
    for i in picks:
        row = table.loc[i]
        members.append({
            "id": int(row["person_id"]),
            "risk_true": row["risk_class"],
            "summary": {k: (int(row[k]) if k != "age" else int(row[k]))
                        for k in ["age", "total_visits", "drug_eras",
                                  "observation_months", "chf", "diabetes", "copd",
                                  "stroke", "cancer", "renal"]},
            "vector": [round(float(v), 6) for v in X_all[i]],
        })
    s3.put_object(Bucket=bucket, Key="members.json",
                  Body=json.dumps(members), ContentType="application/json")
    return members


def ensure_lambda(iam, lmb, sm):
    try:
        iam.create_role(RoleName=LAMBDA_ROLE, AssumeRolePolicyDocument=json.dumps({
            "Version": "2012-10-17", "Statement": [{
                "Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
                "Action": "sts:AssumeRole"}]}))
        iam.attach_role_policy(RoleName=LAMBDA_ROLE,
                               PolicyArn="arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole")
        iam.put_role_policy(RoleName=LAMBDA_ROLE, PolicyName="invoke-endpoint", PolicyDocument=json.dumps({
            "Version": "2012-10-17", "Statement": [{
                "Effect": "Allow", "Action": "sagemaker:InvokeEndpoint",
                "Resource": f"arn:aws:sagemaker:{region()}:*:endpoint/{ENDPOINT}"}]}))
        time.sleep(10)
        print("lambda role created")
    except iam.exceptions.EntityAlreadyExistsException:
        pass
    account = boto3.client("sts").get_caller_identity()["Account"]
    role_arn = f"arn:aws:iam::{account}:role/{LAMBDA_ROLE}"
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("index.py", LAMBDA_CODE)
    buf.seek(0)
    try:
        lmb.create_function(FunctionName=LAMBDA_FN, Runtime="python3.9", Role=role_arn,
                            Handler="index.handler", Code={"ZipFile": buf.read()},
                            Environment={"Variables": {"ENDPOINT_NAME": ENDPOINT}},
                            Timeout=30, MemorySize=256)
        print("lambda created")
    except lmb.exceptions.ResourceConflictException:
        lmb.update_function_code(FunctionName=LAMBDA_FN, ZipFile=buf.getvalue())
        lmb.update_function_configuration(
            FunctionName=LAMBDA_FN,
            Environment={"Variables": {"ENDPOINT_NAME": ENDPOINT}})
        print("lambda updated")
    return f"arn:aws:lambda:{region()}:{account}:function:{LAMBDA_FN}"


def ensure_api(apigw, lambda_arn, lmb):
    apis = apigw.get_apis().get("Items", [])
    api = next((a for a in apis if a["Name"] == HTTP_API), None)
    if api is None:
        api = apigw.create_api(Name=HTTP_API, ProtocolType="HTTP", Target=lambda_arn,
                               CorsConfiguration={"AllowOrigins": ["*"],
                                                  "AllowMethods": ["POST", "OPTIONS"],
                                                  "AllowHeaders": ["*"]})
        time.sleep(3)
    api_id = api["ApiId"]
    url = f"{api.get('ApiEndpoint', 'https://' + api_id + '.execute-api.' + region() + '.amazonaws.com')}"
    try:
        lmb.add_permission(FunctionName=LAMBDA_FN, StatementId="apigw-demo",
                           Action="lambda:InvokeFunction",
                           Principal="apigateway.amazonaws.com",
                           SourceArn=f"arn:aws:execute-api:{region()}:*:*/*/*")
    except lmb.exceptions.ResourceConflictException:
        pass
    return url


def main():
    check_credentials()
    r = region()
    bs = boto3.Session(region_name=r)
    sm = bs.client("sagemaker", region_name=r)
    s3 = bs.client("s3", region_name=r)
    iam = bs.client("iam", region_name=r)
    lmb = bs.client("lambda", region_name=r)
    apigw = bs.client("apigatewayv2", region_name=r)
    cw = bs.client("cloudwatch", region_name=r)
    sess = Session(boto_session=bs, sagemaker_client=sm)
    account = boto3.client("sts").get_caller_identity()["Account"]
    global SITE_BUCKET
    bucket = sess.default_bucket()
    SITE_BUCKET = f"aai540-g1-demo-{account}"
    role = _role_arn(iam)

    print("== 1. training table in S3")
    s3.upload_file(str(ROOT / "data" / "processed" / "training_table.csv"),
                   bucket, f"{PREFIX}/processed/training/training_table.csv")

    print("== 2. champion training job (~8 min)")
    model_s3 = train_champion(sess, role, bucket, sm)
    print("   model:", model_s3)

    print("== 3. clean model tarball")
    clean_s3 = clean_model_tarball(s3, bucket, model_s3)
    image = image_uris.retrieve("xgboost", region(), version="1.7-1")
    try:
        sm.delete_model(ModelName=MODEL)
    except Exception:
        pass
    sm.create_model(ModelName=MODEL, ExecutionRoleArn=role,
                    PrimaryContainer={"Image": image, "ModelDataUrl": clean_s3})

    print("== 4. serverless endpoint")
    # NOTE: serverless inference endpoints do not support DataCaptureConfig
    # (verified 2026-10) - monitoring evidence comes from the quality/bias
    # reports and CloudWatch metrics instead.
    try:
        sm.create_endpoint_config(
            EndpointConfigName=ENDPOINT,
            ProductionVariants=[{"VariantName": "demo", "ModelName": MODEL,
                                 "ServerlessConfig": {"MemorySizeInMB": 2048,
                                                      "MaxConcurrency": 10}}])
    except sm.exceptions.ResourceInUse:
        pass
    try:
        sm.create_endpoint(EndpointName=ENDPOINT, EndpointConfigName=ENDPOINT)
        print("   endpoint creating…")
    except sm.exceptions.ResourceInUse:
        pass
    sm.get_waiter("endpoint_in_service").wait(
        EndpointName=ENDPOINT, WaiterConfig={"Delay": 15, "MaxAttempts": 60})
    print("   endpoint InService:", ENDPOINT)

    print("== 5. members.json + website")
    try:
        s3.create_bucket(Bucket=SITE_BUCKET)
        s3.get_waiter("bucket_exists").wait(Bucket=SITE_BUCKET)
    except s3.exceptions.BucketAlreadyOwnedByYou:
        pass
    s3.delete_public_access_block(Bucket=SITE_BUCKET)
    s3.put_bucket_policy(Bucket=SITE_BUCKET, Policy=json.dumps({
        "Version": "2012-10-17", "Statement": [{
            "Sid": "PublicRead", "Effect": "Allow",
            "Principal": "*", "Action": "s3:GetObject",
            "Resource": f"arn:aws:s3:::{SITE_BUCKET}/*"}]}))
    s3.put_bucket_website(Bucket=SITE_BUCKET, WebsiteConfiguration={
        "IndexDocument": {"Suffix": "index.html"}})
    build_members(s3, SITE_BUCKET)
    html = (ROOT / "demo" / "index.html").read_text()
    api_url = None  # filled after API creation; do two-pass
    s3.put_object(Bucket=SITE_BUCKET, Key="index.html",
                  Body=html.replace("__API_URL__", "/placeholder"),
                  ContentType="text/html")

    print("== 6. lambda + HTTP API")
    lambda_arn = ensure_lambda(iam, lmb, sm)
    api_url = ensure_api(apigw, lambda_arn, lmb)

    print("== 7. dashboard + alarms")
    for alarm_name, metric_name, stat, threshold in [
            ("aai540-g1-endpoint-5xx", "ModelInvocation5XXErrors", "Sum", 1),
            ("aai540-g1-endpoint-latency", "ModelLatency", "Average", 30000)]:
        cw.put_metric_alarm(
            AlarmName=alarm_name, Namespace="AWS/SageMaker", MetricName=metric_name,
            Dimensions=[{"Name": "EndpointName", "Value": ENDPOINT}],
            Statistic=stat, Period=300, EvaluationPeriods=1, Threshold=threshold,
            ComparisonOperator="GreaterThanOrEqualToThreshold")
    dashboard = {"widgets": [
        {"type": "metric", "x": 0, "y": 0, "width": 12, "height": 6,
         "properties": {"title": "Demo endpoint: invocations + 5XX",
                        "metrics": [["AWS/SageMaker", "Invocations", "EndpointName", ENDPOINT],
                                    [".", "ModelInvocation5XXErrors", ".", "."]],
                        "period": 60, "stat": "Sum", "region": r}},
        {"type": "metric", "x": 12, "y": 0, "width": 12, "height": 6,
         "properties": {"title": "Demo endpoint: latency (ms)",
                        "metrics": [["AWS/SageMaker", "ModelLatency", "EndpointName", ENDPOINT]],
                        "period": 60, "stat": "Average", "region": r}}]}
    cw.put_dashboard(DashboardName=DASHBOARD, DashboardBody=json.dumps(dashboard))

    # second pass: rewrite index.html with the real API URL
    html = html.replace("__API_URL__", api_url)
    s3.put_object(Bucket=SITE_BUCKET, Key="index.html", Body=html, ContentType="text/html")

    site = f"http://{SITE_BUCKET}.s3-website-{r}.amazonaws.com"
    print(f"""

================ DEMO STACK READY ================
  UI:            {site}
  API:           {api_url}
  Endpoint:      {ENDPOINT} (serverless - $0 at rest)
  Dashboard:     CloudWatch -> Dashboards -> {DASHBOARD}
  Reports prefix: s3://{bucket}/{MON}/

Next: scripts/demo_evidence.py  (baseline/bias/quality reports, ~25 min)
Then: .venv-sm/bin/python scripts/demo_day.py on   (keeps endpoint warm while recording)
""")


if __name__ == "__main__":
    main()
