"""Regenerate the demo bucket's data assets (run before recording).

Produces, in the public demo bucket (aai540-g1-demo-<account>):
  members.json       12 demo members + summary_raw (for the vs-class chart)
  class_profile.json z-scored per-class feature means (Low/Medium/High + all)
  monitoring/metrics.json  last 3h endpoint Invocations + latency series (CloudWatch)
  architecture.png   copied from assets/

Run:  set -a && source .env && set +a && .venv/bin/python scripts/refresh_demo_assets.py
"""

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import boto3
import pandas as pd

from aws_common import account_id, check_credentials, region
from model_common import encode_features, load_table

VS_FEATURES = ["total_visits", "inpatient_visits", "drug_eras",
               "observation_months", "age", "distinct_conditions"]
ENDPOINT = "aai540-g1-demo-endpoint"


def main():
    check_credentials()
    r = region()
    s3 = boto3.client("s3", region_name=r)
    cw = boto3.client("cloudwatch", region_name=r)
    bucket = f"aai540-g1-demo-{account_id()}"

    table = load_table(ROOT / "data" / "processed" / "training_table.csv")
    X_all, feature_names, audit = encode_features(table)

    # class z-profiles
    prof = {"features": VS_FEATURES, "all": {}}
    for f in VS_FEATURES:
        mu, sd = table[f].mean(), table[f].std() or 1.0
        prof["all"][f] = {"mean": float(mu), "std": float(sd)}
    for tier in ["Low", "Medium", "High"]:
        prof[tier] = {f: float((table.loc[table["risk_class"] == tier, f].mean() - prof["all"][f]["mean"])
                               / prof["all"][f]["std"]) for f in VS_FEATURES}
    s3.put_object(Bucket=bucket, Key="class_profile.json", Body=json.dumps(prof))
    print("class_profile.json:", {t: [round(prof[t][f], 2) for f in VS_FEATURES] for t in ["Low", "High"]})

    # members with raw feature values
    picks = []
    for cls in ["Low", "Medium", "High"]:
        picks += list(table[table["risk_class"] == cls].index[:4])
    members = []
    for i in picks:
        row = table.loc[i]
        members.append({
            "id": int(row["person_id"]),
            "risk_true": row["risk_class"],
            "summary": {k: int(row[k]) for k in
                        ["age", "total_visits", "drug_eras", "observation_months",
                         "chf", "diabetes", "copd", "stroke", "cancer", "renal_disease"]},
            "summary_raw": {f: float(row[f]) for f in VS_FEATURES},
            "vector": [round(float(v), 6) for v in X_all[i]],
        })
    s3.put_object(Bucket=bucket, Key="members.json", Body=json.dumps(members))
    print(f"members.json: {len(members)} members with raw profiles")

    # CloudWatch metrics series (last 3h, 5-min periods)
    now = datetime.now(timezone.utc)
    series = {}
    for name, stat in [("Invocations", "Sum"), ("ModelLatency", "Average")]:
        data = cw.get_metric_statistics(
            Namespace="AWS/SageMaker", MetricName=name,
            Dimensions=[{"Name": "EndpointName", "Value": ENDPOINT},
                        {"Name": "VariantName", "Value": "demo"}],
            StartTime=now - timedelta(hours=3), EndTime=now,
            Period=300, Statistics=[stat])
        series[name] = sorted(data["Datapoints"], key=lambda p: p["Timestamp"])
    timestamps = [p["Timestamp"].strftime("%H:%M") for p in series["Invocations"]]
    metrics = {
        "timestamps": timestamps,
        "invocations": [p["Sum"] for p in series["Invocations"]],
        "latency_ms": [round(p["Average"]) for p in series["ModelLatency"]],
        "total_invocations": int(sum(p["Sum"] for p in series["Invocations"])),
        "avg_latency_ms": float(sum(p["Average"] for p in series["ModelLatency"])
                                / max(1, len(series["ModelLatency"]))),
    }
    s3.put_object(Bucket=bucket, Key="monitoring/metrics.json", Body=json.dumps(metrics))
    print(f"metrics.json: {metrics['total_invocations']} invocations over {len(timestamps)} periods")

    arch = ROOT / "assets" / "architecture_diagram.png"
    if arch.is_file():
        s3.upload_file(str(arch), bucket, "architecture.png")
        print("architecture.png uploaded")
    print("\ndemo assets refreshed.")


if __name__ == "__main__":
    main()
