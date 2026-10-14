# Demo & Recording Runbook — AAI-540 Group 1

Everything below assumes `scripts/deploy_demo_stack.py` + `scripts/demo_evidence.py`
have run (one command chain; ~55 min, <$2). The stack then costs **~$0.05–0.10/month**
at rest (S3 website + reports; the serverless endpoint is $0 until invoked).

## Part 1 — Recording-day setup (5 minutes, before you hit record)

```bash
cd /Users/sourangshupal/Downloads/final-project-usd
set -a && source .env && set +a
.venv-sm/bin/python scripts/demo_day.py on      # keeps the endpoint warm; leave running
```

Open these four tabs before recording:
1. **Demo UI**: `http://aai540-g1-demo-<account>.s3-website-us-east-1.amazonaws.com`
   (URL is printed by the deploy script)
2. **CloudWatch dashboard**: console → CloudWatch → Dashboards → `aai540-g1-monitoring`
3. **S3 reports**: bucket `sagemaker-us-east-1-<account>` → `aai-540-g1/monitoring/`
   (`data-baseline/` = statistics/constraints/report, `reports/` = model quality,
   `bias-report/` = Clarify bias)
4. **SageMaker Studio → Pipelines**: `aai-540-g1-demo-pipeline-ok` (green DAG) and
   `aai-540-g1-demo-pipeline-fail` (red DAG)

## Part 2 — The recording script (maps to the module steps)

| # | Show | Say | Covers |
|---|---|---|---|
| 1 | UI tab → pick **person 582710** → "Predict" → **HIGH RISK 91%**, probability bars → scroll to driving features (21 visits, CHF) → "true outcome: High" | "Champion XGBoost, test macro F1 0.774, High-Risk recall 0.847. Served by SageMaker *serverless* inference — scales to zero, so this demo costs pennies." | deployment |
| 2 | Switch to **dashboard tab** → point at the invocation spike from that prediction; latency graph | "Every request is captured — watch the dashboard light up live. Infrastructure monitors: 5XX and latency alarms are armed." | infra monitors + dashboard |
| 3 | **S3 tab** → `data-baseline/` → open `statistics.json` + `constraints.json` + the report | "The data monitor: a SageMaker baseline job profiles the training distribution; scoring-time data is checked against these constraints — that's the data-quality report." | data monitors + reports |
| 4 | **S3 tab** → `reports/model_quality_report.html` | "The model monitor: live endpoint predictions vs held-out ground truth — macro F1 on 400 fresh members. The classic schedule APIs are in AWS maintenance mode, so we run monitoring as explicit evidence jobs — the finding is documented in notebook 06." | model monitors + reports |
| 5 | **S3 tab** → `bias-report/` (Clarify) — or the NOTE if the job was skipped | "Bias monitor: Clarify measures whether the model treats groups differently — here, gender." | model monitors |
| 6 | **Studio tab** → green DAG → then red DAG | "CI/CD: same pipeline, two outcomes. Green: champion config trains and would register. Red: an invalid hyperparameter fails the training step — the DAG shows exactly where, and nothing downstream runs." | CI/CD evidence |
| 7 | (optional) **Console** → SageMaker → Feature Store → `aai-540-g1-member-features` → check a record | "Features live in a Feature Store — the same record the model scores." | feature groups |

## Part 3 — After recording

1. Ctrl+C the `demo_day.py` warmth process (endpoint returns to $0 at rest)
2. Optional full teardown: `.venv/bin/python scripts/teardown_aws.py`
   (deletes endpoint/registry/feature group; re-run the two demo scripts any time to restore)

## Cost summary
- At rest: <$0.10/month (S3 + static hosting + report storage)
- Recording day: < $1 (warmth invokes + any cold starts)
- Never: an always-on instance
