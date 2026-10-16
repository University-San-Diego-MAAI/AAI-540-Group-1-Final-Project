"""Generate the v2 design-document diagram suite (7 figures) + mermaid sources.

Renders matplotlib PNGs (200 dpi, white background, repo palette) into
assets/v2/ and writes the equivalent mermaid flowchart source next to each
figure under assets/v2/mermaid/ for editing in mermaid.live.

Run from the repo root:  .venv/bin/python scripts/build_v2_diagrams.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "assets" / "v2"
MERMAID = OUT / "mermaid"
ACCENT = "#1F4E79"
LIGHT = "#dbe7f3"
GREEN = "#2e7d32"
RED = "#c62828"
AMBER = "#f9a825"


def _new(w=13, h=7.4):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def _box(ax, x, y, w, h, title, lines=None, fc="white", ec=ACCENT, title_c=ACCENT, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.6",
                                fc=fc, ec=ec, lw=1.6, linestyle="--" if dashed else "-"))
    ax.text(x + w / 2, y + h - 3.0, title, ha="center", va="top",
            fontsize=10.5, fontweight="bold", color=title_c)
    if lines:
        ax.text(x + w / 2, y + h - 8.0, "\n".join(lines), ha="center", va="top",
                fontsize=8.2, color="#333333", linespacing=1.45)


def _arrow(ax, x1, y1, x2, y2, label="", color=ACCENT):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=16, lw=1.8, color=color))
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.6, label, ha="center",
                fontsize=7.6, color="#555555", style="italic")


def _note(ax, x, y, text, w=46):
    _box(ax, x, y, w, 12, "NOTE", [text], fc="#fff8e1", ec=AMBER, title_c="#8d6e00")


def _save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", f"assets/v2/{name}.png")


def _mmd(name, source):
    MERMAID.mkdir(parents=True, exist_ok=True)
    (MERMAID / f"{name}.mmd").write_text(source)
    print("wrote", f"assets/v2/mermaid/{name}.mmd")


# ---------------------------------------------------------------- fig 1
def fig1():
    fig, ax = _new(14, 7.6)
    _box(ax, 1, 74, 20, 20, "PUBLIC SOURCE",
         ["s3://synpuf-omop", "CMS DE-SynPUF", "OMOP CDM v5.x", "17 tables, 100k members"])
    _box(ax, 27, 74, 22, 20, "S3 DATALAKE", ["raw/ mirrored tables",
         "processed/ training table", "artifacts/ models+reports", "default bucket + aai-540-g1/"])
    _box(ax, 55, 74, 20, 20, "ATHENA", ["workgroup aai540_g1",
         "database + external tables", "SQL validation queries"])
    _box(ax, 79, 74, 20, 20, "FEATURE ENGINEERING", ["39-check validation gate",
         "temporal features (2008)", "labels from 2009-2010"], fc=LIGHT)
    _box(ax, 27, 44, 22, 20, "FEATURE STORE", ["aai-540-g1-member-features",
         "82,624 records", "online + offline (Glue/Athena)"])
    _box(ax, 55, 44, 20, 20, "TRAINING (SageMaker)", ["logreg / RF / XGBoost jobs",
         "champion: xgb-e005-d6-w5", "macro F1 0.774"])
    _box(ax, 79, 44, 20, 20, "MODEL REGISTRY", ["group: member-utilization-risk",
         "package + model card", "PendingManualApproval gate"])
    _box(ax, 27, 12, 32, 20, "DEPLOY A · BATCH (production)", [
         "Batch Transform, ml.m5.xlarge", "33,048 members scored",
         "ranked High-Risk worklist"])
    _box(ax, 67, 12, 32, 20, "DEPLOY B · SERVERLESS (demo)", [
         "S3 site -> API GW -> Lambda", "serverless endpoint, $0 at rest",
         "cold-start trade-off"])
    _box(ax, 1, 44, 20, 20, "MONITORING", ["CloudWatch dashboard", "+ 2 alarms",
         "baseline + quality + bias", "reports in S3"], fc="#e8f5e9", ec=GREEN, title_c=GREEN)
    _arrow(ax, 21, 84, 27, 84); _arrow(ax, 49, 84, 55, 84); _arrow(ax, 75, 84, 79, 84)
    _arrow(ax, 89, 74, 38, 64, "features+labels"); _arrow(ax, 49, 54, 55, 54)
    _arrow(ax, 75, 54, 79, 54); _arrow(ax, 89, 54, 87, 32, "approved model")
    _arrow(ax, 65, 54, 46, 32, "champion artifact")
    _arrow(ax, 21, 54, 27, 24, "")
    ax.text(50, 99, "Figure 1 - System architecture (v2, as implemented)",
            ha="center", fontsize=13, fontweight="bold", color=ACCENT)
    _save(fig, "fig1_system_architecture")
    _mmd("fig1_system_architecture", """flowchart LR
    A[s3://synpuf-omop<br/>17 OMOP tables] --> B[S3 datalake<br/>raw/ processed/ artifacts/]
    B --> C[Athena<br/>workgroup aai540_g1]
    C --> D[Feature engineering<br/>39-check gate, temporal features]
    D --> E[Feature Store<br/>82,624 records]
    E --> F[Training jobs<br/>logreg / RF / XGBoost]
    F --> G[Model Registry<br/>approval gate]
    G --> H[Batch Transform<br/>33,048 scored]
    G --> I[Serverless endpoint<br/>demo UI]
    M[Monitoring<br/>CloudWatch + reports] -.-> H
    M -.-> I
""")


# ---------------------------------------------------------------- fig 2
def fig2():
    fig, ax = _new(13, 7)
    _box(ax, 1, 66, 24, 26, "17 OMOP CDM TABLES", [
        "person 100k rows", "visit_occurrence 4.79M", "condition_occurrence 12.7M",
        "drug_exposure 5.4M", "procedure_occurrence 11.9M", "+ 12 more"])
    _box(ax, 31, 66, 24, 26, "VALIDATION GATE", ["src/load_synpuf.py",
         "39 automated checks", "row counts, PK uniqueness,",
         "referential integrity", "FAILS LOUDLY before modeling"],
         fc=LIGHT)
    _box(ax, 61, 66, 18, 26, "MEMBER TABLE", ["build_features.py",
         "1 row / member", "build_training_data.py", "82,624 labeled rows"])
    _box(ax, 83, 66, 16, 26, "S3", ["processed/",
         "training/", "training_table.csv"])
    _box(ax, 31, 30, 24, 24, "ATHENA SETUP", ["scripts/setup_athena.py",
         "workgroup aai540_g1", "DB aai540_g1",
         "CREATE EXTERNAL TABLE"])
    _box(ax, 61, 30, 38, 24, "VALIDATION QUERIES (screenshot evidence)", [
         "member_count  = 82,624",
         "class balance 33.7 / 33.3 / 33.0",
         "CHF prevalence climbs 11.7% -> 28.2% -> 51.5%",
         "by risk class (Low / Medium / High)"], fc="#e8f5e9", ec=GREEN, title_c=GREEN)
    _arrow(ax, 25, 79, 31, 79); _arrow(ax, 55, 79, 61, 79); _arrow(ax, 79, 79, 83, 79)
    _arrow(ax, 72, 66, 46, 54, "external table over S3")
    _arrow(ax, 55, 42, 61, 42)
    ax.text(50, 99, "Figure 2 - Data engineering + Athena cataloging",
            ha="center", fontsize=13, fontweight="bold", color=ACCENT)
    _save(fig, "fig2_data_lifecycle_athena")
    _mmd("fig2_data_lifecycle_athena", """flowchart LR
    A[17 OMOP tables] --> B[validation gate<br/>39 checks]
    B --> C[member table<br/>82,624 rows]
    C --> D[S3 processed/training/]
    D --> E[Athena<br/>workgroup + external table]
    E --> F[validation queries<br/>counts, balance, prevalence]
""")


# ---------------------------------------------------------------- fig 3
def fig3():
    fig, ax = _new(14, 7.4)
    ax.add_patch(plt.Rectangle((2, 62), 46, 28, fc="#dbe7f3", ec=ACCENT, lw=1.4))
    ax.text(25, 87, "HISTORY WINDOW = 2008", ha="center", fontsize=10.5,
            fontweight="bold", color=ACCENT)
    ax.add_patch(plt.Rectangle((52, 62), 46, 28, fc="#fde9d9", ec="#c55a11", lw=1.4))
    ax.text(75, 87, "OUTCOME WINDOW = 2009-2010", ha="center", fontsize=10.5,
            fontweight="bold", color="#c55a11")
    for i, f in enumerate(["age / band", "visit counts", "chronic flags (ICD-9)",
                           "pharmacy intensity", "breadth (conditions,",
                           "procedures)", "observation months"]):
        ax.text(25, 81 - i * 3.1, "- " + f, ha="center", fontsize=8.2, color="#333")
    ax.text(75, 76, "outcome_score =", ha="center", fontsize=9, fontweight="bold")
    ax.text(75, 71, "total_visits + 3 x inpatient_visits + drug_eras",
            ha="center", fontsize=8.6, color="#333")
    ax.text(75, 65, "rank terciles -> Low / Medium / High (tie-safe)",
            ha="center", fontsize=8.6, color="#333")
    _box(ax, 2, 40, 96, 14, "ELIGIBILITY + CENSORING", [
        "rule: observation period overlaps BOTH windows (start <= 2008-12-31, end >= 2009-01-01)",
        "-> 85,323 eligible (85.3%) -> 2,699 outcome-window deaths censored -> 82,624 labeled members"],
        fc=LIGHT)
    splits = [("train", 40, ACCENT), ("validation", 10, "#4a80b4"),
              ("test", 10, "#7aa6c9"), ("production", 40, AMBER)]
    x = 2
    ax.text(50, 34, "COURSE-MANDATED SPLIT (Team Update 3.1) - stratified per class, seed 42",
            ha="center", fontsize=9.5, fontweight="bold")
    for name, share, color in splits:
        w = share * 0.96
        ax.add_patch(plt.Rectangle((x, 20), w, 10, fc=color, ec="white", lw=2))
        ax.text(x + w / 2, 25, f"{name}\n{share}%", ha="center", va="center",
                fontsize=9, color="white", fontweight="bold")
        x += w
    ax.text(50, 14, "production 40% = reserved 'production data' for Module 5 monitoring (ground truth known, scored blind)",
            ha="center", fontsize=8.2, style="italic", color="#555")
    ax.text(50, 99, "Figure 3 - Temporal windows, rule-based labels, and the 40/10/10/40 split",
            ha="center", fontsize=13, fontweight="bold", color=ACCENT)
    _save(fig, "fig3_temporal_split_labels")
    _mmd("fig3_temporal_split_labels", """flowchart TB
    subgraph H["HISTORY WINDOW = 2008"]
        H1[demographics + utilization counts + chronic flags + pharmacy + breadth]
    end
    subgraph O["OUTCOME WINDOW = 2009-2010"]
        O1["score = visits + 3*inpatient + drug_eras"]
        O1 --> O2[rank terciles -> Low / Medium / High]
    end
    H --> S[split 40/10/10/40<br/>train / val / test / production]
    O2 --> S
    S --> P[production 40% reserved<br/>for monitoring ground truth]
""")


# ---------------------------------------------------------------- fig 4
def fig4():
    fig, ax = _new(14, 7.2)
    _box(ax, 1, 62, 20, 26, "BENCHMARKS", ["B1 majority class",
         "macro F1 0.168", "B2 visits-tercile rule",
         "macro F1 0.530"], fc="#eeeeee", ec="#666666", title_c="#333333")
    _box(ax, 27, 62, 26, 26, "5 SAGEMAKER TRAINING JOBS", [
         "logreg-c1 (saga, L2)", "rf-d16-l5 (balanced)", "xgb-e01-d6-w5",
         "xgb-e005-d6-w5", "xgb-e01-d8-w1", "seed 42, identical preprocessing"],
         fc=LIGHT)
    _box(ax, 59, 62, 38, 26, "CHAMPION SELECTION RULE", [
         "1. validation macro F1, ranked",
         "2. High-Risk recall >= 0.70 floor",
         "3. must beat benchmark B2 on test",
         "champion: xgb-e005-d6-w5"], fc="#e8f5e9", ec=GREEN, title_c=GREEN)
    _box(ax, 1, 22, 30, 26, "HELD-OUT TEST (10%)", [
         "macro F1  0.774", "High-Risk recall  0.847",
         "ROC-AUC (OvR)  0.916", "confusion matrix +",
         "fairness slices (gender / race / age band)"], fc=LIGHT)
    _box(ax, 37, 22, 28, 26, "INTERPRETABILITY", [
         "booster feature importance:",
         "total_visits, chronic flags,",
         "pharmacy intensity dominate;",
         "no label-leakage columns"], fc=LIGHT)
    _box(ax, 71, 22, 26, 26, "REGISTERED", [
         "Model Registry group", "package v + model card",
         "PendingManualApproval", "-> Figure 7 (CI/CD)"])
    _arrow(ax, 21, 75, 27, 75); _arrow(ax, 53, 75, 59, 75)
    _arrow(ax, 78, 62, 20, 48, "test once"); _arrow(ax, 31, 35, 37, 35)
    _arrow(ax, 65, 35, 71, 35)
    ax.text(50, 99, "Figure 4 - Training, evaluation, and champion selection (measured)",
            ha="center", fontsize=13, fontweight="bold", color=ACCENT)
    _save(fig, "fig4_training_evaluation")
    _mmd("fig4_training_evaluation", """flowchart LR
    B[benchmarks<br/>0.168 / 0.530 macro F1] --> T[5 training jobs<br/>logreg / RF / 3x XGB]
    T --> C["champion rule:<br/>val F1 + recall floor >= 0.70 + beats benchmark"]
    C --> X["test: macro F1 0.774<br/>High recall 0.847, AUC 0.916"]
    X --> I[feature importance<br/>+ fairness slices]
    I --> R[Model Registry<br/>approval gate]
""")


# ---------------------------------------------------------------- fig 5
def fig5():
    fig, ax = _new(14, 7.2)
    _box(ax, 27, 78, 44, 14, "CHAMPION ARTIFACT (model-clean.tar.gz, XGBOOST 1.7-1)",
         fc=LIGHT)
    ax.text(2, 88, "PATH A - PRODUCTION (batch)", fontsize=11.5, fontweight="bold", color=ACCENT)
    _box(ax, 2, 46, 30, 24, "BATCH TRANSFORM", ["ml.m5.xlarge, input_filter",
         "drops person_id; predictions", "keep input row order",
         "33,048 production members"])
    _box(ax, 2, 12, 30, 24, "OUTPUT", ["production_scores.csv:",
         "person_id + 3 probabilities", "+ predicted tier",
         "-> ranked High-Risk worklist", "(care-management cohort)"],
         fc="#e8f5e9", ec=GREEN, title_c=GREEN)
    ax.text(52, 88, "PATH B - DEMO / WHAT-IF (real-time, serverless)", fontsize=11.5,
            fontweight="bold", color="#c55a11")
    _box(ax, 52, 46, 22, 24, "STATIC UI", ["S3 website bucket",
         "operations console", "Chart.js dashboards"])
    _box(ax, 78, 46, 20, 24, "API GW + LAMBDA", ["HTTP API (CORS)",
         "lambda proxy ->", "invoke_endpoint", "retries on cold start"])
    _box(ax, 64, 12, 34, 24, "SAGEMAKER SERVERLESS ENDPOINT", [
         "scales to ZERO -> $0 at rest", "~$0.000035 / invocation",
         "cold start 5-8 s (retry absorbs)",
         "NO data capture (limitation)"], fc="#fde9d9", ec="#c55a11", title_c="#c55a11")
    _arrow(ax, 49, 80, 20, 70); _arrow(ax, 20, 46, 20, 36)
    _arrow(ax, 49, 80, 82, 70); _arrow(ax, 74, 58, 78, 58)
    _arrow(ax, 88, 46, 84, 36)
    ax.text(50, 99, "Figure 5 - Deployment: batch production scorer + serverless demo endpoint",
            ha="center", fontsize=13, fontweight="bold", color=ACCENT)
    _save(fig, "fig5_deployment_paths")
    _mmd("fig5_deployment_paths", """flowchart TB
    M[champion artifact] --> A[Batch Transform<br/>33,048 scored -> worklist]
    M --> S[S3 static UI] --> G[API GW + Lambda] --> E[serverless endpoint<br/>$0 at rest, cold-start trade-off]
""")


# ---------------------------------------------------------------- fig 6
def fig6():
    fig, ax = _new(14, 7.4)
    _box(ax, 1, 74, 24, 20, "ENDPOINT TRAFFIC", ["UI clicks + replays",
         "100% of requests", "(serverless: no data", "capture available)"])
    _box(ax, 31, 74, 30, 20, "CLOUDWATCH - INFRA MONITORS", [
         "dashboard aai540-g1-monitoring", "alarms: 5XX errors, latency > 30s",
         "Invocations / ModelLatency /", "OverheadLatency metrics"], fc=LIGHT)
    _box(ax, 67, 74, 32, 20, "DATA MONITOR (evidence job)", [
         "DefaultModelMonitor baseline job", "(processing, ml.m5.xlarge)",
         "-> statistics.json + constraints.json", "-> S3 aai-540-g1/monitoring/"])
    _box(ax, 1, 38, 30, 24, "MODEL MONITOR (evidence job)", [
         "400 live-scored held-out members", "vs ground truth ->",
         "model_quality_report.html/json", "Clarify bias: NOTE (job could",
         "not run - documented)"])
    _box(ax, 37, 38, 30, 24, "FINDING: API RETIREMENT", [
         "CreateDataQualityJobDefinition +",
         "CreateModelQualityJobDefinition in",
         "AWS MAINTENANCE MODE (new customers",
         "blocked) -> schedules impossible;",
         "evidence-job design adopted"], fc="#fff8e1", ec=AMBER, title_c="#8d6e00")
    _box(ax, 73, 38, 26, 24, "REPORTING SURFACE", [
         "reports.html (public page)",
         "renders report JSONs as charts",
         "+ CloudWatch console dashboard",
         "for the video demo"], fc="#e8f5e9", ec=GREEN, title_c=GREEN)
    _arrow(ax, 25, 84, 31, 84); _arrow(ax, 61, 84, 67, 84)
    _arrow(ax, 79, 74, 18, 62, "baseline reference")
    _arrow(ax, 31, 50, 37, 50); _arrow(ax, 67, 50, 73, 50)
    _note(ax, 8, 12, "designed-but-future: PSI drift schedules, capture-based monitors (require provisioned endpoint)")
    ax.text(50, 99, "Figure 6 - Monitoring architecture (as implemented, with maintenance-mode finding)",
            ha="center", fontsize=13, fontweight="bold", color=ACCENT)
    _save(fig, "fig6_monitoring_architecture")
    _mmd("fig6_monitoring_architecture", """flowchart LR
    T[endpoint traffic] --> C[CloudWatch<br/>dashboard + alarms]
    T --> Q[model quality evidence job<br/>400 members vs ground truth]
    B[baseline job] --> R[statistics + constraints]
    Q --> P[reports web page]
    R --> P
    N[NOTE: schedule APIs in<br/>AWS maintenance mode] -.-> Q
""")


# ---------------------------------------------------------------- fig 7
def fig7():
    fig, ax = _new(14, 7.2)
    _box(ax, 4, 66, 26, 22, "STEP 1 · TRAIN", ["TrainingStep, XGBoost 1.7-1",
         "ml.m5.2xlarge, src/train.py", "champion hyperparameters", "(eta 0.05 / depth 6 / mcw 5)"])
    _box(ax, 40, 66, 26, 22, "STEP 2 · REGISTER", ["RegisterModel ->",
         "group member-utilization-risk", "ModelApprovalStatus =",
         "PendingManualApproval (HUMAN GATE)"])
    _box(ax, 76, 66, 22, 22, "PRODUCTION", ["human approves package,",
         "then: batch transform or", "endpoint update;",
         "rollback = repoint to prior", "package version"])
    _box(ax, 40, 34, 26, 18, "DESIGNED · EVALUATION GATE", [
         "auto-check: test F1 >= champion",
         "AND High-recall >= floor",
         "(ConditionStep - roadmap,",
         "evaluation currently in nb 04)"], dashed=True)
    _box(ax, 4, 8, 42, 18, "EXECUTION · OK (green DAG)", [
         "aai-540-g1-demo-pipeline-ok",
         "status: Succeeded - train -> register"],
         fc="#e8f5e9", ec=GREEN, title_c=GREEN)
    _box(ax, 54, 8, 42, 18, "EXECUTION · FAIL (red DAG)", [
         "aai-540-g1-demo-pipeline-fail",
         "eta='abc' -> argparse exit 2",
         "status: Failed - train step red,"],
         fc="#ffebee", ec=RED, title_c=RED)
    _arrow(ax, 30, 77, 40, 77); _arrow(ax, 66, 77, 76, 77)
    _arrow(ax, 53, 66, 53, 52)
    ax.text(25, 3, "nothing downstream runs", fontsize=8, color=RED, ha="center")
    ax.text(50, 99, "Figure 7 - CI/CD pipeline: DAG, human gate, and both execution states",
            ha="center", fontsize=13, fontweight="bold", color=ACCENT)
    _save(fig, "fig7_cicd_pipeline")
    _mmd("fig7_cicd_pipeline", """flowchart LR
    T[train step] --> R[register step<br/>PendingManualApproval]
    R --> P[production<br/>after human approval]
    T -.->|"evaluation gate (roadmap)"| G[dashed]
    subgraph executions
      OK[ok run: Succeeded] 
      FAIL[fail run: train step red]
    end
""")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7):
        fn()
    print("\nAll 7 figures + mermaid sources written to assets/v2/")


if __name__ == "__main__":
    main()
