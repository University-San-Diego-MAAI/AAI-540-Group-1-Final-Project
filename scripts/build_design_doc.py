"""Build the AAI-540 ML Design Document (Team1_ML_Design_Document.docx).

Generates the architecture diagram (assets/architecture_diagram.png) and the
filled-in design document from the template structure. Content is grounded in
the actual project: CMS DE-SynPUF (OMOP CDM) from the AWS Open Data bucket
s3://synpuf-omop, the EDA/feature pipeline in src/, and the RFC objective of
multiclass member utilization-risk classification.

Run from the repo root:  .venv/bin/python scripts/build_design_doc.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
DIAGRAM = ASSETS / "architecture_diagram.png"
OUTPUT = ROOT / "Team1_ML_Design_Document.docx"

ACCENT = "#1F4E79"
LIGHT = "#D9E2F3"


# --------------------------------------------------------------------------- #
# Architecture diagram
# --------------------------------------------------------------------------- #
def build_diagram() -> None:
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    def box(x, y, w, h, title, lines, fc="white", title_color=ACCENT):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.08,rounding_size=0.12",
                linewidth=1.4, edgecolor=ACCENT, facecolor=fc,
            )
        )
        ax.text(x + w / 2, y + h - 0.32, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color=title_color)
        body = "\n".join(lines)
        ax.text(x + w / 2, y + (h - 0.55) / 2, body, ha="center", va="center",
                fontsize=7.6, color="#222222")

    def arrow(x1, y1, x2, y2, label=""):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
            linewidth=1.3, color=ACCENT,
        ))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.14, label,
                    ha="center", fontsize=7, color=ACCENT, style="italic")

    # Row 1: pipeline stages (left -> right)
    box(0.2, 4.6, 2.5, 1.9, "DATA SOURCE", [
        "Public S3 bucket",
        "s3://synpuf-omop",
        "(AWS Open Data, no",
        "credentials needed)",
        "17 OMOP CDM v5.x",
        "tables, 100k persons",
    ], fc="#EAF1FB")
    box(3.1, 4.6, 2.5, 1.9, "PROJECT STORAGE", [
        "Project S3 bucket",
        "aai-540-g1-*/",
        "raw/  processed/",
        "artifacts/",
        "SSE-S3 encryption,",
        "no public access",
    ], fc="#EAF1FB")
    box(6.0, 4.6, 2.7, 1.9, "SAGEMAKER\nPROCESSING", [
        "validate (39-check",
        "gate), clean, parse",
        "dates & ICD-9,",
        "memory-safe reads",
        "-> member_features",
        "(24 cols, per person)",
    ])
    box(9.1, 4.6, 2.4, 1.9, "SAGEMAKER\nTRAINING", [
        "LogReg / RF /",
        "XGBoost (3-class)",
        "hyperparameter",
        "tuning, stratified",
        "temporal split",
    ])
    box(11.9, 4.6, 1.4, 1.9, "MODEL\nREGISTRY", [
        "versioned",
        "approved",
        "models",
    ])

    # Row 2: consumption + monitoring
    box(3.1, 2.3, 3.2, 1.5, "BATCH SCORING", [
        "SageMaker Batch Transform",
        "monthly cohort refresh",
        "-> Low / Medium / High",
        "risk assignments (CSV)",
    ], fc="#EAF1FB")
    box(6.9, 2.3, 2.6, 1.5, "CARE-MGMT\nCONSUMPTION", [
        "ranked member list",
        "feeds outreach",
        "worklists (batch,",
        "not real-time)",
    ])
    box(9.9, 2.3, 3.4, 1.5, "MONITORING & CI/CD", [
        "SageMaker Pipelines: validate",
        "-> train -> eval gate -> approve",
        "CloudWatch: job health, data &",
        "prediction drift, macro-F1 decay",
    ])

    arrow(2.7, 5.55, 3.1, 5.55, "s3 sync")
    arrow(5.6, 5.55, 6.0, 5.55)
    arrow(8.7, 5.55, 9.1, 5.55, "features")
    arrow(11.5, 5.55, 11.9, 5.55)
    arrow(10.3, 4.6, 6.5, 3.8, "approved model")
    arrow(6.3, 3.05, 6.9, 3.05, "scores")
    arrow(11.0, 3.8, 11.6, 4.6)

    ax.text(6.75, 6.95, "Member Utilization-Risk Classification - System Architecture",
            ha="center", fontsize=13, fontweight="bold", color=ACCENT)
    ax.text(6.75, 0.55,
            "History window (2008) features  ->  outcome window (2009-2010) rule-based risk terciles  |  "
            "batch design: care-management cohort selection has no latency requirement",
            ha="center", fontsize=8.2, color="#444444", style="italic")

    fig.savefig(DIAGRAM, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Document helpers
# --------------------------------------------------------------------------- #
def set_base_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    for style_name, size in [("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11.5)]:
        style = doc.styles[style_name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        style.font.bold = True


def para(doc, text, bold=False, italic=False, size=None, align=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    if align:
        p.alignment = align
    return p


def lead(doc, question, answer):
    """Render a template guiding question in bold followed by the answer."""
    p = doc.add_paragraph()
    q = p.add_run(question + "  ")
    q.bold = True
    p.add_run(answer)
    return p


def bullets(doc, items):
    for item in items:
        if isinstance(item, tuple):
            head, rest = item
            p = doc.add_paragraph(style="List Bullet")
            r = p.add_run(head)
            r.bold = True
            p.add_run(rest)
        else:
            doc.add_paragraph(item, style="List Bullet")


def numbered(doc, items):
    for item in items:
        doc.add_paragraph(item, style="List Number")


def table(doc, headers, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(h)
        run.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
    return t


# --------------------------------------------------------------------------- #
# Content
# --------------------------------------------------------------------------- #
def build_document() -> None:
    build_diagram()
    doc = Document()
    set_base_styles(doc)

    # ---- Title & team info (preserved from template) ---------------------- #
    para(doc, "AAI-540 ML Design Document", bold=True, size=22,
         align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()

    doc.add_heading("Team Info", level=1)
    bullets(doc, [
        ("Project Team Group #: ", "AAI 540 Group 1"),
        ("Authors: ", "Sourangshu Pal, Tanvi Singh, Riyaz Khorasi"),
        ("Business Name: ",
         "UnitedHealthcare (hypothetical care-management scenario)"),
        ("Publication Date: ", "14 September 2026"),
    ])
    doc.add_heading("Team Workflows", level=2)
    bullets(doc, [
        ("GitHub Project Link: ",
         "https://github.com/University-San-Diego-MAAI/AAI-540-Group-1-Final-Project"),
        ("Asana Board Link: ",
         "https://app.asana.com/1/952672460738672/project/1218443371995338/board/1218443464762200"),
        ("Team Tracker Link: ",
         "https://docs.google.com/document/d/1LCGo8x_G62LcFTUVHO3pQvP-OAxle8Y0W-f_ht_J0XI/edit?usp=sharing"),
    ])

    # ---- 1. Project Scope -------------------------------------------------- #
    doc.add_heading("1. Project Scope", level=1)

    doc.add_heading("Project Background", level=2)
    para(doc,
         "Healthcare insurers need to understand which members are likely to require "
         "different levels of healthcare utilization and care-management attention in an "
         "upcoming period. For this project we build an ML system for a hypothetical "
         "UnitedHealthcare scenario that uses historical member, claims, prescription, and "
         "utilization data to predict each member's utilization-risk category - Low Risk, "
         "Medium Risk, or High Risk - so the organization can proactively direct "
         "care-management resources (nurse outreach, care coordinators, disease-management "
         "programs) toward the members who need them most. Catching a High-Risk member early "
         "is far cheaper than reacting after avoidable admissions accumulate.")
    para(doc,
         "The model's objective is a supervised multiclass classification: given a member's "
         "history-window profile (demographics, chronic conditions, inpatient/outpatient "
         "utilization, physician/carrier services, prescription utilization, and coverage), "
         "predict Member_Utilization_Risk with three mutually exclusive classes (Low / "
         "Medium / High). Because no labeled risk scores exist in claims data, we construct "
         "the target with a transparent, rule-based labeling strategy over a future outcome "
         "window (detailed in the Training Data section).")

    doc.add_heading("Technical Background", level=2)
    lead(doc, "How will you evaluate your model?",
         "With complementary multiclass metrics rather than accuracy alone. Primary: macro "
         "F1-score (equal weight to Low/Medium/High). Supporting: per-class precision and "
         "recall - with recall on the High-Risk class treated as the most consequential "
         "number (missing a High-Risk member is worse than over-flagging), multiclass "
         "ROC-AUC (one-vs-rest), and the confusion matrix to see which risk tiers are "
         "confused. We use stratified train/validation/test splits and k-fold "
         "cross-validation during tuning so all three classes are represented in every fold.")
    lead(doc, "What is your data source?",
         "The CMS Data Entrepreneurs' Synthetic Public Use File (DE-SynPUF), 2008-2010, "
         "converted to the OHDSI OMOP Common Data Model v5.x and hosted as a public dataset "
         "on AWS Open Data (S3 bucket arn:aws:s3:::synpuf-omop, region us-east-1, no "
         "credentials or subscription required). We use the 100,000-person sample "
         "(~915 MB compressed, 17 tables), with a 1,000-person sample for fast development. "
         "It contains beneficiary demographics, inpatient/outpatient visits, physician/"
         "carrier-style procedures, prescription drug exposures/eras, observation periods, "
         "payer plan periods, and deaths - all linkable by person_id.")
    lead(doc, "How will you need to prepare your data?",
         "The raw OMOP tables need: (1) schema normalization (the 1k sample ships uppercase "
         "headers and duplicated versioned files, which we skip); (2) a validation gate - 39 "
         "automated checks covering row counts, primary-key uniqueness, and person_id "
         "referential integrity - that fails loudly before any modeling; (3) date parsing "
         "(ISO strings), dot-less ICD-9 handling for condition source codes; and (4) "
         "memory-safe, column-subset reads of the 100k gzips (full load would need 4-6 GB "
         "RAM). We then aggregate event-level records into one member-level feature table.")
    lead(doc, "How will you explore your data?",
         "Per-table EDA on the 1k sample (demographics, observation-period coverage, visit "
         "mix and lengths of stay, ICD-9 condition profiles and chronic-condition "
         "prevalence, drug exposure/era volumes, mortality), with a memory-safe streaming "
         "validation section against the 100k sample to confirm the 1k sample is "
         "representative (visit density 47.5 vs 47.9 rows/member). Findings, distributions, "
         "and caveats are captured in an executed notebook and a written analysis report.")
    lead(doc, "What do you hypothesize your main features will be?",
         "Age band, gender, race, state; utilization counts (inpatient admissions, "
         "outpatient visits, total visits, distinct care sites, observation months); "
         "chronic-condition flags derived from ICD-9 (diabetes, CHF, COPD, stroke, cancer, "
         "renal disease); pharmacy intensity (drug exposures, distinct drugs, drug eras); "
         "and utilization breadth (distinct conditions, distinct procedures). Because the "
         "OMOP conversion carries no payment columns, 'previous spending' is proxied by "
         "utilization intensity. Early evidence supports these hypotheses: chronic-flag "
         "prevalence climbs steeply with utilization (CHF 7% to 53% to 86% across visit "
         "terciles) and a transparent utilization-score proxy separates monotonically on "
         "breadth features.")
    lead(doc, "What type of model do you want to use?",
         "A regularized multinomial logistic regression as the interpretable baseline, "
         "Random Forest for comparison, and gradient-boosted trees (XGBoost) as the primary "
         "model - all native multiclass classifiers, selected on validation macro F1 with "
         "interpretability and the High-Risk-recall business objective as tie-breakers.")

    doc.add_heading("Goals vs Non-Goals", level=2)
    para(doc, "Goals:", bold=True)
    bullets(doc, [
        "Develop a multiclass model that accurately classifies members into Low, Medium, and "
        "High utilization-risk categories from history-window features.",
        "Integrate and engineer features from all five source domains (beneficiary, "
        "inpatient/outpatient, carrier/procedures, prescriptions, coverage) into one "
        "member-level dataset keyed by person_id.",
        "Identify the factors most associated with utilization risk (chronic conditions, "
        "utilization intensity, pharmacy volume) with per-class interpretability.",
        "Compare Logistic Regression, Random Forest, and XGBoost using macro F1, precision, "
        "recall, and confusion matrices, with particular attention to High-Risk recall.",
        "Establish an end-to-end ML-ops approach: automated validation gates, model "
        "registry, drift monitoring, and a CI/CD pipeline with evaluation checkpoints.",
    ])
    para(doc, "Non-Goals:", bold=True)
    bullets(doc, [
        "Build a medical diagnosis or disease-prediction system, or provide individualized "
        "treatment/clinical recommendations.",
        "Predict exact individual healthcare costs or medical expenses (no payment data "
        "exists in this conversion; risk tiers, not dollars).",
        "Develop a real-time claims-streaming system (care-management cohort selection is a "
        "batch workload).",
        "Create a consumer-facing insurance or healthcare application.",
        "Draw real-world conclusions about actual Medicare beneficiaries - DE-SynPUF is "
        "synthetic and CMS states it has limited inferential research value.",
    ])

    # ---- 2. Solution Overview ---------------------------------------------- #
    doc.add_heading("2. Solution Overview", level=1)
    para(doc,
         "The system is an AWS-native batch ML pipeline. Raw OMOP CDM tables are mirrored "
         "from the public synpuf-omop bucket into a private project S3 bucket. A SageMaker "
         "Processing job runs the validation gate (39 checks) and feature engineering to "
         "produce a member-level table (one row per person, 24 columns). A temporal split "
         "defines training data: features from the 2008 history window, rule-based risk "
         "labels from the 2009-2010 outcome window. SageMaker Training jobs train and tune "
         "Logistic Regression, Random Forest, and XGBoost; the best model by validation "
         "macro F1 (subject to a High-Risk recall floor) is registered in the SageMaker "
         "Model Registry and promoted through pipeline stages. Scoring runs as SageMaker "
         "Batch Transform on a monthly cadence, writing Low/Medium/High risk assignments "
         "that care-management teams consume as ranked worklists. Prior to release we test "
         "data validation gates, training reproducibility, evaluation regression versus the "
         "baseline, and bias metrics; in production we monitor data drift, prediction "
         "distribution drift, pipeline health, and macro-F1 decay as outcome windows mature.")
    doc.add_picture(str(DIAGRAM), width=Inches(6.9))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    para(doc, "Figure 1. System architecture: storage, preprocessing, feature engineering, "
              "training, deployment, and monitoring components.",
         italic=True, size=9, align=WD_ALIGN_PARAGRAPH.CENTER)

    doc.add_heading("Data Sources", level=2)
    lead(doc, "What is your data source?",
         "CMS DE-SynPUF 2008-2010 in OMOP CDM v5.x, published by AWS as an Open Data "
         "dataset (public S3 bucket s3://synpuf-omop, us-east-1; originally distributed by "
         "CMS; OMOP conversion by OHDSI). Access requires no AWS account - reads use "
         "--no-sign-request.")
    lead(doc, "What is your data volume?",
         "100,000 synthetic Medicare beneficiaries across 17 OMOP tables (~915 MB "
         "compressed): person 100k rows; visit_occurrence 4.79M; condition_occurrence "
         "12.70M; drug_exposure 5.42M; drug_era 5.37M; procedure_occurrence 11.90M; "
         "measurement 2.91M; observation 1.66M; observation_period 90k; payer_plan_period "
         "335k; death 4,635; plus care_site, location, provider, drug_strength. A 1k-person "
         "sample supports fast development.")
    lead(doc, "Why did you select this data set?",
         "It is the only public dataset with the true structure of CMS Medicare claims "
         "(beneficiary summary, inpatient/outpatient, carrier, and prescription domains "
         "linkable by a single member key) without the cost, privacy, and Data Use "
         "Agreement barriers of real CMS data. It is free, cloud-optimized, standardizes on "
         "the OHDSI OMOP model used across industry research, and is explicitly designed by "
         "CMS for building and benchmarking analytics software before applying to real "
         "Limited Data Sets.")
    lead(doc, "Any risks (bias, sensitive features, etc)?",
         "Yes: (1) synthetic generation - CMS states the file has limited inferential "
         "research value, so patterns may not match real populations; (2) demographic "
         "skew - 82.8% White, 55.6% female, mean age 71.7, and disabled-eligible under-65 "
         "members show distinct utilization; (3) sensitive attributes (race, gender, age, "
         "state) are present and could leak into predictions; (4) 85% of visit rows have "
         "unmapped visit-type concepts, so utilization is measured imperfectly; (5) "
         "claim-line duplication inflates condition counts; (6) no cost/payment fields. "
         "All are documented and mitigated in the design (see Security & Risks).")

    doc.add_heading("Data Engineering", level=2)
    lead(doc, "How will you store this data?",
         "Three S3 prefixes in a private project bucket (e.g., aai-540-g1-final-project): "
         "raw/ (mirrored public data, immutable), processed/ (member-level feature CSV + "
         "intermediate aggregates), and artifacts/ (trained models, evaluation reports, "
         "scoring outputs). SSE-S3 encryption, blocked public access, and least-privilege "
         "IAM roles; the pipeline only reads from the public source bucket. Development "
         "uses a git-tracked local mirror under data/ with data/ gitignored except the "
         "committed feature-table artifact.")
    lead(doc, "What data pre-processing do you need to do before you feed it into your ML "
              "system?",
         "Schema normalization (lowercase headers; skip duplicated versioned files like "
         "*5.2.2/*5.3.1); the 39-check validation gate (row-count ranges, primary-key "
         "uniqueness, person_id referential integrity across visit/condition/drug/"
         "observation tables) that raises before modeling on any failure; parsing of ISO "
         "dates and dot-less ICD-9 source codes (V/E codes safely excluded from numeric "
         "chapter matching); aggregation of event-level rows to member level with explicit "
         "zero-fill semantics for members with no records; and temporal windowing into "
         "history (2008) and outcome (2009-2010) periods keyed on observation-period dates.")

    doc.add_heading("Training Data", level=2)
    lead(doc, "How will you split your data into training, test and validation?",
         "Two layers. First, a temporal eligibility filter: only members observable in both "
         "the 2008 history window and the 2009-2010 outcome window (83.3% of the "
         "population; median coverage ~975 days) enter the labeled set. Second, a "
         "stratified 70/15/15 train/validation/test split on the history-window features, "
         "stratified by the constructed risk class so all three tiers are represented; "
         "k-fold cross-validation on the training fold during hyperparameter tuning. "
         "Members who die during the outcome window are censored and excluded from "
         "evaluation, since their truncated utilization is not a fair 'Low' signal.")
    lead(doc, "Will you use any data labeling techniques?",
         "Yes - rule-based (heuristic) weak supervision. Claims data arrives with no risk "
         "labels, so we construct the Member_Utilization_Risk target from observed future "
         "utilization: an outcome score per member over 2009-2010 (weighted mix of total "
         "visits + 3x inpatient admissions + drug eras, chosen so inpatient-heavy members - "
         "the care-management priority - score high), then terciles mapped to Low / Medium / "
         "High. The rule is versioned and documented; a sensitivity check with alternative "
         "weightings (e.g., adding distinct-condition breadth) is run to confirm label "
         "robustness before model comparison.")

    doc.add_heading("Feature Engineering", level=2)
    lead(doc, "Which fields from your data will you use or exclude?",
         "Use: person demographics (year_of_birth, gender, race, location->state); "
         "visit_occurrence counts by type and care_site; condition_occurrence ICD-9 source "
         "codes (chronic flags + distinct count); drug_exposure/drug_era volumes; "
         "procedure_occurrence distinct codes; observation_period months; "
         "payer_plan_period plan types (Part A/B/D, HMO) as coverage features. Exclude: "
         "person_id (row key, never a feature - prevents identifier memorization); "
         "er_visits (zero-variance in this export - no ER visit rows exist); ethnicity "
         "(byte-for-byte duplicate of race in this conversion); all concept_id vocabulary "
         "columns where 85% are unmapped (use source values instead); and free-text/empty "
         "operational columns surfaced by the missingness profile.")
    lead(doc, "Which fields will be combined or bucketed?",
         "year_of_birth -> age at history window, bucketed to Medicare-style bands (<65, "
         "65-74, 75-84, 85+); ICD-9 source codes -> six binary chronic-condition flags via "
         "prefix chapters (diabetes 250, CHF 428, COPD 491-496, stroke 430-438, cancer "
         "140-209, renal 580-589); location_id -> state; visit concepts -> inpatient/"
         "outpatient/ER counts (with total visits as the robust signal given 85% unmapped "
         "types); nominal columns (age_band, gender, race, state, plan type) one-hot "
         "encoded - never treated as ordinal; utilization rates per observation-month "
         "computed alongside raw counts to control for exposure-time confounding (10.7% of "
         "members have no observation period and 99.6% of those have zero visits, so raw "
         "zero tails partly mean 'not observed').")
    lead(doc, "What other data transformations will you apply to your data?",
         "Zero-fill with explicit semantics (no records = 0, distinct from unknown); "
         "standardization of count features for the logistic-regression baseline (tree "
         "models are scale-invariant); optional log1p transform of heavy-tailed counts "
         "(right-skewed, max ~300 visits); deterministic column ordering and dtypes "
         "(int64 counts/flags) so the training schema is enforceable at scoring time; and a "
         "modeling contract documented in the feature-builder module so training and "
         "scoring apply identical transformations.")

    doc.add_heading("Model Training & Evaluation", level=2)
    lead(doc, "How will you train your model?",
         "Via SageMaker Training jobs fed by the processed feature CSV in S3. A "
         "hyperparameter-tuning job (grid/random search, 20-30 candidates, 5-fold CV on "
         "the training fold) optimizes macro F1 for each algorithm; the champion is chosen "
         "on the validation fold subject to a High-Risk recall floor, then confirmed once "
         "on the untouched test set. Fixed random seeds and versioned data snapshots make "
         "runs reproducible.")
    lead(doc, "What algorithm will you use?",
         "Three native multiclass candidates: (1) multinomial Logistic Regression "
         "(interpretable baseline); (2) Random Forest (nonlinear interactions, robust to "
         "skewed features); (3) XGBoost (primary - gradient boosting typically wins on "
         "tabular claims-style data and handles zero-inflation well).")
    lead(doc, "What parameters will you use?",
         "Logistic Regression: C in {0.01, 0.1, 1, 10} (L2, saga/multinomial). Random "
         "Forest: n_estimators 300-500, max_depth {8, 16, 32, none}, min_samples_leaf "
         "{1, 5, 20}, class_weight='balanced' so High-Risk recall is not drowned by class "
         "frequency. XGBoost: eta {0.03, 0.05, 0.1}, max_depth {4, 6, 8}, subsample 0.8, "
         "colsample_bytree 0.8, min_child_weight {1, 5}, num_class=3, eval_metric=mlogloss, "
         "scale_pos_weight-style balancing via sample weights per class. Final values are "
         "selected by tuning, not fixed a priori.")
    lead(doc, "How will you evaluate your model?",
         "Primary macro F1 on the held-out test set; per-class precision/recall with "
         "High-Risk recall reported as the headline business metric; confusion matrix to "
         "inspect Low/Medium/High confusions (Medium-High boundary errors matter most); "
         "multiclass ROC-AUC (one-vs-rest). We also audit fairness slices (per gender, "
         "race, age band) and record feature importance/SHAP for the champion to satisfy "
         "the interpretability goal.")

    doc.add_heading("Model Deployment", level=2)
    lead(doc, "What instance size will you use?",
         "Training: ml.m5.2xlarge (8 vCPU, 32 GB) - tabular data, no GPU needed; "
         "Processing/feature builds: ml.m5.xlarge; Batch Transform scoring: ml.m5.xlarge, "
         "scaled to ml.m5.2xlarge if the monthly cohort grows. Total monthly cost stays in "
         "single-digit dollars at this data volume; a real-time what-if endpoint, if "
         "added, would run on ml.m5.large.")
    lead(doc, "Will your model function as a batch or real time model? Why?",
         "Batch. The business use case is proactive care-management cohort selection on a "
         "weekly/monthly refresh cycle - there is no user-facing latency requirement, "
         "input is the full member population at once, and batch scoring (SageMaker Batch "
         "Transform) is simpler, cheaper, and easier to monitor than an always-on endpoint. "
         "A small real-time endpoint would only be justified for ad-hoc what-if lookups by "
         "care managers, which is out of scope for the initial release.")

    doc.add_heading("Model Monitoring", level=2)
    lead(doc, "How will you monitor your model?",
         "Two horizons. (1) Immediate: prediction-drift tracking - the distribution of "
         "predicted Low/Medium/High shares per scoring run against a baseline (PSI-style "
         "alerts), score-distribution statistics, and feature-attribution drift for the "
         "champion model. (2) Delayed: once outcome windows mature, recompute ground-truth "
         "labels and track macro F1, per-class recall (especially High-Risk), and the "
         "confusion matrix over time; alert on decay beyond a threshold to trigger "
         "retraining. All metrics land in CloudWatch with dashboards and alarms.")
    lead(doc, "How will you monitor your infrastructure?",
         "CloudWatch alarms on SageMaker job failures, processing/training/batch-transform "
         "duration and throughput anomalies, instance utilization, S3 bucket metrics, and "
         "pipeline execution status; dead-letter/SNS notification to the team on any "
         "failure, plus a daily pipeline health check in the team tracker.")
    lead(doc, "How will you monitor your data?",
         "A data-validation gate runs on every new extract before anything downstream "
         "executes: schema/shape tests, the 39-check suite (row counts, PK uniqueness, "
         "referential integrity), null-rate thresholds, and feature-distribution drift "
         "versus the training snapshot (per-column PSI with alerting thresholds). "
         "Observation-window coverage is tracked explicitly (the not-observed population "
         "is the main data-quality trap in this dataset), and labeling-rule versions are "
         "recorded with each scored cohort.")

    doc.add_heading("Model CI/CD", level=2)
    lead(doc, "What checkpoints will your CI/CD pipeline contain?",
         "SageMaker Pipelines with human approval steps: (1) data validation gate must "
         "pass; (2) unit/lint checks; (3) training job completes within budget; (4) "
         "evaluation gate - test macro F1 >= registered champion AND High-Risk recall >= "
         "floor, plus bias-audit sign-off; (5) model registered with metrics + lineage in "
         "the Model Registry; (6) staging batch transform on a holdout cohort, manually "
         "reviewed; (7) production promotion, with one-command rollback to the previous "
         "registered version.")
    lead(doc, "What tests will your CI/CD pipeline contain?",
         "Data tests (schema, null rates, distribution drift, window-coverage checks); "
         "code tests (feature-builder unit tests with hand-computed expectations on fixed "
         "persons, loader validation tests); model tests (reproducibility - identical "
         "metrics on a seeded rerun; evaluation regression - no metric below champion by "
         "more than tolerance; sanity - predicted class distribution within expected "
         "bounds; fairness - per-slice recall gaps under threshold); and integration tests "
         "(end-to-end batch scoring smoke test on a 1k sample with output schema checks).")

    # ---- 3. Security & Risks ----------------------------------------------- #
    doc.add_heading("3. Security Checklist, Privacy and Other Risks", level=1)
    lead(doc, "Will this store or process Personal Health Information (PHI)?",
         "No. The dataset is fully synthetic - DE-SynPUF was engineered by CMS specifically "
         "so that no real Medicare beneficiary's protected health information can be "
         "derived from it; the OMOP conversion preserves that property. No real diagnoses, "
         "dates of care, or clinical facts about any real person exist in the data.")
    lead(doc, "Will this store or process Personal Identifiable Information (PII)?",
         "No. Person identifiers are synthetic person_id values with no linkage to real "
         "individuals; state-level geography is the coarsest real attribute. Names, SSNs, "
         "addresses, and contact data are not present.")
    lead(doc, "Will user behavior be tracked and stored?",
         "No. This is an internal batch analytics system; it does not instrument or track "
         "end users.")
    lead(doc, "Will this store or process credit card information?",
         "No.")
    lead(doc, "If you answered yes to any of the above questions, please justify.",
         "Not applicable - all four answers are no. Defense-in-depth is still applied: S3 "
         "server-side encryption, blocked public access on project buckets, least-privilege "
         "IAM, and no credentials anywhere in code (the public source bucket is accessed "
         "with unsigned/anonymous reads).")
    lead(doc, "What S3 buckets will this application read from or write to?",
         "Read-only, anonymous: arn:aws:s3:::synpuf-omop (us-east-1) - the AWS Open Data "
         "source, prefixes cmsdesynpuf1k/ and cmsdesynpuf100k/. Project-private (write): "
         "aai-540-g1-final-project with raw/, processed/, artifacts/ prefixes, SSE-S3 "
         "encryption, versioning on artifacts/, and bucket policies restricting access to "
         "the pipeline's IAM role and team members.")
    lead(doc, "What data bias should be considered?",
         "Demographic skew (82.8% White, 55.6% female, mean age 71.7; under-65 members are "
         "disabled-eligible with distinct, higher utilization); geographic sparsity; "
         "synthetic-generation artifacts (claim-line duplication inflates condition counts; "
         "85% of visit types unmapped so utilization is measured with error; 10.7% of "
         "members unobserved in any period and thus unlabelable); and label-construction "
         "bias (terciles make the target a relative ranking, not an absolute clinical "
         "truth). Evaluation reports include per-slice metrics to make these visible.")
    lead(doc, "Will your model have potential for bias along sensitive features?",
         "Yes - race, gender, age, and state are available as features and utilization "
         "patterns differ across them. Mitigations: race is excluded from the primary "
         "model configuration (reported as a fairness audit slice instead); per-slice "
         "precision/recall are computed for gender, race, and age band with gap "
         "thresholds in the CI/CD gate; class weighting prevents the majority Low tier "
         "from dominating; and predictions are used only to prioritize outreach, not to "
         "deny coverage or set premiums, which limits harm from residual disparity.")
    lead(doc, "Are there any ethical concerns with the data or business problems that "
              "should be addressed?",
         "Three. (1) Allocation ethics: risk scores direct finite care-management "
         "attention, so systematically missing High-Risk members (especially in "
         "under-observed subgroups) has real opportunity cost - hence the recall floor and "
         "slice audits. (2) Transparency: the labeling rule is heuristic and versioned; "
         "stakeholders must understand scores indicate relative utilization burden, not "
         "clinical judgments. (3) Validity: synthetic data means demonstrated performance "
         "does not transfer to real members; the report states this limitation prominently "
         "and positions the system as a design template validated on synthetic claims "
         "before any real-data use.")

    # ---- 4. Future Enhancements -------------------------------------------- #
    doc.add_heading("4. Future Enhancements", level=1)
    numbered(doc, [
        "Move from synthetic to real data under a CMS Data Use Agreement: apply the same "
        "OMOP pipeline to CMS Limited Data Sets or the CMS Synthetic Claims or "
        "Transforming Collective Health Insights datasets, adding privacy-preserving "
        "techniques (de-identification review, differential-privacy aggregation for "
        "reporting) as real PHI enters the system.",
        "Add cost and acuity dimensions: incorporate claims payment/allowed-amount fields "
        "(available in the classic DE-SynPUF format and real CMS files) so the model "
        "predicts cost-weighted risk tiers and supports ROI-based care-management "
        "prioritization rather than utilization-count proxies.",
        "Model claim sequences instead of aggregates: replace hand-engineered counts with "
        "temporal architectures (LSTM/Transformer over visit/condition/drug sequences) or "
        "gradient boosting on learned embeddings, and frame the problem as survival/"
        "time-to-high-utilization analysis for finer-grained risk trajectories.",
        "Strengthen fairness and explainability: SHAP-based per-member explanation "
        "dashboards for care managers, fairness-constrained training (e.g., equalized-odds "
        "post-processing), and counterfactual 'what would move this member to Medium' "
        "reporting to guide intervention design.",
        "Productionize streaming and continuous evaluation: near-real-time scoring as "
        "claims land (Kinesis/SageMaker endpoints), automated ground-truth refresh as "
        "outcome windows close, and scheduled champion-challenger retraining with drift-"
        "triggered pipelines instead of monthly batch cadence.",
    ])

    doc.save(OUTPUT)
    print(f"Wrote {OUTPUT}")
    print(f"Diagram at {DIAGRAM}")


if __name__ == "__main__":
    build_document()
