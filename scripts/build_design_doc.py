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
def figure_placeholder(doc, n: int, title: str, filename: str) -> None:
    """Bordered, shaded placeholder the team replaces with the rendered figure."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    ph = doc.add_paragraph()
    ph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = ph.add_run(f"[  INSERT FIGURE {n} - {title}  ]\n"
                     f"(attach: assets/v2/{filename})")
    run.font.size = Pt(10)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    p_pr = ph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), "EAF1F8")
    p_pr.append(shd)
    para(doc, f"Figure {n}. {title}.", italic=True, size=9)


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
        ("Version: ", "2.0 - 16 October 2026 (supersedes v1; as-implemented updates: "
         "course-mandated 40/10/10/40 split, measured 85.3% eligibility, Feature Store + "
         "Athena added, deployment/monitoring/CI-CD as-built notes, measured results)"),
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
         "confused. As implemented: stratified course-mandated split with seeded grid runs "
         "over the design hyperparameters (3 XGBoost configs + LogReg + RF, fixed seed 42), "
         "champion selected on validation macro F1 under a High-Risk recall floor, confirmed "
         "once on the untouched test set.")
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
         "The system is an AWS-native ML pipeline (as implemented, October 2026). Raw OMOP "
         "CDM tables are mirrored from the public synpuf-omop bucket into the project's S3 "
         "datalake prefix. The 39-check validation gate and feature engineering run as "
         "deterministic, memory-safe scripts (src/load_synpuf.py, build_features.py, "
         "build_training_data.py), validated locally before upload; a cloud-side SageMaker "
         "Processing port is designed and documented but was deferred as a stretch goal. "
         "Athena catalogs raw and processed data for SQL validation; the member-level table "
         "(82,624 labeled members) is ingested into a SageMaker Feature Store feature "
         "group whose offline store feeds dataset extraction. SageMaker Training jobs train "
         "Logistic Regression, Random Forest, and XGBoost; the champion is registered in "
         "the Model Registry behind a human approval gate. Two deployment paths serve "
         "scoring: production cohorts run as SageMaker Batch Transform producing ranked "
         "worklists; a zero-idle-cost SageMaker Serverless Inference endpoint powers an "
         "operations console for live what-if predictions and the video demonstration. "
         "Monitoring is implemented as a data-quality baseline job plus live-evaluation "
         "model-quality and bias evidence reports in S3, with CloudWatch dashboards and "
         "alarms for infrastructure health - the classic schedule-based monitor APIs having "
         "been retired by AWS into maintenance mode during the project (see Model "
         "Monitoring).")
    figure_placeholder(doc, 1, "System architecture (as implemented)",
                       "fig1_system_architecture.png")

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
         "Three S3 prefixes - raw/ (mirrored public data, immutable), processed/ "
         "(member-level training table + split CSVs), artifacts/ (trained models, evaluation "
         "and monitoring reports, scoring outputs) - under the account's default SageMaker "
         "bucket (sagemaker-us-east-1-<account>/aai-540-g1/). Design change from v1: a "
         "dedicated project bucket was replaced by the default bucket + prefix to avoid "
         "new-bucket IAM friction in a personal/Learner-Lab account; the prefix layout, "
         "SSE-S3 encryption, blocked public access, and least-privilege access are "
         "unchanged. Development uses a git-tracked local mirror under data/ with data/ "
         "gitignored except the committed feature-table artifact.")
    lead(doc, "How are features stored and served (Feature Store)?",
         "The training table is ingested into a SageMaker Feature Store feature group "
         "(aai-540-g1-member-features; record id person_id; event time at ingestion). "
         "Bulk ingestion uses a client-side threaded put_record pool - the SDK's serial "
         "ingest took >1 hour for 82,624 records and the assumed batch API does not exist "
         "(both measured; notes in notebook 02). The online store serves point lookups "
         "(get_record spot-checks return MATCH); the offline store lands in Glue/Athena, "
         "where split-filtered queries materialize the train/validation/test/production "
         "datasets - the Feature Store is therefore both the serving layer for the demo UI "
         "and the extraction layer for training (course Update 3.1, steps 4-5).")
    lead(doc, "How is the data cataloged and queried (Athena)?",
         "An Athena workgroup (aai540_g1, SSE-S3 results in the project prefix) and "
         "database host external tables over the raw and processed prefixes "
         "(scripts/setup_athena.py). Validation queries - member counts, per-class "
         "balances, age-band x class cross-tabs, chronic-flag prevalence by class (CHF "
         "climbs 11.7% -> 28.2% -> 51.5% across Low/Medium/High) - are the SQL-level data "
         "checks executed before any training (Update 3.1, step 2).")
    figure_placeholder(doc, 2, "Data engineering and Athena cataloging",
                       "fig2_data_lifecycle_athena.png")
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
         "Two layers. First, a temporal eligibility filter: only members whose observation "
         "period overlaps both the 2008 history window and the 2009-2010 outcome window "
         "(start <= 2008-12-31 AND end >= 2009-01-01) enter the labeled set - 85,323 "
         "members (85.3%; the v1 estimate of 83.3% was superseded by the measured figure). "
         "Second, the course-mandated split from Team Update 3.1: stratified 40% train / "
         "10% validation / 10% test / 40% production, per class, seed 42 (supersedes the "
         "v1 70/15/15 design). The 40% production reserve is scored blind and doubles as "
         "the 'production data' whose known labels feed the Module-5 model-quality monitor. "
         "Members who die during the outcome window are censored (2,699 excluded), leaving "
         "82,624 labeled members with zero null cells and 33/33/33 class balance.")
    figure_placeholder(doc, 3, "Temporal windows, rule-based labels, and the 40/10/10/40 split",
                       "fig3_temporal_split_labels.png")
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
         "Via SageMaker Training jobs fed by the training table in S3 (single 'train' "
         "channel; train.py derives the splits internally and early-stops XGBoost on the "
         "validation slice). As implemented: five seeded runs over the design grid - "
         "logreg-c1, rf-d16-l5, and three XGBoost configurations (eta 0.1/0.05, depth "
         "6/8, min_child_weight 5/1) - tracked as Experiments, with the champion chosen on "
         "validation macro F1 subject to the High-Risk recall floor. The full 20-30-"
         "candidate HyperparameterTuner with k-fold CV remains the designed tuning "
         "upgrade; a one-line container quirk worth recording: single-character "
         "hyperparameter keys are rendered with one dash by the training container, so the "
         "argument is named --c-reg rather than --C.")
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
         "multiclass ROC-AUC (one-vs-rest). Measured results are summarized in the Results "
         "section; fairness slices (per gender, race, age band) and booster feature "
         "importance for the champion are computed in notebook 04 - history-window "
         "utilization and chronic flags dominate, with no leakage columns among the top "
         "features.")
    figure_placeholder(doc, 4, "Training, evaluation, and champion selection (measured)",
                       "fig4_training_evaluation.png")

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
         "As implemented, a real-time path was ADDED for ad-hoc what-if lookups and the "
         "course demonstration: a SageMaker Serverless Inference endpoint (2048 MB, max "
         "concurrency 10) fronted by an S3-hosted operations console, API Gateway HTTP API, "
         "and a Lambda proxy. Rationale: it scales to zero, so the always-in-AWS demo costs "
         "~$0 at rest and <$1 on recording day; the accepted trade-offs are 5-8 s cold "
         "starts (absorbed by a retry in the Lambda) and the absence of data capture on "
         "the serverless tier.")
    figure_placeholder(doc, 5, "Deployment paths: batch production scorer + serverless demo endpoint",
                       "fig5_deployment_paths.png")

    doc.add_heading("Model Monitoring", level=2)
    lead(doc, "How will you monitor your model? (as implemented)",
         "Two horizons, delivered as explicit evidence jobs rather than scheduled monitors "
         "for a blocking reason discovered mid-project: AWS retired the classic monitoring "
         "schedule APIs (CreateDataQualityJobDefinition, CreateModelQualityJobDefinition) "
         "into MAINTENANCE MODE - unavailable to new customers - so recurring Clarify "
         "schedules cannot be created on a fresh account. (1) Immediate: a "
         "DefaultModelMonitor baseline job profiles the training distribution "
         "(statistics.json + constraints.json in S3); the champion is live-evaluated "
         "against 400 held-out members through the endpoint (model_quality_report.html/.json "
         "- macro F1, per-class metrics, confusion matrix); a one-off Clarify bias report "
         "was attempted with the fallback documented. (2) Delayed: the 40% production "
         "reserve has known labels, so macro-F1/per-class-recall decay tracking is a "
         "recomputation job, ready to schedule the moment AWS re-opens the APIs. "
         "Prediction-drift PSI schedules remain designed-future.")
    figure_placeholder(doc, 6, "Monitoring architecture (as implemented, with maintenance-mode finding)",
                       "fig6_monitoring_architecture.png")
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
    lead(doc, "What checkpoints will your CI/CD pipeline contain? (as implemented)",
         "Implemented scope (src/pipeline.py): a SageMaker Pipeline (aai-540-g1-"
         "utilization-risk) with two steps - TrainingStep running train.py with the "
         "champion configuration, then RegisterModel into the member-utilization-risk "
         "group with ModelApprovalStatus=PendingManualApproval. The human approval gate is "
         "the registry status flip; rollback is repointing the batch transform at the "
         "previous package version. Both execution states were demonstrated and recorded: "
         "demo-pipeline-ok Succeeded (green DAG) and demo-pipeline-fail Failed (red DAG, "
         "invalid eta argument aborts the train step before anything downstream runs). "
         "Designed-but-not-yet-wired: the automated evaluation gate (ConditionStep: test "
         "macro F1 >= champion AND High-Risk recall >= floor), unit/lint checks, staging "
         "batch review, and SNS notifications - evaluation currently runs in notebook 04 "
         "and is the first CI/CD hardening task on the roadmap.")
    figure_placeholder(doc, 7, "CI/CD pipeline: DAG, human approval gate, and both execution states",
                       "fig7_cicd_pipeline.png")
    lead(doc, "What tests will your CI/CD pipeline contain?",
         "Data tests (schema, null rates, distribution drift, window-coverage checks); "
         "code tests (feature-builder unit tests with hand-computed expectations on fixed "
         "persons, loader validation tests); model tests (reproducibility - identical "
         "metrics on a seeded rerun; evaluation regression - no metric below champion by "
         "more than tolerance; sanity - predicted class distribution within expected "
         "bounds; fairness - per-slice recall gaps under threshold); and integration tests "
         "(end-to-end batch scoring smoke test on a 1k sample with output schema checks).")

    doc.add_heading("Results (measured, October 2026)", level=2)
    bullets(doc, [
        "Benchmarks (Team Update 4.1 step 1): majority-class macro F1 0.168; simple "
        "visits-tercile heuristic macro F1 0.530 with High-Risk recall 0.640.",
        "Champion: xgb-e005-d6-w5 (XGBoost 1.7-1, eta 0.05, depth 6, min_child_weight 5) - "
        "held-out test macro F1 0.7743, High-Risk recall 0.8468, multiclass ROC-AUC 0.916; "
        "LogReg 0.763 / RF ~0.75 also clear both benchmarks.",
        "Batch scoring (production path): 33,048 reserved production members scored via "
        "Batch Transform; output distribution Low 33.3% / Medium 34.5% / High 32.2%; "
        "ranked High-Risk worklist published to S3.",
        "Feature Store: 82,624 records ingested (0 failures); online lookups and offline "
        "Athena extraction verified.",
        "Monitoring: data-quality baseline job completed (statistics + constraints); "
        "live-evaluation model-quality report published (macro F1 0.774 on 400 endpoint "
        "scored members); CloudWatch dashboard + 2 infra alarms; maintenance-mode "
        "constraint documented.",
        "Demo: serverless endpoint + operations console end-to-end; CI/CD DAGs recorded in "
        "both success and failure states; total project AWS spend for the build+demo "
        "window under $25 (personal account, shared with unrelated workloads).",
        "Standing caveat: all findings are methodological exercises on synthetic data - "
        "they do not transfer to real Medicare beneficiaries.",
    ])

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
