"""Build the filled AAI-540 Weekly Teamwork Tracker (Team1_Weekly_Teamwork_Tracker.docx).

Recreates the college tracker template structure and fills per-member task
assignments for Modules 2, 3, and 6 (Modules 4-5 left blank for weekly
updates), plus the module-by-module guidance appendix.

Run from the repo root:  .venv/bin/python scripts/build_tracker_doc.py
"""

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "Team1_Weekly_Teamwork_Tracker.docx"

ACCENT = RGBColor(0x1F, 0x4E, 0x79)

TEAM = [
    ("Team Member 1", "Sourangshu Pal"),
    ("Team Member 2", "Riyaz Khorasi"),
    ("Team Member 3", "Tanvi Singh"),
]

MODULES = [
    {
        "title": "Module 2  Weekly Teamwork Tracker",
        "intro": (
            "Please copy and share a single Team Tracker document for your team. "
            "Then, individually update by Day 7 of Modules 2 thru 6. This should take "
            "1-2 minutes for each person to update individually.\n"
            "For Module 2, update your Design Document with your Team Tracker link"
        ),
        "summary": (
            "2 team meetings (kickoff + mid-week check-in) and daily asynchronous "
            "updates via Asana comments and team chat. (Update this count each week.)"
        ),
        "theme": "Week 2 focus - Team Project Proposal, Data Engineering",
        "tasks": [
            [
                "Finalize the team RFC (problem statement: multiclass member "
                "utilization-risk classification for a hypothetical UnitedHealthcare "
                "scenario) and post it to the Canvas discussion forum.",
                "Own the ML Design Document draft: Project Scope, Data Sources, and "
                "Feature Engineering sections, grounded in the CMS DE-SynPUF (OMOP CDM) "
                "dataset from the public s3://synpuf-omop bucket.",
                "Insert the Team Tracker link into the Design Document (Module 2 "
                "requirement) and post the tracker link to the Team Project Update.",
            ],
            [
                "Set up the project S3 datalake: aai-540-g1-final-project bucket with "
                "raw/, processed/, and artifacts/ prefixes, SSE-S3 encryption, blocked "
                "public access, and least-privilege IAM role for the pipeline.",
                "Configure AWS Glue crawler + Athena tables to catalog and query the 17 "
                "OMOP CDM tables (person, visit_occurrence, condition_occurrence, "
                "drug_exposure, etc.).",
                "Validate Athena queries over the person, visit_occurrence, and "
                "condition_occurrence tables and record table row counts in the design "
                "document.",
            ],
            [
                "Create the Asana board: workstreams per module (Proposal & Data Eng, "
                "EDA & Features, Model Dev, Monitoring, CI/CD) with task assignments "
                "and due dates mirroring this tracker.",
                "Set up the GitHub repository: folder structure, .gitignore, README with "
                "environment setup and reproduction steps.",
                "Run the dataset download into the datalake: aws s3 sync --no-sign-request "
                "from s3://synpuf-omop (cmsdesynpuf1k and cmsdesynpuf100k prefixes) and "
                "verify file counts/sizes against the bucket listing.",
            ],
        ],
        "roadblocks": [
            "None blocking. Watch: unsigned (--no-sign-request) access to the public "
            "bucket must not pick up stale local AWS credentials.",
            "Glue crawler may need explicit CSV/bz2/gz classifier and header settings to "
            "infer OMOP schemas cleanly.",
            "None blocking. Watch: keeping Asana due dates aligned with Canvas module "
            "deadlines (Day 7 of each module).",
        ],
    },
    {
        "title": "Module 3 Weekly Teamwork Tracker",
        "intro": (
            "List briefly your weekly contribution to teamwork and any issues or "
            "roadblocks. Post the link to the Team Project Update in Canvas for this "
            "module. (This is a 2-3 minute task)."
        ),
        "summary": (
            "2 team meetings (planning + review of EDA findings) plus asynchronous "
            "collaboration on Asana. (Update this count each week.)"
        ),
        "theme": "Week 3 focus - Training Data and Feature Engineering",
        "tasks": [
            [
                "Port the exploratory data analysis to a SageMaker notebook: population "
                "structure, visit mix (incl. 85% unmapped visit-type caveat), chronic-"
                "condition prevalence, drug utilization, 1k-vs-100k representativeness.",
                "Run the automated data-validation gate (39 checks: row counts, primary-"
                "key uniqueness, person_id referential integrity) on each new extract.",
                "Document EDA findings and data quirks in the analysis report section of "
                "the design document.",
            ],
            [
                "Initialize the SageMaker Feature Store and define feature groups: "
                "demographics, utilization counts, chronic-condition flags, pharmacy "
                "intensity, and utilization breadth.",
                "Load the member-level feature table (100k persons x 24 features) into "
                "the feature groups with person_id as the record key.",
                "Verify offline-store ingestion (row counts, null rates, dtype checks) "
                "and record the feature-group ARN list in the design document.",
            ],
            [
                "Build the feature-engineering pipeline (chronic flags via ICD-9 "
                "chapters, age bands, observation-month exposure) producing the "
                "modeling-ready member-level table.",
                "Create the dataset split per course guidance: ~40% training, ~10% "
                "validation, ~10% test, ~40% reserved as 'production' holdout, stratified "
                "by the constructed risk class with a fixed seed.",
                "Apply the temporal-eligibility filter (members observable in both the "
                "2008 history window and 2009-2010 outcome window, 83.3% of the "
                "population) before splitting; document the split and post the tracker.",
            ],
        ],
        "roadblocks": [
            "Temporal-split eligibility reduces the labeled pool to ~83k members; "
            "document the exclusion so the instructor sees the reasoning.",
            "Feature Store ingestion throughput for 100k records may need batched "
            "PutRecord calls; monitor for throttling.",
            "None blocking. Watch: keeping the split seed and filters scripted so the "
            "40% production reserve stays untouched until monitoring week.",
        ],
    },
    {
        "title": "Module 4 Weekly Teamwork Tracker",
        "intro": (
            "List briefly your weekly contribution to teamwork and any issues or "
            "roadblocks. Post the link individually to the Team Project Update in "
            "Canvas for this module. (This is a 2-3 minute task)."
        ),
        "summary": "(To be updated during Module 4.)",
        "theme": "Week 4 focus - Model Development and Deployment (Check-In #2)",
        "tasks": None,  # blank table for the future week
        "roadblocks": None,
    },
    {
        "title": "Module 5 Weekly Teamwork Tracker",
        "intro": (
            "List briefly your weekly contribution to teamwork and any issues or "
            "roadblocks. Post the link to the Team Project Update in Canvas for this "
            "module. (This is a 2-3 minute task)."
        ),
        "summary": "(To be updated during Module 5.)",
        "theme": "Week 5 focus - Monitoring",
        "tasks": None,
        "roadblocks": None,
    },
    {
        "title": "Module 6 Weekly Teamwork Tracker",
        "intro": (
            "List briefly your weekly contribution to teamwork and any issues or "
            "roadblocks. Post the link to the Team Project Update in Canvas for this "
            "module. (This is a 2-3 minute task)."
        ),
        "summary": (
            "2 team meetings (pipeline design review + final end-to-end run review) plus "
            "daily Asana coordination during the pipeline build-out. (Update this count "
            "each week.)"
        ),
        "theme": "Week 6 focus - CI/CD",
        "tasks": [
            [
                "Build the SageMaker Pipelines CI/CD workflow: data-validation gate -> "
                "training -> evaluation gate -> model registration -> batch deployment, "
                "with human approval steps.",
                "Implement the evaluation checkpoint: macro F1 must beat the registered "
                "champion and High-Risk recall must stay above the agreed floor; wire "
                "one-command rollback to the previous model version.",
                "Link the pipeline execution history and Model Registry versions in the "
                "design document.",
            ],
            [
                "Improve the initial model: hyperparameter tuning of XGBoost against the "
                "Random Forest and Logistic Regression baselines (validation macro F1).",
                "Retrain the champion through the CI/CD pipeline using the training "
                "split, with evaluation on the test split as the pipeline requires.",
                "Prepare the champion-vs-challenger comparison table (metrics, confusion "
                "matrix, fairness slices) for the final report.",
            ],
            [
                "Implement pipeline tests/checkpoints: data validation gate, seeded "
                "reproducibility check, integration smoke test scoring the 1k sample, "
                "and a bias audit across gender/race/age slices.",
                "Run the full pipeline end-to-end on the processed dataset and capture "
                "timing, cost, and evaluation artifacts.",
                "Update the design document with the CI/CD section evidence and post the "
                "tracker link to the Module 6 Team Project Update.",
            ],
        ],
        "roadblocks": [
            "Pipeline runtime/cost on the 100k dataset may require ml.m5.xlarge caps or "
            "spot instances for tuning jobs.",
            "Evaluation-gate thresholds (macro F1 floor, recall floor) may need one "
            "calibration run before they are strict enough to be meaningful.",
            "Smoke-test data (1k sample) must use the same feature contract as the 100k "
            "pipeline; schema drift here fails the integration checkpoint.",
        ],
    },
]

GUIDANCE = [
    ("Week 1", "Individual Project Proposal/Group Sign-ups (Self-Enroll)", [
        "Propose a problem that is solvable with a Machine Learning System.",
        "Write a RFC (Request for Comments) Document.",
        "After completing your RFC document, post your RFC in the canvas discussion forum.",
        "Comment on 2-3 projects that you find interesting and try to find a partner.",
        "Confirm a team of 2-3 people to work on your final project with (Day 7).",
        "Complete Group Sign-ups in Canvas (Day 7, ALL students, even those working alone)",
        "Copy/share, and then update/post this tracker to the Team Project Update in Canvas (Day 7).",
    ]),
    ("Week 2", "Team Project Proposal, Data Engineering", [
        "With your team, agree on a problem that is solvable with a Machine Learning System.",
        "Begin writing your Design Document. (See AAI-540 Design Document Template).",
        "Set up an Asana board and Github to manage your project.",
        "Collect a raw data set and store in an S3 Datalake.",
        "Set up Athena tables to enable cataloging and querying of your data.",
        "Complete indicated fields and update the design document draft to submit to the Team Assignment (see Canvas Assignment instructions).",
    ]),
    ("Week 3", "Training Data and Feature Engineering", [
        "Perform exploratory data analysis on your data in a SageMaker notebook.",
        "Initialize a feature store.",
        "Design the feature groups needed for your system.",
        "Perform feature engineering on raw data and store features in feature groups.",
        "Split your feature data into training (~40%), test (~10%) validation (~10%) datasets.",
        "Reserve some data for \"production data\" (~40%).",
        "Update and post the tracker to the Team Project Update.",
    ]),
    ("Week 4", "Model Development and Deployment", [
        "Check In #2 (See Check-In Outline for Deliverables)",
        "Set up a benchmark model in SageMaker.",
        "The benchmark model should be simple, it can be a simple heuristic or a model with just a couple of features.",
        "The idea here is to have a baseline for further improvements, and to get a minimum viable product out before you start improving your system.",
        "Build, train and debug your ML model in a SageMaker model.",
        "This is your first real iteration on your model, it doesn't need to be perfect, you may want to revisit model development again once we implement CI/CD in module 6",
        "Evaluate your model and compare against your simple benchmark model.",
        "Deploy your model to SageMaker (Batch process or Real Time Endpoint).",
        "Link your initial findings and codebase in your ML Design Document.",
        "Update and post the tracker to the Team Project Update.",
    ]),
    ("Week 5", "Monitoring", [
        "Implement model monitors on your ML system.",
        "Implement data monitors on your ML system.",
        "Implement infrastructure monitors on your ML system.",
        "Create a monitoring dashboard for your ML endpoint/job on Cloudwatch",
        "Generate model and data reports on SageMaker.",
        "Update and post the tracker to the Team Project Update.",
    ]),
    ("Week 6", "CI/CD", [
        "Implement CI/CD Pipeline to automate training, evaluation, and deployment.",
        "CI/CD pipeline should have checkpoints to evaluate model performance, model code and system integration.",
        "Your CI/CD pipeline should train with training data, and evaluate with testing data.",
        "You should try to improve your initial model and run it through your CI/CD pipeline.",
        "Update and post the tracker to the Team Project Update.",
    ]),
    ("Week 7", "Wrap Up Final Project", [
        "Group Deliverables Due:",
        "Deliverable 1: ML Design Document",
        "Deliverable 2: ML System Operation Validation",
        "Deliverable 3: Codebase GitHub Repository",
        "Individual Deliverable Due:",
        "Peer Evaluation Form",
    ]),
]


def set_styles(doc):
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    for name, size in [("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11.5)]:
        style = doc.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = ACCENT
        style.font.bold = True


def bullets_in_cell(cell, items):
    cell.text = ""
    first = True
    for item in items:
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        p.style = "List Bullet"
        p.text = item


def contribution_table(doc, module):
    """3-column member table: name header row, contributions row, roadblocks row."""
    table = doc.add_table(rows=0, cols=3)
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    header = table.add_row().cells
    for i, (label, name) in enumerate(TEAM):
        header[i].text = ""
        p = header[i].paragraphs[0]
        run = p.add_run(f"{label}\n{name}")
        run.bold = True

    contrib = table.add_row().cells
    block = table.add_row().cells
    for i in range(3):
        label = "List of contributions"
        tasks = module["tasks"][i] if module["tasks"] else []
        bullets_in_cell(contrib[i], [label] if not tasks else tasks)
        if tasks:
            # prepend a bold-ish label line by keeping label as first bullet
            contrib[i].paragraphs[0].text = f"{label}: " + contrib[i].paragraphs[0].text
        rb = module["roadblocks"][i] if module["roadblocks"] else ""
        block[i].text = f"Comments/ Roadblocks\n{rb}" if rb else "Comments/ Roadblocks\n(to be updated)"
    return table


def build():
    doc = Document()
    set_styles(doc)

    title = doc.add_paragraph()
    run = title.add_run("AAI-540 Group 1 Weekly Team Tracker")
    run.bold = True
    run.font.size = Pt(20)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_paragraph(
        "Please copy and share this Google document with your project teammates. "
        "Then, set general access sharing settings to \"University of San Diego\" and "
        "\"Commenter\" so your instructor can view or comment."
    )

    doc.add_heading("Project Group #: AAI 540 Group 1", level=2)
    bullets = doc.add_paragraph()
    for line in [
        "Team member 1: Sourangshu Pal",
        "Team member 2: Riyaz Khorasi",
        "Team member 3: Tanvi Singh",
        "Asana Board link: https://app.asana.com/1/952672460738672/project/1218443371995338/board/1218443464762200",
        "Project GitHub link: https://github.com/University-San-Diego-MAAI/AAI-540-Group-1-Final-Project",
    ]:
        doc.add_paragraph(line, style="List Bullet")

    doc.add_heading("In this Document:", level=2)
    doc.add_paragraph(
        "Teamwork Tracker: For accountability, each team member should list briefly "
        "your weekly responsibilities whether you have completed the tasks or not. The "
        "point of this is to disperse work evenly among team members and ensure "
        "equitable work and continuous communication. Include the Team Tracker link in "
        "your Design Document draft in Module 2. Post the Team Tracker link to the Team "
        "Project Update in Canvas (Modules 3 thru 6.) This is a 2-3 minute task.",
        style="List Bullet",
    )
    doc.add_paragraph(
        "Module Steps: Recommended Module-by-Module project steps are found on the "
        "last pages in this document.",
        style="List Bullet",
    )

    doc.add_page_break()
    doc.add_heading("Weekly Teamwork Tracker", level=1)

    for module in MODULES:
        doc.add_heading(module["title"], level=2)
        for chunk in module["intro"].split("\n"):
            doc.add_paragraph(chunk)
        doc.add_paragraph(module["theme"], style="Intense Quote")
        doc.add_heading("Teamwork Summary", level=3)
        doc.add_paragraph(
            "How many times have your team members met, exchanged project "
            "communications, or collaborated on work in the last week?"
        )
        doc.add_paragraph(module["summary"])
        doc.add_heading("Weekly Project Snapshot - Individual Contributions", level=3)
        doc.add_paragraph(
            "Summarize your responsibilities and progress for the week in the table "
            "below."
        )
        contribution_table(doc, module)
        doc.add_paragraph()
        doc.add_page_break()

    doc.add_heading("Module-by-Module Steps", level=1)
    doc.add_heading("General Guidance", level=2)
    doc.add_paragraph(
        "Below are general steps as guidance for expected teamwork for each module. "
        "Refer to the specific details in Canvas for the Team Project Update each week."
    )
    for week, subtitle, items in GUIDANCE:
        doc.add_heading(f"{week} - {subtitle}", level=3)
        for item in items:
            doc.add_paragraph(item, style="List Bullet")

    doc.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()
