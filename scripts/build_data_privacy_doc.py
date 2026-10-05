"""Build the Canvas discussion "Data Privacy" post (Data_Privacy_Discussion.docx).

Word-count-sensitive deliverable: the assignment limit is 700-800 words. The
script counts every word in the generated document (title + headings + table +
body + references) and fails loudly outside the range.

Run from the repo root:  .venv/bin/python scripts/build_data_privacy_doc.py
"""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "Data_Privacy_Discussion.docx"
WORD_MIN, WORD_MAX = 700, 800

ACCENT = RGBColor(0x1F, 0x4E, 0x79)

TITLE = (
    "Data Privacy in Healthcare Machine Learning: An Analysis of the "
    "CMS DE-SynPUF Dataset (AAI-540, Group 1)"
)

INTRO_HEADING = "Introduction"
INTRO_BULLETS = [
    "This report examines the data privacy posture of the dataset behind our "
    "team's project: a multiclass machine-learning classifier that assigns "
    "Medicare-like members a Low / Medium / High utilization-risk tier.",
    "The dataset is the CMS DE-SynPUF (Data Entrepreneurs' Synthetic Public "
    "Use File), a fully synthetic Medicare claims extract converted to the "
    "OMOP Common Data Model and published as a public AWS Open Data dataset.",
    "We trace the data lifecycle - data subject, data types, actors, "
    "collection method, purpose, and storage - and evaluate our collection "
    "and use against the two discussion prompts.",
]

TABLE_HEADING = "Data Lifecycle and Privacy Profile"
TABLE_ROWS = [
    ("Data subject",
     "100,000 synthetic Medicare fee-for-service beneficiaries (elderly "
     "cohort, mean age ~72). No real individuals; records were generated "
     "from real seed beneficiaries."),
    ("Data types",
     "Demographics (age, gender, race, state), ICD-9 diagnoses, procedures, "
     "drug exposures, visit/utilization events, mortality. No names, SSNs, "
     "addresses, or free-text clinical notes."),
    ("Actors",
     "CMS (original claims collection and synthesis); OHDSI community (OMOP "
     "CDM conversion); AWS (public S3 hosting); our team (download, "
     "validation, feature engineering)."),
    ("Method",
     "Original claims collected under statutory authority for Medicare "
     "payment and program operations; DE-SynPUF produced with disclosure-"
     "limitation techniques (variable reduction, suppression, substitution, "
     "imputation, date perturbation, coarsening); distributed as anonymous, "
     "credential-free S3 objects."),
    ("Purpose and consent",
     "Released for software development, researcher training, and safe "
     "data-mining innovation while protecting beneficiary PHI; distributed "
     "under a CMS data use agreement; consent is moot because the records "
     "are synthetic."),
    ("Storage and processing",
     "Local raw copies kept out of version control; records aggregated to a "
     "member-level feature table; only the processed CSV is committed. "
     "Planned AWS architecture specifies SSE-S3 encryption, blocked public "
     "access, and least-privilege IAM."),
]

PROMPT1_HEADING = "Prompt 1: Is Storage or Use Outside the Intended Collection Purpose?"
PROMPT1_BULLETS = [
    "No. Our use sits inside the stated collection purpose. CMS released "
    "DE-SynPUF explicitly so developers can create software that may later "
    "run on real CMS claims data and to support safe data-mining innovation "
    "while preserving beneficiary privacy (CMS, 2013).",
    "Training a utilization-risk classifier is precisely the class of "
    "application the release was designed for, so neither our storage nor "
    "our modeling use exceeds scope.",
    "One boundary we observe: CMS states the file has limited inferential "
    "value, so we report all findings as methodological exercises, never as "
    "conclusions about real beneficiaries.",
    "Storage is proportionate: raw extracts remain local and gitignored; "
    "only derived, aggregated features enter the repository.",
]

PROMPT2_HEADING = "Prompt 2: Is Use Beyond the Data Subject's Consented (or Unconsented) Purpose?"
PROMPT2_BULLETS = [
    "Strictly, no data subject exists in our file: every record is "
    "synthetic, so no living person's consented or unconsented purpose is "
    "engaged by our processing.",
    "Upstream, real Medicare beneficiaries never consented in the research "
    "sense; their claims were collected under statutory authority for "
    "treatment, payment, and health-care operations, not voluntary research "
    "participation.",
    "CMS resolved that tension through synthetic substitution rather than "
    "consent or de-identification alone: records were seeded from real "
    "beneficiaries, then altered via disclosure limitation until no "
    "individual's data remained (CMS, 2013b).",
    "Our residual obligation is continuity: we must not portray "
    "synthetic-derived results as knowledge about real patients, and the "
    "moment real claims data enters the pipeline, HIPAA obligations - data "
    "use agreements, minimum-necessary access, audit controls - attach "
    "(HHS Office for Civil Rights, 2012).",
]

RISK_HEADING = "Residual Risks and Safeguards"
RISK_BULLETS = [
    "The main residual risks are methodological, not legal: models tuned on "
    "synthetic demographics may inherit distributions unrepresentative of "
    "real members, and fairness audits on this cohort cannot certify "
    "fairness in production.",
    "Safeguards already in place: credential-free anonymous download, no "
    "PHI in the repository, and HIPAA-aligned encryption, access control, "
    "and monitoring specified in our design document for the planned "
    "SageMaker deployment.",
]

CONCLUSION_HEADING = "Conclusion"
CONCLUSION_BULLETS = [
    "DE-SynPUF is a deliberately privacy-engineered public use file: "
    "synthetic subjects, cryptographic identifiers, and a data use "
    "agreement that contemplates exactly our use (Amazon Web Services, "
    "n.d.).",
    "Within this project, storage and processing stay within the intended "
    "collection purpose, and no consented scope is exceeded because no "
    "data subject remains in the data.",
    "The privacy discipline that matters going forward is procedural: keep "
    "the synthetic/real boundary explicit in every deliverable, and treat "
    "the HIPAA-aligned controls in our design as mandatory the moment real "
    "claims data enters the pipeline.",
]

REFERENCES_HEADING = "References"
REFERENCES = [
    "Amazon Web Services. (n.d.). CMS 2008-2010 Data Entrepreneurs' "
    "Synthetic Public Use File (DE-SynPUF) in OMOP Common Data Model "
    "[Data set]. Registry of Open Data on AWS. "
    "https://registry.opendata.aws/cmsdesynpuf-omop/",
    "Centers for Medicare & Medicaid Services. (2013a). CMS 2008-2010 Data "
    "Entrepreneurs' Synthetic Public Use File (DE-SynPUF). "
    "https://www.cms.gov/data-research/statistics-trends-and-reports/"
    "medicare-claims-synthetic-public-use-files/cms-2008-2010-data-"
    "entrepreneurs-synthetic-public-use-file-de-synpuf",
    "Centers for Medicare & Medicaid Services. (2013b). Linkable "
    "2008-2010 Medicare DE-SynPUF: Frequently asked questions. "
    "https://www.cms.gov/files/document/de-10-frequently-asked-questions.pdf",
    "Observational Health Data Sciences and Informatics. (n.d.). OMOP "
    "common data model. https://ohdsi.github.io/CommonDataModel/",
    "U.S. Department of Health and Human Services, Office for Civil "
    "Rights. (2012). Guidance on de-identification of protected health "
    "information. https://www.hhs.gov/hipaa/for-professionals/privacy/"
    "special-topics/de-identification/index.html",
]


def set_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    for name, size in [("Heading 1", 14), ("Heading 2", 12)]:
        style = doc.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = ACCENT
        style.font.bold = True


def count_words(doc: Document) -> int:
    words = 0
    for para in doc.paragraphs:
        words += len(para.text.split())
    for tbl in doc.tables:
        for row in tbl.rows:
            for cell in row.cells:
                words += len(cell.text.split())
    return words


def add_bullets(doc: Document, items: list[str]) -> None:
    for text in items:
        doc.add_paragraph(text, style="List Bullet")


def build() -> None:
    doc = Document()
    set_styles(doc)

    title = doc.add_paragraph()
    run = title.add_run(TITLE)
    run.bold = True
    run.font.size = Pt(15)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading(INTRO_HEADING, level=2)
    add_bullets(doc, INTRO_BULLETS)

    doc.add_heading(TABLE_HEADING, level=2)
    table = doc.add_table(rows=1, cols=2)
    table.style = "Light Grid Accent 1"
    header = table.rows[0].cells
    header[0].text = "Dimension"
    header[1].text = "Detail"
    for dim, detail in TABLE_ROWS:
        cells = table.add_row().cells
        cells[0].text = dim
        cells[1].text = detail

    doc.add_heading(PROMPT1_HEADING, level=2)
    add_bullets(doc, PROMPT1_BULLETS)

    doc.add_heading(PROMPT2_HEADING, level=2)
    add_bullets(doc, PROMPT2_BULLETS)

    doc.add_heading(RISK_HEADING, level=2)
    add_bullets(doc, RISK_BULLETS)

    doc.add_heading(CONCLUSION_HEADING, level=2)
    add_bullets(doc, CONCLUSION_BULLETS)

    doc.add_heading(REFERENCES_HEADING, level=2)
    for ref in REFERENCES:
        doc.add_paragraph(ref)

    total = count_words(doc)
    print(f"Word count: {total} (range {WORD_MIN}-{WORD_MAX})")
    if not WORD_MIN <= total <= WORD_MAX:
        raise SystemExit(f"FAIL: {total} words outside {WORD_MIN}-{WORD_MAX}")

    doc.save(OUTPUT)
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    build()
