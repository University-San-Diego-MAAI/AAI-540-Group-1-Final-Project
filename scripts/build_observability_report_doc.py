"""Build the Canvas discussion 5.1 "ML System Observability" technical report
(Discussion_5.1_ML_System_Observability_Report.docx).

Word-count-sensitive deliverable: target 600-700 words, hard cap 800. The
script counts every word in the generated document (title + byline + headings
+ bullets + references) and fails loudly outside the range.

Run from the repo root:  .venv/bin/python scripts/build_observability_report_doc.py
"""

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "Discussion_5.1_ML_System_Observability_Report.docx"
WORD_MIN, WORD_MAX = 600, 700

ACCENT = RGBColor(0x1F, 0x4E, 0x79)

TITLE = "ML System Observability: Diagnosing and Monitoring Hallucinations in a Production LLM Chatbot"
BYLINE = "Sourangshu Pal (Group 1) | AAI-540 | University of San Diego"

INTRO_HEADING = "1. Introduction"
INTRO_BULLETS = [
    "Scenario: under CEO pressure we shipped an early LLM chatbot publicly; "
    "within days it trended on Twitter for hallucinating at scale.",
    "A hallucination is generated text that is plausible but factually "
    "incorrect or nonsensical - fluent output the user cannot trust "
    "unverified (Smith, 2023; Ji et al., 2023).",
    "Model-level fixes such as RLHF reduce but do not eliminate "
    "hallucinations and cannot surface new failure modes in production "
    "(Smith, 2023). The missing capability is observability: continuous "
    "measurement of data, model, and system behavior, turning failures into "
    "telemetry rather than headlines (Beyer et al., 2018; Amazon Web "
    "Services, n.d.).",
]

GAPS_HEADING = "2. Why the Launch Failed: Observability Gaps"
GAPS_BULLETS = [
    "No pre-release groundedness testing on red-team prompts or per-topic suites.",
    "No production logging: the team learned of the failure from social "
    "media, not its own telemetry.",
    "Silent failure mode: hallucinated answers return HTTP 200 with "
    "confident prose, so error-rate and uptime monitoring showed a healthy service.",
    "No labeled ground-truth sample to quantify the hallucination rate or "
    "slice it by topic or cohort.",
    "Deployment mismatch: LLMs suit uses where errors are not high impact "
    "(Smith, 2023); a public chatbot is high impact by reach.",
]

STRATEGY_HEADING = "3. Observability Strategy for v1: Three Pillars"
STRATEGY_BULLETS = [
    "Data: log every prompt-response pair with PII redaction; monitor input "
    "distributions (topic mix, language, prompt length, embedding drift) and "
    "toxicity/PII-attempt rates.",
    "Model: estimate hallucination rate automatically - verify claims "
    "against retrieved context (NLI entailment checks, citation/URL "
    "verifiability), score a sample with an LLM-as-judge, and overlay human "
    "audits; slice by topic (medical, legal, financial = high-severity).",
    "System: per-request tracing, p95 latency, token cost, and dependency "
    "health against explicit SLOs (Beyer et al., 2018).",
]

MONITORS_HEADING = "4. Monitors and Alerts for the Chatbot"
MONITORS_BULLETS = [
    "M1 - Hallucination rate: hourly scored sample of responses; "
    "alert if the estimated rate exceeds 2% or drifts three sigma from baseline.",
    "M2 - Groundedness (retrieval path): fraction of responses fully "
    "supported by retrieved context; alert on sustained drops signaling "
    "retrieval regressions.",
    "M3 - Input data quality/drift: schema violations and PSI > 0.25 on "
    "prompt embeddings signal a shifting user population.",
    "M4 - Bias and fairness: scheduled quality audits sliced by language "
    "and demographic proxies; alert on disparities (Clarify-style; Amazon "
    "Web Services, n.d.).",
    "M5 - Safety filters: spikes in jailbreak attempts, toxic outputs, or "
    "PII leakage blocks page the on-call immediately.",
    "M6 - User feedback: thumbs-down rate, correction reports, and "
    "flagged-conversation volume - the instrumented Twitter signal.",
    "M7 - System health: p95 latency, 5xx rate, and cost per 1,000 "
    "requests, against SLO error budgets (Beyer et al., 2018).",
    "Alerting: two severities (page for safety/SLO breaches, ticket for "
    "drift); every alert has a runbook; thresholds version-controlled.",
    "Governance: metrics, cadence, and owners documented per the NIST AI "
    "RMF Map/Measure functions (Tabassi, 2023).",
]

IMPROVE_HEADING = "5. From Observation to Improvement"
IMPROVE_BULLETS = [
    "Closed loop: samples flagged by M1-M2 are human-reviewed and fed into "
    "evaluation sets and the RLHF pipeline OpenAI itself relies on (Smith, "
    "2023).",
    "Weekly audits of high-severity topics; postmortems per paged alert.",
    "v1 discipline: retrain, canary a small fraction, promote after a green "
    "soak, and keep an automated rollback path.",
]

CONCLUSION_HEADING = "6. Conclusion"
CONCLUSION_BULLETS = [
    "Hallucination is an emergent LLM property: plausible, silent, and "
    "discoverable only through deliberate measurement (Ji et al., 2023).",
    "Observability converts a reputational crisis into telemetry: tiered "
    "monitors let the team measure, alert, annotate, retrain, and canary "
    "toward a trustworthy v1.",
    "Whether Sutskever or LeCun proves right about model-side fixes "
    "(Smith, 2023), production monitoring is the guarantee available today.",
]

REFERENCES_HEADING = "References"
REFERENCES = [
    "Amazon Web Services. (n.d.). Monitor models for data and model "
    "quality, bias, and explainability. In Amazon SageMaker AI Developer "
    "Guide. https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html",
    "Beyer, B., Jones, C., Petoff, J., & Murphy, N. R. (Eds.). (2018). "
    "The Site Reliability Workbook: Practical ways to implement SRE. "
    "O'Reilly Media. https://sre.google/workbook/observability/",
    "Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., "
    "Bang, Y. J., Madotto, A., & Fung, P. (2023). Survey of hallucination "
    "in natural language generation. ACM Computing Surveys, 55(12), 1-38. "
    "https://doi.org/10.1145/3571730",
    "Smith, C. S. (2023, March 13). Hallucinations could blunt ChatGPT's "
    "success. IEEE Spectrum. https://spectrum.ieee.org/ai-hallucination",
    "Tabassi, E. (2023). Artificial intelligence risk management "
    "framework (AI RMF 1.0) (NIST AI 100-1). National Institute of "
    "Standards and Technology. https://doi.org/10.6028/NIST.AI.100-1",
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

    byline = doc.add_paragraph()
    byline.add_run(BYLINE).italic = True
    byline.alignment = WD_ALIGN_PARAGRAPH.CENTER

    for heading, bullets in [
        (INTRO_HEADING, INTRO_BULLETS),
        (GAPS_HEADING, GAPS_BULLETS),
        (STRATEGY_HEADING, STRATEGY_BULLETS),
        (MONITORS_HEADING, MONITORS_BULLETS),
        (IMPROVE_HEADING, IMPROVE_BULLETS),
        (CONCLUSION_HEADING, CONCLUSION_BULLETS),
    ]:
        doc.add_heading(heading, level=2)
        add_bullets(doc, bullets)

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
