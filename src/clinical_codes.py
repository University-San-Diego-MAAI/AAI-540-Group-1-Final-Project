"""Canonical ICD-9-CM helpers for chronic-condition matching.

Single source of truth for the 3-digit ICD-9 prefix extraction and the
chronic-condition definition ranges used by both the EDA notebook
(`notebooks/01_eda_synpuf.ipynb`) and the feature builder
(`src/build_features.py`). Keep the definitions here in sync with the RFC.

SynPUF ICD-9 codes are dot-less (e.g. "25000" for 250.00) but may also
appear dotted; V/E-codes ("V10", "E999") carry no 3-digit numeric prefix
and must not crash parsing — they simply return NaN / no match.
"""

import re

import pandas as pd

# Member-level ANY-match ranges on the 3-digit ICD-9 prefix (inclusive).
CHRONIC_CONDITIONS: dict[str, tuple[int, int]] = {
    "diabetes": (250, 250),        # diabetes mellitus (250.x)
    "chf": (428, 428),             # congestive heart failure (428.x)
    "copd": (491, 496),            # chronic obstructive pulmonary disease (491-496)
    "stroke": (430, 438),          # cerebrovascular disease (430-438)
    "cancer": (140, 209),          # malignant neoplasms (140-209)
    "renal_disease": (580, 589),   # chronic kidney disease / nephritis (580-589)
}

_PREFIX_RE = re.compile(r"\s*(\d{3})")


def icd9_prefix(code) -> float:
    """3-digit numeric ICD-9 prefix of one code; NaN if none (V/E codes)."""
    m = _PREFIX_RE.match(str(code))
    return int(m.group(1)) if m else float("nan")


def icd9_prefixes(codes: pd.Series) -> pd.Series:
    """Vectorized variant of :func:`icd9_prefix` over a Series of codes."""
    extracted = codes.astype("string").str.extract(r"^\s*(\d{3})", expand=False)
    return pd.to_numeric(extracted, errors="coerce")
