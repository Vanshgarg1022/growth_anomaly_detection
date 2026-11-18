"""
WHO Child Growth Standards (0-5 years) z-score engine.

Reads the WHO "z-score" Excel reference tables (columns: Month, L, M, S, SD3neg..SD3)
with pandas, picks the correct table from the child's sex, age and measurement type,
and computes the z-score with the WHO LMS method.

Indicators
----------
wfa   Weight-for-Age
lhfa  Length/Height-for-Age   (0-2 y = recumbent LENGTH table, 2-5 y = standing HEIGHT table)
bfa   BMI-for-Age             (0-2 y table based on length, 2-5 y on height)
hcfa  Head-Circumference-for-Age

Method follows the WHO Anthro / igrowup macros:
  * age in days; 1 month = 30.4375 days; the 0-2 y (length) tables are used for age < 731 days
  * L, M, S are cubic-spline interpolated between the monthly rows (the WHO macros use daily rows;
    a spline reproduces them to within ~0.03 z, a straight line is off by up to 0.07 z at 1-3 months)
  * z = ((y/M)^L - 1) / (L*S)            (ln(y/M)/S when L == 0)
  * WFA and BMI-for-age: for |z| > 3 the z-score is extended linearly using the SD2/SD3 cut-offs
  * measurement type: a standing height measured under 2 y gets +0.7 cm, a recumbent length
    measured at 2 y or over gets -0.7 cm, so that it matches the table that is used
  * implausible values (WHO "flag" ranges) are reported as flagged """

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import pandas as pd
from scipy.interpolate import CubicSpline

TABLE_DIR = Path(__file__).resolve().parent / "who_tables"

DAYS_PER_MONTH = 30.4375
LENGTH_TO_HEIGHT_DAYS = 731          # WHO macros: age >= 731 days -> standing-height table
LENGTH_HEIGHT_OFFSET_CM = 0.7        # standing height is 0.7 cm shorter than recumbent length
MAX_AGE_DAYS = 1856                  # last day in the WHO daily tables (60.98 months)
MAX_AGE_MONTHS = 60

RECUMBENT = "Recumbent Length"
STANDING = "Standing Height"

# WHO download file names per indicator; {sex} is "boys" or "girls".
TABLE_FILES = {
    "wfa":  {"all": "wfa_{sex}_0-to-5-years_zscores.xlsx"},
    "lhfa": {"0-2": "lhfa_{sex}_0-to-2-years_zscores.xlsx",
             "2-5": "lhfa_{sex}_2-to-5-years_zscores.xlsx"},
    "bfa":  {"0-2": "bmi_{sex}_0-to-2-years_zscores.xlsx",   # boys file is spelled "..._zcores.xlsx"
             "2-5": "bmi_{sex}_2-to-5-years_zscores.xlsx"},
    "hcfa": {"all": "hcfa-{sex}-0-5-zscores.xlsx"},
}
# the downloaded files do not follow one naming pattern; exceptions are listed here
FILE_EXCEPTIONS = {("bfa", "boys", "0-2"): "bmi_boys_0-to-2-years_zcores.xlsx"}


def table_file_name(indicator: str, sex: str, part: str) -> str:
    return FILE_EXCEPTIONS.get((indicator, sex, part), TABLE_FILES[indicator][part].format(sex=sex))

INDICATOR_LABELS = {
    "wfa": "Weight-for-Age",
    "lhfa": "Length/Height-for-Age",
    "bfa": "BMI-for-Age",
    "hcfa": "Head-Circumference-for-Age",
}

# WHO biologically-implausible ("flag") limits
FLAG_LIMITS = {"wfa": (-6, 5), "lhfa": (-6, 6), "bfa": (-5, 5), "hcfa": (-5, 5)}
# indicators where |z| > 3 is extended using the SD2/SD3 cut-offs
EXTEND_BEYOND_3SD = {"wfa", "bfa"}

REQUIRED_COLUMNS = {"Month", "L", "M", "S"}


# --------------------------------------------------------------------------- tables
def normalise_sex(sex: str) -> str:
    s = str(sex).strip().lower()
    if s in {"boy", "boys", "male", "m", "1"}:
        return "boys"
    if s in {"girl", "girls", "female", "f", "2"}:
        return "girls"
    raise ValueError(f"Unknown gender: {sex!r} (expected 'Boy' or 'Girl')")


@lru_cache(maxsize=None)
def load_table(indicator: str, sex: str, part: str) -> pd.DataFrame:
    """Read one WHO Excel table with pandas (cached)."""
    sex = normalise_sex(sex)
    path = TABLE_DIR / table_file_name(indicator, sex, part)
    if not path.exists():
        raise FileNotFoundError(f"WHO reference file not found: {path}")
    df = pd.read_excel(path)
    df.columns = [str(c).strip() for c in df.columns]   # some WHO files have 'M       ' etc.
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns {sorted(missing)}")
    df = df[["Month", "L", "M", "S"]].astype(float).sort_values("Month").reset_index(drop=True)
    return df


def age_in_months(age_days: float) -> float:
    return age_days / DAYS_PER_MONTH


def select_table(indicator: str, sex: str, age_days: int):
    """Choose the WHO table from sex + age. Returns ((indicator, sex, part), file_name)."""
    sex = normalise_sex(sex)
    if age_days < 0:
        raise ValueError("Age cannot be negative.")
    if age_days > MAX_AGE_DAYS:
        raise ValueError("Age is outside the WHO 0-5 year reference range.")
    parts = TABLE_FILES[indicator]
    part = "all" if "all" in parts else ("0-2" if age_days < LENGTH_TO_HEIGHT_DAYS else "2-5")
    return (indicator, sex, part), table_file_name(indicator, sex, part)


@lru_cache(maxsize=None)
def _splines(indicator: str, sex: str, part: str):
    """Cubic splines for L, M and S of one table, plus its first/last month."""
    t = load_table(indicator, sex, part)
    splines = tuple(CubicSpline(t["Month"].values, t[c].values) for c in ("L", "M", "S"))
    return splines, (t["Month"].iloc[0], t["Month"].iloc[-1])


def lms_at(key: tuple, months: float):
    """L, M, S at a fractional age in months (cubic-spline interpolation of the monthly rows).
    `key` is the (indicator, sex, part) tuple returned by select_table."""
    (sL, sM, sS), (lo, hi) = _splines(*key)
    months = min(max(months, lo), hi)
    return float(sL(months)), float(sM(months)), float(sS(months))


# --------------------------------------------------------------------------- maths
def _cutoff(L: float, M: float, S: float, z: int) -> float:
    """Measurement value at a given SD from the LMS parameters."""
    if L == 0:
        return M * math.exp(S * z)
    return M * (1 + L * S * z) ** (1 / L)


def zscore_from_lms(y: float, L: float, M: float, S: float, extend: bool = False) -> float:
    if y <= 0:
        raise ValueError("Measurement must be greater than zero.")
    z = math.log(y / M) / S if L == 0 else ((y / M) ** L - 1) / (L * S)
    if extend and z > 3:
        sd3, sd2 = _cutoff(L, M, S, 3), _cutoff(L, M, S, 2)
        z = 3 + (y - sd3) / (sd3 - sd2)
    elif extend and z < -3:
        sd3n, sd2n = _cutoff(L, M, S, -3), _cutoff(L, M, S, -2)
        z = -3 + (y - sd3n) / (sd2n - sd3n)
    return z


# --------------------------------------------------------------------------- mapping
def classify(indicator: str, z: float) -> str:
    """Map a z-score to the WHO descriptive category."""
    if indicator == "wfa":
        if z < -3: return "Severely underweight"
        if z < -2: return "Underweight"
        if z <= 2: return "Normal"
        return "Above normal range (> +2 SD)"
    if indicator == "lhfa":
        if z < -3: return "Severely stunted"
        if z < -2: return "Stunted"
        if z <= 3: return "Normal"
        return "Very tall (> +3 SD)"
    if indicator == "bfa":
        if z < -3: return "Severely wasted (severe thinness)"
        if z < -2: return "Wasted (thinness)"
        if z <= 1: return "Normal"
        if z <= 2: return "Possible risk of overweight"
        if z <= 3: return "Overweight"
        return "Obese"
    if indicator == "hcfa":
        if z < -3: return "Severe microcephaly"
        if z < -2: return "Microcephaly"
        if z <= 2: return "Normal"
        if z <= 3: return "Macrocephaly"
        return "Severe macrocephaly"
    raise KeyError(indicator)


# --------------------------------------------------------------------------- indicators
def compute_indicator(indicator: str, sex: str, age_days: int, value: float) -> dict:
    """z-score for one indicator. `value` is the measurement matching the table
    (kg, cm of length/height already adjusted, kg/m2, or cm)."""
    key, table_name = select_table(indicator, sex, age_days)
    L, M, S = lms_at(key, age_in_months(age_days))
    z = zscore_from_lms(value, L, M, S, extend=indicator in EXTEND_BEYOND_3SD)
    lo, hi = FLAG_LIMITS[indicator]
    flagged = not (lo <= z <= hi)
    return {
        "indicator": indicator,
        "label": INDICATOR_LABELS[indicator],
        "value": value,
        "z": z,
        "flag": flagged,
        "category": classify(indicator, z),
        "table": table_name,
        "L": L, "M": M, "S": S,
    }


def adjust_length_height(height_cm: float, measurement_type: str, age_days: int):
    """Match the measurement to the table used (<731 d = length, >=731 d = height).
    Returns (adjusted_cm, note_or_None)."""
    if measurement_type not in (RECUMBENT, STANDING):
        raise ValueError(f"Unknown measurement type: {measurement_type!r}")
    if age_days < LENGTH_TO_HEIGHT_DAYS and measurement_type == STANDING:
        return height_cm + LENGTH_HEIGHT_OFFSET_CM, (
            f"Standing height measured under 2 years: +{LENGTH_HEIGHT_OFFSET_CM} cm added "
            "to convert it to length (WHO convention).")
    if age_days >= LENGTH_TO_HEIGHT_DAYS and measurement_type == RECUMBENT:
        return height_cm - LENGTH_HEIGHT_OFFSET_CM, (
            f"Recumbent length measured at 2 years or older: {LENGTH_HEIGHT_OFFSET_CM} cm "
            "subtracted to convert it to standing height (WHO convention).")
    return height_cm, None


def analyse_child(sex: str, age_days: int, weight_kg: float, length_height_cm: float,
                  measurement_type: str, head_circ_cm: float | None = None) -> dict:
    """Run all four indicators for one child."""
    adj_cm, note = adjust_length_height(length_height_cm, measurement_type, age_days)
    bmi = weight_kg / (adj_cm / 100) ** 2      # WHO macros compute BMI from the adjusted length/height
    results = {
        "wfa": compute_indicator("wfa", sex, age_days, weight_kg),
        "lhfa": compute_indicator("lhfa", sex, age_days, adj_cm),
        "bfa": compute_indicator("bfa", sex, age_days, bmi),
    }
    if head_circ_cm:
        results["hcfa"] = compute_indicator("hcfa", sex, age_days, head_circ_cm)
    return {"results": results, "bmi": bmi, "adjusted_length_height": adj_cm, "adjustment_note": note,
            "length_height_table": results["lhfa"]["table"]}
