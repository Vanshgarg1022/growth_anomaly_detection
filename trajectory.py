"""
Longitudinal trajectory features (plan Week 5) computed from one child's visits.

Only visits with dq_status == 'valid' are used. Visits are ordered by age, never treated as
independent samples. No clinical interpretation is attached here - these are numerical descriptors
of how the z-scores change across visits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAYS_PER_MONTH = 30.4375
MIN_VISITS = 3          # fewer valid visits -> "insufficient longitudinal history"
RECENT_WINDOW = 3       # "recent change" = change across the last 3 visits

INDICATORS = {
    "waz": "Weight-for-Age",
    "haz": "Length/Height-for-Age",
    "bmiz": "BMI-for-Age",
    "hcz": "Head-Circumference-for-Age",
}

FEATURE_LABELS = {
    "n_visits": "Valid visits used",
    "current_z": "Current z-score",
    "previous_z": "Previous z-score",
    "delta_z": "Change since previous visit",
    "overall_change": "Overall change (first to latest)",
    "recent_change": f"Change over last {RECENT_WINDOW} visits",
    "mean_z": "Mean z-score",
    "min_z": "Minimum z-score",
    "max_z": "Maximum z-score",
    "sd_z": "Variability (SD of z-scores)",
    "slope_per_month": "Slope (z-score per month)",
    "largest_decline": "Largest single-visit decline",
    "consecutive_declines": "Consecutive declines (ending now)",
    "consecutive_increases": "Consecutive increases (ending now)",
}


def prepare_timeline(visits: pd.DataFrame) -> pd.DataFrame:
    """Sort visits chronologically, coerce numerics, add age_months and a boolean `usable`."""
    df = visits.copy()
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    for c in ("age_days", *INDICATORS):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if "dq_status" not in df:
        df["dq_status"] = "valid"
    df["dq_status"] = df["dq_status"].fillna("valid")
    df = df.sort_values("visit_date").reset_index(drop=True)
    df["age_months"] = df["age_days"] / DAYS_PER_MONTH
    df["usable"] = df["dq_status"].eq("valid")
    return df


def indicator_features(age_months: np.ndarray, z: np.ndarray) -> dict:
    """Features for one indicator. `z` is ordered by age; NaNs must already be removed."""
    n = len(z)
    f = {k: np.nan for k in FEATURE_LABELS}
    f["n_visits"] = n
    if n == 0:
        return f
    f["current_z"] = z[-1]
    f["mean_z"], f["min_z"], f["max_z"] = z.mean(), z.min(), z.max()
    if n >= 2:
        d = np.diff(z)
        f["previous_z"] = z[-2]
        f["delta_z"] = d[-1]
        f["overall_change"] = z[-1] - z[0]
        f["recent_change"] = z[-1] - z[-min(n, RECENT_WINDOW)]
        f["largest_decline"] = min(d.min(), 0.0)
        f["consecutive_declines"] = _trailing_run(d < 0)
        f["consecutive_increases"] = _trailing_run(d > 0)
    if n >= MIN_VISITS:
        f["sd_z"] = z.std(ddof=1)
        if np.ptp(age_months) > 0:
            f["slope_per_month"] = np.polyfit(age_months, z, 1)[0]
    return f


def _trailing_run(mask: np.ndarray) -> int:
    run = 0
    for flag in mask[::-1]:
        if not flag:
            break
        run += 1
    return run


def patient_features(timeline: pd.DataFrame):
    """Return (features DataFrame [rows = features, columns = indicators], info dict)."""
    usable = timeline[timeline["usable"]]
    cols = {}
    for key, label in INDICATORS.items():
        d = usable.dropna(subset=[key, "age_months"]).sort_values("age_months")
        cols[label] = indicator_features(d["age_months"].to_numpy(float), d[key].to_numpy(float))
    feats = pd.DataFrame(cols).rename(index=FEATURE_LABELS)
    n_valid = int(usable.shape[0])
    info = {
        "n_total": int(timeline.shape[0]),
        "n_valid": n_valid,
        "n_excluded": int(timeline.shape[0]) - n_valid,
        "sufficient": n_valid >= MIN_VISITS,
    }
    return feats, info
