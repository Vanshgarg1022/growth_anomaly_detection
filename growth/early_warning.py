import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

DETERIORATING_ARCHETYPES = {
    "gradual_decline",
    "rapid_decline",
    "rising_bmi",
    "abnormal_hc",
    "gradual_weight_faltering",
    "rapid_weight_loss",
    "progressive_stunting",
    "hc_acceleration",
    "hc_deceleration",
    "multi_indicator_failure",
}


def compute_matched_threshold(test_df: pd.DataFrame) -> float:
    """
    Finds the probability threshold where the model's false alarm rate
    on non-deteriorating children equals the conventional baseline's false alarm rate.
    """
    non_det = test_df[~test_df["archetype"].isin(DETERIORATING_ARCHETYPES)].copy()

    # Baseline false alarm rate (FAR)
    baseline_far = (
        non_det["baseline_status"] == "Potential Growth Anomaly"
    ).mean()

    # Quantile threshold to match baseline FAR
    matched_threshold = np.quantile(non_det["rf_prob"], 1.0 - baseline_far)
    return float(matched_threshold)


def evaluate_lead_time(test_df: pd.DataFrame, threshold: float):
    """
    Calculates first-flag visit and lead time per deteriorating child.
    """
    det_children = test_df[
        test_df["archetype"].isin(DETERIORATING_ARCHETYPES)
    ].copy()

    records = []
    for child_id, g in det_children.groupby("child_id"):
        g_sorted = g.sort_values("visit_index")
        archetype = g_sorted["archetype"].iloc[0]

        # First visit flagged by Baseline
        base_flags = g_sorted[
            g_sorted["baseline_status"] == "Potential Growth Anomaly"
        ]
        first_base = base_flags["visit_index"].min() if not base_flags.empty else np.nan

        # First visit flagged by RF Model
        model_flags = g_sorted[g_sorted["rf_prob"] >= threshold]
        first_model = (
            model_flags["visit_index"].min() if not model_flags.empty else np.nan
        )

        lead_visits = (
            first_base - first_model
            if (pd.notna(first_base) and pd.notna(first_model))
            else np.nan
        )

        records.append(
            {
                "child_id": child_id,
                "archetype": archetype,
                "first_base_visit": first_base,
                "first_model_visit": first_model,
                "lead_visits": lead_visits,
                "model_detected": pd.notna(first_model),
                "base_detected": pd.notna(first_base),
            }
        )

    return pd.DataFrame(records)


def bootstrap_lead_time_ci(
    lead_series: pd.Series, n_boot: int = 1000, seed: int = 42
):
    """Computes 95% Confidence Interval for median lead time."""
    rng = np.random.default_rng(seed)
    valid_leads = lead_series.dropna().to_numpy()
    if len(valid_leads) == 0:
        return np.nan, (np.nan, np.nan)

    boot_medians = [
        np.median(rng.choice(valid_leads, size=len(valid_leads), replace=True))
        for _ in range(n_boot)
    ]
    ci_lower, ci_upper = np.percentile(boot_medians, [2.5, 97.5])
    return np.median(valid_leads), (ci_lower, ci_upper)