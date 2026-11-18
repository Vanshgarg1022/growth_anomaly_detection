import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import trajectory as T

INDICATORS = ("waz", "haz", "bmiz", "hcz")

# Archetypes that represent actual emerging deterioration
DETERIORATING_ARCHETYPES = {
    "gradual_decline",
    "rapid_decline",
    "rising_bmi",
    "abnormal_hc",
}


def compute_asof_features(df_child_visits: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # k >= 2 ensures at least 3 visits for slope and variability calculation
    for k in range(2, len(df_child_visits)):
        h = df_child_visits.iloc[: k + 1]

        row = {
            "child_id": (
                h["patient_code"].iloc[0]
                if "patient_code" in h.columns
                else h["child"].iloc[0]
            ),
            "visit_index": k,
            "age_months": h["age_months"].iloc[-1],
            "visit_id": (
                h["visit_id"].iloc[-1] if "visit_id" in h.columns else None
            ),
        }

        for ind in INDICATORS:
            feats = T.indicator_features(
                h["age_months"].to_numpy(), h[ind].to_numpy()
            )
            for name, val in feats.items():
                if name != "n_visits":
                    row[f"{ind}_{name}"] = val

        rows.append(row)

    return pd.DataFrame(rows)


def build_feature_matrix(
    df_all_visits: pd.DataFrame, df_truth: pd.DataFrame = None
) -> pd.DataFrame:
    all_features = []
    id_col = (
        "patient_code" if "patient_code" in df_all_visits.columns else "child"
    )

    for _, g in df_all_visits.groupby(id_col):
        g_sorted = g.sort_values("age_months")
        feats_k = compute_asof_features(g_sorted)
        if not feats_k.empty:
            all_features.append(feats_k)

    if not all_features:
        return pd.DataFrame()

    feat_df = pd.concat(all_features, ignore_index=True)

    if df_truth is not None and "patient_code" in df_truth.columns:
        # Merge patient-level truth (archetype, onset_visit_index)
        truth_cols = [
            c
            for c in ["patient_code", "archetype", "onset_visit_index"]
            if c in df_truth.columns
        ]
        feat_df = feat_df.merge(
            df_truth[truth_cols],
            left_on="child_id",
            right_on="patient_code",
            how="left",
        )

        # Assign per-visit label: 1 if deteriorating archetype AND at or after onset_visit_index
        if "archetype" in feat_df.columns and "onset_visit_index" in feat_df.columns:

            def derive_label(r):
                if pd.isna(r["onset_visit_index"]):
                    return 0
                is_det = r["archetype"] in DETERIORATING_ARCHETYPES
                is_after_onset = r["visit_index"] >= r["onset_visit_index"]
                return 1 if (is_det and is_after_onset) else 0

            feat_df["label"] = feat_df.apply(derive_label, axis=1)

    return feat_df