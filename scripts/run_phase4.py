import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from growth.early_warning import (
    bootstrap_lead_time_ci,
    compute_matched_threshold,
    evaluate_lead_time,
)
from growth.explain import (
    explain_single_visit_fallback,
    get_permutation_importances,
)
from growth.models import split_data_by_child


def main():
    feat_path = os.path.join(PROJECT_ROOT, "data", "asof_features.csv")
    model_path = os.path.join(PROJECT_ROOT, "models", "rf_v1.joblib")

    if not os.path.exists(feat_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            "Missing feature matrix or model artifact. Run Phase 2 & 3 scripts first."
        )

    print("Loading features and trained model artifact...")
    df = pd.read_csv(feat_path)
    artifact = joblib.load(model_path)
    rf_model = artifact["model"]
    fcols = artifact["features"]

    # Get test set
    *_, X_test, y_test, _, fcols, test_df = split_data_by_child(df)

    print("Generating model probabilities for test set...")
    test_df["rf_prob"] = rf_model.predict_proba(X_test)[:, 1]

    # 1. Matched FAR Threshold
    matched_thr = compute_matched_threshold(test_df)
    print(f"Matched False-Alarm Threshold: {matched_thr:.4f}")

    # 2. Evaluate Lead Time
    lead_df = evaluate_lead_time(test_df, threshold=matched_thr)

    median_lead, (ci_low, ci_high) = bootstrap_lead_time_ci(
        lead_df["lead_visits"]
    )
    print(
        f"Overall Median Lead Time: {median_lead:.1f} visits (95% CI: [{ci_low:.1f}, {ci_high:.1f}])"
    )

    # Breakdown per Archetype
    summary = []
    for arch, g in lead_df.groupby("archetype"):
        med, _ = bootstrap_lead_time_ci(g["lead_visits"])
        det_model = g["model_detected"].sum()
        det_base = g["base_detected"].sum()
        summary.append(
            {
                "Archetype": arch,
                "Children": len(g),
                "Model Detected": det_model,
                "Base Detected": det_base,
                "Median Lead (Visits)": med,
            }
        )

    summary_df = pd.DataFrame(summary)
    print("\n--- PHASE 4 EARLY WARNING LEAD TIME SUMMARY ---")
    print(summary_df.to_string(index=False))

    # 3. Global Permutation Importance
    print("\nCalculating Permutation Feature Importances...")
    importances = get_permutation_importances(
        rf_model, X_test, y_test, fcols
    )
    print("Top 5 Drivers of Early Anomalies:")
    print(importances.head(5).to_string())

    # 4. Sample Per-Alert Explanation
    alert_sample = X_test[test_df["rf_prob"] >= matched_thr].iloc[0]
    explanation = explain_single_visit_fallback(rf_model, alert_sample, fcols)
    print(f"\nSample Alert Explanation: {explanation}")


if __name__ == "__main__":
    main()