import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from growth.models import (
    split_data_by_child,
    train_isolation_forest,
    train_logistic_regression,
    train_random_forest,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold


def evaluate_model(name, y_true, y_pred, y_prob):
    return {
        "Model": name,
        "ROC-AUC": round(roc_auc_score(y_true, y_prob), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
    }


def main():
    feat_path = os.path.join(PROJECT_ROOT, "data", "asof_features.csv")
    if not os.path.exists(feat_path):
        raise FileNotFoundError(
            f"Could not find {feat_path}. Run scripts/run_phase2.py first."
        )

    print("Loading feature matrix...")
    df = pd.read_csv(feat_path)

    print("Splitting data by child (80% Train, 20% Test)...")
    (
        X_train,
        y_train,
        groups_train,
        X_test,
        y_test,
        groups_test,
        fcols,
        test_df,
    ) = split_data_by_child(df)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    print("Training Logistic Regression...")
    lr_grid = train_logistic_regression(X_train, y_train, groups_train, cv)

    print("Training Random Forest Classifier...")
    rf_grid = train_random_forest(X_train, y_train, groups_train, cv)

    print("Training Isolation Forest Anomaly Detector...")
    iso_model = train_isolation_forest(X_train)

    # Evaluate on held-out test set
    results = []

    # 1. Logistic Regression
    lr_prob = lr_grid.predict_proba(X_test)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)
    results.append(evaluate_model("Logistic Regression", y_test, lr_pred, lr_prob))

    # 2. Random Forest
    rf_prob = rf_grid.predict_proba(X_test)[:, 1]
    rf_pred = (rf_prob >= 0.5).astype(int)
    results.append(evaluate_model("Random Forest", y_test, rf_pred, rf_prob))

    # 3. Isolation Forest (Negated anomaly score as probability proxy)
    iso_scores = -iso_model.score_samples(X_test)
    iso_prob = (iso_scores - iso_scores.min()) / (
        iso_scores.max() - iso_scores.min()
    )
    iso_pred = (iso_prob >= 0.5).astype(int)
    results.append(
        evaluate_model("Isolation Forest (Anomaly)", y_test, iso_pred, iso_prob)
    )

    # 4. Conventional Baseline (R1-R4 category flag)
    base_pred = (test_df["baseline_status"] == "Potential Growth Anomaly").astype(
        int
    )
    results.append(
        evaluate_model("Conventional Rule Baseline", y_test, base_pred, base_pred)
    )

    summary_df = pd.DataFrame(results)
    print("\n--- PHASE 3 MODEL EVALUATION SUMMARY ---")
    print(summary_df.to_string(index=False))

    # Save Best Model Artifact
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_artifact = {
        "model": rf_grid.best_estimator_,
        "features": fcols,
        "version": "rf_v1",
        "trained_on": f"Synthetic seed 42, {len(df['child_id'].unique())} children",
        "best_params": rf_grid.best_params_,
    }

    artifact_path = os.path.join(models_dir, "rf_v1.joblib")
    joblib.dump(model_artifact, artifact_path)
    print(f"\nSaved trained model artifact to {artifact_path}")


if __name__ == "__main__":
    main()