import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import pandas as pd
import pytest
from growth.models import split_data_by_child


def test_train_test_children_are_disjoint():
    feat_path = os.path.join(PROJECT_ROOT, "data", "asof_features.csv")
    if not os.path.exists(feat_path):
        pytest.skip("data/asof_features.csv not generated yet.")

    df = pd.read_csv(feat_path)
    X_train, y_train, g_train, X_test, y_test, g_test, fcols, _ = (
        split_data_by_child(df)
    )

    train_children = set(g_train)
    test_children = set(g_test)

    assert train_children.isdisjoint(
        test_children
    ), "Data leakage detected: Children found in both train and test sets!"


def test_model_artifact_integrity():
    artifact_path = os.path.join(PROJECT_ROOT, "models", "rf_v1.joblib")
    if not os.path.exists(artifact_path):
        pytest.skip("models/rf_v1.joblib not saved yet.")

    data = joblib.load(artifact_path)
    assert "model" in data, "Artifact missing 'model' key"
    assert "features" in data, "Artifact missing 'features' key"
    assert (
        len(data["features"]) == 52
    ), f"Expected 52 features, got {len(data['features'])}"