import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedGroupKFold,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def get_feature_columns(df: pd.DataFrame) -> list:
    """Extracts the 52 trajectory feature column names from the feature matrix."""
    return [
        c
        for c in df.columns
        if any(c.startswith(f"{ind}_") for ind in ("waz", "haz", "bmiz", "hcz"))
    ]


def split_data_by_child(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    """
    Performs a child-disjoint split.
    Guarantees no child has visits in both train and test sets.
    """
    fcols = get_feature_columns(df)

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, df["label"], groups=df["child_id"]))

    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()

    X_train, y_train, groups_train = (
        train_df[fcols],
        train_df["label"],
        train_df["child_id"],
    )
    X_test, y_test, groups_test = (
        test_df[fcols],
        test_df["label"],
        test_df["child_id"],
    )

    return (
        X_train,
        y_train,
        groups_train,
        X_test,
        y_test,
        groups_test,
        fcols,
        test_df,
    )


def train_logistic_regression(X_train, y_train, groups_train, cv):
    """Trains Logistic Regression with standardized scaling and hyperparameter tuning."""
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42
        ),
    )
    param_grid = {"logisticregression__C": [0.01, 0.1, 1.0, 10.0]}
    grid = GridSearchCV(
        pipe, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True
    )
    grid.fit(X_train, y_train, groups=groups_train)
    return grid


def train_random_forest(X_train, y_train, groups_train, cv):
    """Trains Random Forest Classifier with class balancing and leaf tuning."""
    rf = RandomForestClassifier(
        class_weight="balanced_subsample", random_state=42, n_jobs=-1
    )
    param_grid = {
        "n_estimators": [200, 300],
        "min_samples_leaf": [3, 5, 10],
        "max_depth": [8, 12, None],
    }
    grid = GridSearchCV(
        rf, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True
    )
    grid.fit(X_train, y_train, groups=groups_train)
    return grid


def train_isolation_forest(X_train):
    """Fits an unsupervised Isolation Forest model on training features."""
    iso = IsolationForest(
        n_estimators=300, random_state=42, contamination="auto", n_jobs=-1
    )
    iso.fit(X_train)
    return iso