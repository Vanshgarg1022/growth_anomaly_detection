import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def get_permutation_importances(model, X_test, y_test, feature_names):
    """Calculates permutation feature importances on held-out test data."""
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    importances = pd.Series(result.importances_mean, index=feature_names)
    return importances.sort_values(ascending=False)


def explain_single_visit_fallback(model, X_row, feature_names, top_n=3):
    """
    Generates top-N feature contributions for a single alert visit.
    Fallback module if SHAP is not installed.
    """
    if isinstance(X_row, pd.DataFrame):
        X_vec = X_row.iloc[0].to_numpy()
    else:
        X_vec = np.array(X_row)

    # Use tree feature importances weighted by normalized feature values
    tree_importances = model.feature_importances_
    contributions = tree_importances * X_vec

    top_indices = np.argsort(np.abs(contributions))[::-1][:top_n]

    top_reasons = []
    for idx in top_indices:
        fname = feature_names[idx]
        val = X_vec[idx]
        top_reasons.append(f"{fname} ({val:+.2f})")

    return "; ".join(top_reasons)