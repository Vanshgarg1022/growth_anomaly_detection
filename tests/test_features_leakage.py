import pytest
import numpy as np
import pandas as pd
from growth.features import compute_asof_features

def test_features_do_not_look_ahead():
    mock_child = pd.DataFrame({
        "child": ["S-0001"] * 8,
        "age_months": [1.0, 3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 24.0],
        "waz": [0.1, 0.0, -0.2, -0.5, -0.9, -1.2, -1.5, -1.8],
        "haz": [0.2, 0.1, 0.0, -0.1, -0.3, -0.4, -0.6, -0.8],
        "bmiz": [0.0, -0.1, -0.3, -0.6, -1.0, -1.3, -1.6, -1.9],
        "hcz": [0.1, 0.1, 0.0, 0.0, -0.1, -0.1, -0.2, -0.2]
    })
    
    early = compute_asof_features(mock_child.iloc[:5]).query("visit_index == 4").iloc[0]
    late = compute_asof_features(mock_child).query("visit_index == 4").iloc[0]
    
    feature_cols = [c for c in early.index if c not in ("child_id", "visit_index", "age_months", "visit_id")]
    
    np.testing.assert_allclose(
        early[feature_cols].astype(float).values,
        late[feature_cols].astype(float).values,
        err_msg="Features at visit index 4 modified when future visits were added!"
    )