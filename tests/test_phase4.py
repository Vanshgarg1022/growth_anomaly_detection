import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import pytest
from growth.early_warning import bootstrap_lead_time_ci


def test_bootstrap_ci_validity():
    leads = pd.Series([1.0, 2.0, 1.0, 3.0, 0.0, 2.0, 1.0, np.nan])
    median, (ci_low, ci_high) = bootstrap_lead_time_ci(leads, n_boot=200)

    assert not np.isnan(median), "Median should not be NaN"
    assert ci_low <= median <= ci_high, "CI bounds invalid"