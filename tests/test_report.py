import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pytest
import pandas as pd
from growth.report import generate_patient_pdf_report

def test_pdf_generation_sufficient_history():
    df_mock = pd.DataFrame({
        "visit_date": pd.date_range("2025-01-01", periods=4, freq="90D"),
        "age_months": [2.0, 5.0, 8.0, 11.0],
        "waz": [0.1, -0.2, -0.5, -0.8],
        "haz": [0.0, -0.1, -0.3, -0.4],
        "bmiz": [0.1, -0.1, -0.4, -0.7],
        "hcz": [0.0, 0.0, -0.1, -0.1]
    })
    
    res = {
        "patient_code": "S-TEST",
        "n_visits": 4,
        "risk_status": "Watch",
        "fired_rules": ["R6"],
        "rf_prob": 0.2500,
        "explanation": "waz_slope_per_month (-0.10); bmiz_overall_change (-0.80)",
        "df_visits": df_mock
    }
    
    pdf_bytes = generate_patient_pdf_report(res)
    assert len(pdf_bytes) > 0, "PDF bytes should not be empty"
    assert pdf_bytes.startswith(b"%PDF"), "Output is not a valid PDF file"

def test_pdf_generation_insufficient_history():
    res = {
        "patient_code": "S-SHORT",
        "n_visits": 1,
        "risk_status": "Insufficient history"
    }
    
    pdf_bytes = generate_patient_pdf_report(res)
    assert len(pdf_bytes) > 0
    assert pdf_bytes.startswith(b"%PDF")