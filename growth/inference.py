import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import text
from growth.features import compute_asof_features
from growth.baseline import evaluate_baseline_rules
from growth.explain import explain_single_visit_fallback

# Matched threshold from Phase 4
MATCHED_THRESHOLD = 0.1963

def load_cached_model():
    model_path = os.path.join(PROJECT_ROOT, "models", "rf_v1.joblib")
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

def analyze_patient_trajectory(patient_code: str, session) -> dict:
    """
    Fetches visits for patient_code, computes features as of latest visit,
    runs baseline + RF model, and returns structured results for Streamlit.
    """
    sql = text("""
        SELECT v.visit_id, v.visit_date, v.weight_kg, v.height_cm, v.head_circumference_cm,
               v.waz, v.haz, v.bmiz, v.hcz, p.patient_id, p.date_of_birth, p.sex
        FROM visits v 
        JOIN patients p ON p.patient_id = v.patient_id
        WHERE p.patient_code = :c 
        ORDER BY v.visit_date ASC
    """)
    rows = session.execute(sql, {"c": patient_code}).mappings().all()
    if not rows:
        return {"status": "No visits found", "n_visits": 0}

    df = pd.DataFrame(rows)
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    df["date_of_birth"] = pd.to_datetime(df["date_of_birth"])
    df["age_months"] = ((df["visit_date"] - df["date_of_birth"]).dt.days / 30.4375).round(2)
    
    # Cast Decimal columns to float
    for col in ("waz", "haz", "bmiz", "hcz"):
        df[col] = df[col].astype(float)

    n_visits = len(df)
    latest_visit = df.iloc[-1]
    
    # Base response
    result = {
        "patient_id": int(latest_visit["patient_id"]),
        "visit_id": int(latest_visit["visit_id"]),
        "patient_code": patient_code,
        "n_visits": n_visits,
        "df_visits": df,
        "latest_z": {
            "waz": latest_visit["waz"],
            "haz": latest_visit["haz"],
            "bmiz": latest_visit["bmiz"],
            "hcz": latest_visit["hcz"]
        }
    }

    # Insufficient longitudinal history check
    if n_visits < 3:
        result["risk_status"] = "Insufficient history"
        result["baseline_status"] = "Insufficient history"
        result["rf_prob"] = None
        result["explanation"] = "At least 3 visits are required to compute trajectory features."
        return result

    # 1. Evaluate Conventional Baseline
    base_status, fired_rules = evaluate_baseline_rules(result["latest_z"])
    result["baseline_status"] = base_status
    result["fired_rules"] = fired_rules

    # 2. Evaluate ML Model
    artifact = load_cached_model()
    if artifact:
        rf_model = artifact["model"]
        fcols = artifact["features"]
        
        # Build time-aware feature row for latest visit (k = n_visits - 1)
        df["patient_code"] = patient_code
        feats_df = compute_asof_features(df)
        
        if not feats_df.empty:
            latest_feat = feats_df.iloc[[-1]][fcols]
            rf_prob = float(rf_model.predict_proba(latest_feat)[:, 1][0])
            result["rf_prob"] = rf_prob
            
            # Map probability to risk category using matched threshold
            if rf_prob >= MATCHED_THRESHOLD:
                result["risk_status"] = "Potential Growth Anomaly"
            elif rf_prob >= (MATCHED_THRESHOLD * 0.6):
                result["risk_status"] = "Watch"
            else:
                result["risk_status"] = "Normal"

            # Explanation
            result["explanation"] = explain_single_visit_fallback(rf_model, latest_feat, fcols)
            result["model_version"] = artifact.get("version", "rf_v1")

    return result

def save_analysis_result(result: dict, session):
    """Persists analysis outcome into analysis_results table."""
    if result.get("n_visits", 0) < 3 or "rf_prob" not in result:
        return

    sql = text("""
        INSERT INTO analysis_results 
        (patient_id, visit_id, method, model_version, risk_status, score, n_valid_visits, explanation)
        VALUES (:pid, :vid, 'random_forest', :ver, :status, :score, :n_vis, :exp)
        ON DUPLICATE KEY UPDATE 
            risk_status = VALUES(risk_status),
            score = VALUES(score),
            explanation = VALUES(explanation)
    """)
    
    session.execute(sql, {
        "pid": result["patient_id"],
        "vid": result["visit_id"],
        "ver": result.get("model_version", "rf_v1"),
        "status": result["risk_status"],
        "score": result["rf_prob"],
        "n_vis": result["n_visits"],
        "exp": json.dumps({"reason": result.get("explanation", "")})
    })
    session.commit()