import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from sqlalchemy import text
from db import SessionLocal

def seed_demo_patients():
    """
    Seeds specific test children (D- prefix) from synthetic test set into MySQL
    to guarantee all 13 demonstration steps can be shown live.
    """
    visits = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "synthetic_visits.csv"))
    children = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "synthetic_children.csv"))

    # Select representative children for demo scenarios
    demo_targets = {
        "D-0001": ("gradual_weight_faltering", "Deteriorating case (Model flags before Baseline)"),
        "D-0002": ("recovery_catchup", "Recovery case (Not treated as worsening)"),
        "D-0003": ("normal_typical", "Stable normal child"),
        "D-0004": ("persistent_underweight", "Persistent low baseline case"),
    }

    with SessionLocal.begin() as session:
        # Clear existing demo rows
        session.execute(text("DELETE FROM visits WHERE patient_id IN (SELECT patient_id FROM patients WHERE patient_code LIKE 'D-%')"))
        session.execute(text("DELETE FROM patients WHERE patient_code LIKE 'D-%'"))

        print("Seeding demo patients into database...")

        for demo_code, (arch, desc) in demo_targets.items():
            match = children[children["archetype"] == arch]
            if match.empty:
                continue
            orig_code = match.iloc[0]["patient_code"]

            child_visits = visits[visits["patient_code"] == orig_code].sort_values("age_months")
            if child_visits.empty:
                continue

            # Assign synthetic DOB so latest visit is today
            dob = pd.Timestamp.now().date() - pd.Timedelta(days=int(child_visits["age_days"].max()))

            sex = "Boy" if "boy" in orig_code.lower() else "Girl"
            res = session.execute(
                text("INSERT INTO patients (patient_code, date_of_birth, sex) VALUES (:c, :d, :s)"),
                {"c": demo_code, "d": dob, "s": sex}
            )
            pid = res.lastrowid

            for _, v in child_visits.iterrows():
                v_date = dob + pd.Timedelta(days=int(v["age_days"]))
                session.execute(
                    text("""
                        INSERT INTO visits 
                        (patient_id, visit_date, weight_kg, height_cm, head_circumference_cm, 
                         waz, haz, bmiz, hcz, measurement_type, adjusted_length_height_cm, bmi)
                        VALUES (:pid, :vd, :w, :h, :hc, :waz, :haz, :bmiz, :hcz, :mt, :adj, :bmi)
                    """),
                    {
                        "pid": pid,
                        "vd": v_date,
                        "w": v["weight_kg"],
                        "h": v["height_cm"],
                        "hc": v["head_circumference_cm"],
                        "waz": v["waz"],
                        "haz": v["haz"],
                        "bmiz": v["bmiz"],
                        "hcz": v["hcz"],
                        "mt": v.get("measurement_type", "recumbent"),
                        "adj": v["height_cm"],
                        "bmi": v.get("bmi", 15.0)
                    }
                )

        # Seed a 1-visit patient for demo scenario 12
        session.execute(
            text("INSERT INTO patients (patient_code, date_of_birth, sex) VALUES ('D-0005', :d, 'Boy')"),
            {"d": pd.Timestamp.now().date() - pd.Timedelta(days=60)}
        )
        pid5 = session.execute(text("SELECT patient_id FROM patients WHERE patient_code='D-0005'")).scalar()
        session.execute(
            text("""
                INSERT INTO visits (patient_id, visit_date, weight_kg, height_cm, head_circumference_cm, waz, haz, bmiz, hcz, measurement_type, adjusted_length_height_cm, bmi)
                VALUES (:pid, :vd, 4.5, 58.0, 39.0, 0.1, 0.0, 0.1, 0.0, 'recumbent', 58.0, 13.38)
            """),
            {"pid": pid5, "vd": pd.Timestamp.now().date()}
        )

        print("Seeding complete! Demo patients created: D-0001, D-0002, D-0003, D-0004, D-0005.")

if __name__ == "__main__":
    seed_demo_patients()