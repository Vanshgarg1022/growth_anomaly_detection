"""
Load the synthetic children (codes S-xxxx) into the project database through the app's own db.py.

    python scripts/load_synthetic_to_db.py            # replace all S- rows with the CSV contents
    python scripts/load_synthetic_to_db.py --remove   # only delete the S- rows

Safe to re-run: existing S- rows are deleted first, in the same transaction, so a failure leaves the
database unchanged. Real patients (codes A-####) are never touched. Needs db.py (SessionLocal) on the path,
e.g. run from your app folder or set PYTHONPATH.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import text

ROOT = Path(__file__).resolve().parent.parent
PREFIX = "S-"

VISIT_COLS = ["patient_id", "visit_date", "weight_kg", "height_cm", "head_circumference_cm",
              "waz", "haz", "bmiz", "hcz", "measurement_type", "adjusted_length_height_cm", "bmi"]


def remove_synthetic(session) -> None:
    session.execute(text("DELETE FROM visits WHERE patient_id IN "
                         "(SELECT patient_id FROM patients WHERE patient_code LIKE 'S-%')"))
    session.execute(text("DELETE FROM patients WHERE patient_code LIKE 'S-%'"))


def load(session, visits: pd.DataFrame, children: pd.DataFrame) -> tuple[int, int]:
    """Replace S- rows with the given data in the current transaction (caller commits)."""
    remove_synthetic(session)
    for r in children.itertuples():
        session.execute(text("INSERT INTO patients (patient_code, date_of_birth, sex) VALUES (:c, :d, :s)"),
                        {"c": r.patient_code, "d": str(r.date_of_birth)[:10], "s": r.sex})
    ids = dict(session.execute(text("SELECT patient_code, patient_id FROM patients WHERE patient_code LIKE 'S-%'")).all())
    rows = []
    for r in visits.itertuples():
        rows.append({"patient_id": ids[r.patient_code], "visit_date": str(r.visit_date)[:10], "weight_kg": r.weight_kg,
                     "height_cm": r.height_cm, "head_circumference_cm": r.head_circumference_cm, "waz": r.waz,
                     "haz": r.haz, "bmiz": r.bmiz, "hcz": r.hcz, "measurement_type": r.measurement_type,
                     "adjusted_length_height_cm": r.adjusted_length_height_cm, "bmi": r.bmi})
    sql = text(f"INSERT INTO visits ({', '.join(VISIT_COLS)}) VALUES ({', '.join(':' + c for c in VISIT_COLS)})")
    for i in range(0, len(rows), 1000):
        session.execute(sql, rows[i:i + 1000])
    return len(children), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--data", default=str(ROOT / "data"))
    a = ap.parse_args()
    sys.path.insert(0, str(Path.cwd()))
    from db import SessionLocal  # your existing module

    with SessionLocal() as s:
        if a.remove:
            remove_synthetic(s); s.commit(); print("Removed synthetic rows (S-%)."); return
        visits = pd.read_csv(Path(a.data) / "synthetic_visits.csv")
        children = pd.read_csv(Path(a.data) / "synthetic_children.csv")
        try:
            n_c, n_v = load(s, visits, children)
            s.commit()
        except Exception:
            s.rollback(); raise
    print(f"Loaded {n_c} children and {n_v} visits.")


if __name__ == "__main__":
    main()
