import importlib.util
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import who_zscore as who  # noqa: E402
from growth import synth as S  # noqa: E402

spec = importlib.util.spec_from_file_location("loader", ROOT / "scripts/load_synthetic_to_db.py")
loader = importlib.util.module_from_spec(spec); spec.loader.exec_module(loader)


@pytest.fixture(scope="module")
def data():
    v, c = S.generate_dataset(400, seed=7)
    return v, c, S.make_error_file(v, seed=7)


def test_deterministic():
    a, _ = S.generate_dataset(60, seed=1)
    b, _ = S.generate_dataset(60, seed=1)
    pd.testing.assert_frame_equal(a, b)
    c, _ = S.generate_dataset(60, seed=2)
    assert not a.equals(c)


def test_shares_and_all_archetypes(data):
    _, c, _ = data
    assert set(c.archetype) == {a.name for a in S.ARCHETYPES}
    for a in S.ARCHETYPES:
        assert abs((c.archetype == a.name).mean() - a.share) < 0.01


def test_no_visit_outside_who_limits(data):
    v, _, _ = data
    for col, (lo, hi) in {"waz": (-6, 5), "haz": (-6, 6), "bmiz": (-5, 5), "hcz": (-5, 5)}.items():
        assert v[col].between(lo, hi).all()


def test_engine_reproduces_stored_scores(data):
    v, _, _ = data
    for r in v.sample(150, random_state=0).itertuples():
        a = who.analyse_child(r.sex, r.age_days, r.weight_kg, r.height_cm, r.measurement_type, r.head_circumference_cm)["results"]
        assert abs(a["wfa"]["z"] - r.waz) < 0.01 and abs(a["lhfa"]["z"] - r.haz) < 0.01
        assert abs(a["bfa"]["z"] - r.bmiz) < 0.01 and abs(a["hcfa"]["z"] - r.hcz) < 0.01


def test_labels_consistent(data):
    v, c, _ = data
    det = c.set_index("patient_code")
    for arch in S.ARCHETYPES:
        g = v[v.archetype == arch.name]
        if not arch.deteriorating:
            assert g.label_emerging.sum() == 0
    d = v[v.archetype.map(lambda a: S.ARCH_BY_NAME[a].deteriorating)]
    on = d.patient_code.map(det.onset_visit_index).astype(int)
    assert ((d.visit_index >= on).astype(int) == d.label_emerging).all()
    assert (d.groupby("patient_code").label_emerging.sum() >= 2).all()


def test_dates_and_ages(data):
    v, _, _ = data
    assert not v.duplicated(["patient_code", "visit_date"]).any()
    vd, dob = pd.to_datetime(v.visit_date), pd.to_datetime(v.date_of_birth)
    assert (vd >= dob).all() and (vd <= pd.Timestamp(S.ANCHOR_DATE)).all()
    assert ((vd - dob).dt.days == v.age_days).all()
    assert v.age_days.between(0, S.MAX_AGE_DAYS).all()
    gaps = v.sort_values(["patient_code", "visit_date"]).groupby("patient_code").visit_date.apply(lambda s: pd.to_datetime(s).diff().dt.days.min())
    assert (gaps.dropna() >= S.MIN_GAP_DAYS).all()


def test_splits_by_child_and_stratified(data):
    v, c, _ = data
    assert v.groupby("patient_code").split.nunique().max() == 1
    assert set(c.split) == {"train", "validation", "test"}
    assert abs((c.split == "train").mean() - 0.70) < 0.03
    assert (c.groupby("archetype").split.nunique() == 3).all()


def test_visit_count_variety(data):
    _, c, _ = data
    assert c.n_visits.min() == 1 and c.n_visits.max() >= 12
    assert (c.n_visits <= 2).sum() > 0
    assert c[c.deteriorating == 1].n_visits.min() >= 4


def test_recumbent_standing_both_present(data):
    v, _, _ = data
    young = v[v.age_days < 731]; old = v[v.age_days >= 731]
    assert (young.measurement_type == who.STANDING).any() and (old.measurement_type == who.RECUMBENT).any()


def test_error_file(data):
    v, _, e = data
    assert set(e.error_type) <= set(S.ERROR_TYPES) and e.error_type.nunique() >= len(S.ERROR_TYPES) - 1
    assert len(e) >= 0.04 * len(v)
    assert e.caught_by.notna().all()


DDL = ["""CREATE TABLE patients (patient_id INTEGER PRIMARY KEY AUTOINCREMENT, patient_code TEXT UNIQUE NOT NULL,
          date_of_birth DATE NOT NULL, sex TEXT NOT NULL)""",
       """CREATE TABLE visits (visit_id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id INTEGER NOT NULL, visit_date DATE NOT NULL,
          weight_kg REAL, height_cm REAL, head_circumference_cm REAL, waz REAL, haz REAL, bmiz REAL, hcz REAL,
          measurement_type TEXT, adjusted_length_height_cm REAL, bmi REAL, UNIQUE (patient_id, visit_date))"""]


def test_loader_roundtrip_and_idempotent(data):
    v, c, _ = data
    eng = create_engine("sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False})
    with eng.begin() as cx:
        for s in DDL:
            cx.execute(text(s))
        cx.execute(text("INSERT INTO patients (patient_code, date_of_birth, sex) VALUES ('A-0001','2025-01-01','Girl')"))
        cx.execute(text("INSERT INTO visits (patient_id, visit_date, weight_kg, height_cm) VALUES (1,'2025-02-01',4,55)"))
    with sessionmaker(bind=eng)() as s:
        for _ in range(2):  # re-run must not duplicate
            assert loader.load(s, v, c) == (len(c), len(v))
            s.commit()
        n = lambda q: s.execute(text(q)).scalar()
        assert n("SELECT COUNT(*) FROM patients WHERE patient_code LIKE 'S-%'") == len(c)
        assert n("SELECT COUNT(*) FROM visits") == len(v) + 1
        assert n("SELECT COUNT(*) FROM patients WHERE patient_code='A-0001'") == 1
        loader.remove_synthetic(s); s.commit()
        assert n("SELECT COUNT(*) FROM patients") == 1 and n("SELECT COUNT(*) FROM visits") == 1
