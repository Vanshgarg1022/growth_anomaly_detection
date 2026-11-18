import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import who_zscore as W  # noqa: E402

REF_CSV = ROOT / "tests" / "reference" / "who_macro_survey_z_st.csv"
SD_COLS = {"SD3neg": -3, "SD2neg": -2, "SD1neg": -1, "SD0": 0, "SD1": 1, "SD2": 2, "SD3": 3}


# ----------------------------------------------------------------- table selection
@pytest.mark.parametrize("sex,expected", [("Boy", "boys"), ("Girl", "girls"), ("male", "boys"), ("F", "girls")])
def test_sex_normalisation(sex, expected):
    assert W.normalise_sex(sex) == expected


def test_unknown_sex_rejected():
    with pytest.raises(ValueError):
        W.normalise_sex("other")


@pytest.mark.parametrize("sex", ["Boy", "Girl"])
def test_wfa_table_depends_on_sex(sex):
    _, name = W.select_table("wfa", sex, 400)
    assert name == f"wfa_{W.normalise_sex(sex)}_0-to-5-years_zscores.xlsx"
    assert (W.TABLE_DIR / name).exists()


@pytest.mark.parametrize("days,part", [(0, "0-2"), (400, "0-2"), (730, "0-2"), (731, "2-5"), (1826, "2-5")])
def test_length_height_table_switches_at_2_years(days, part):
    (_, _, got), name = W.select_table("lhfa", "Boy", days)
    assert got == part and part.replace("-", "-to-") in name
    (_, _, got), _ = W.select_table("bfa", "Girl", days)
    assert got == part


def test_age_out_of_range_rejected():
    with pytest.raises(ValueError):
        W.select_table("wfa", "Boy", W.MAX_AGE_DAYS + 1)
    with pytest.raises(ValueError):
        W.select_table("wfa", "Boy", -1)


def test_all_reference_tables_load_and_are_complete():
    for ind, parts in W.TABLE_FILES.items():
        for part in parts:
            for sex in ("boys", "girls"):
                t = W.load_table(ind, sex, part)
                assert t["Month"].is_monotonic_increasing and t.notna().all().all()
                expected_len = {"all": 61, "0-2": 25, "2-5": 37}[part]
                assert len(t) == expected_len, (ind, sex, part)


def test_users_girls_wfa_file_is_the_one_in_use():
    raw = pd.read_excel(W.TABLE_DIR / "wfa_girls_0-to-5-years_zscores.xlsx")
    assert raw.loc[0, "M"] == pytest.approx(3.2322) and raw.loc[60, "M"] == pytest.approx(18.2193)
    assert W.load_table("wfa", "Girl", "all").loc[12, "M"] == pytest.approx(raw.loc[12, "M"])


# ----------------------------------------------------------------- maths
@pytest.mark.parametrize("sex", ["boys", "girls"])
@pytest.mark.parametrize("ind,part", [("wfa", "all"), ("hcfa", "all"), ("lhfa", "0-2"), ("lhfa", "2-5"),
                                      ("bfa", "0-2"), ("bfa", "2-5")])
def test_median_gives_zero_and_sd_columns_round_trip(sex, ind, part):
    """At every monthly row: y = M -> z = 0, and the WHO SD1/SD2/.. columns of the Excel file give back
    +-1/2/3 (tolerance covers the 1-decimal rounding of those columns)."""
    raw = pd.read_excel(W.TABLE_DIR / W.table_file_name(ind, sex, part))
    raw.columns = [c.strip() for c in raw.columns]
    for _, r in raw.iterrows():
        L, M, S = r["L"], r["M"], r["S"]
        assert W.zscore_from_lms(M, L, M, S) == pytest.approx(0, abs=1e-9)
        for col, n in SD_COLS.items():
            z = W.zscore_from_lms(r[col], L, M, S, extend=ind in W.EXTEND_BEYOND_3SD)
            assert abs(z - n) < 0.2, (ind, sex, r["Month"], col, z)


def test_extension_beyond_3sd_is_continuous_and_monotonic():
    L, M, S = -0.2, 10.0, 0.12
    sd3 = W._cutoff(L, M, S, 3)
    assert W.zscore_from_lms(sd3, L, M, S, extend=True) == pytest.approx(3)
    ys = np.linspace(sd3 * 0.9, sd3 * 1.6, 50)
    zs = [W.zscore_from_lms(y, L, M, S, extend=True) for y in ys]
    assert all(b > a for a, b in zip(zs, zs[1:]))
    # beyond +3 SD the z-score is linear in the measurement: 3 + (y - SD3) / (SD3 - SD2)
    sd2, y = W._cutoff(L, M, S, 2), sd3 * 1.5
    assert W.zscore_from_lms(y, L, M, S, extend=True) == pytest.approx(3 + (y - sd3) / (sd3 - sd2))
    sd3n, sd2n = W._cutoff(L, M, S, -3), W._cutoff(L, M, S, -2)
    yn = sd3n * 0.8
    assert W.zscore_from_lms(yn, L, M, S, extend=True) == pytest.approx(-3 + (yn - sd3n) / (sd2n - sd3n))


def test_l_equal_zero_branch():
    assert W.zscore_from_lms(12.0, 0, 10.0, 0.1) == pytest.approx(np.log(1.2) / 0.1)


def test_nonpositive_measurement_rejected():
    with pytest.raises(ValueError):
        W.zscore_from_lms(0, 1, 10, 0.1)


# ----------------------------------------------------------------- mapping
@pytest.mark.parametrize("ind,z,expected", [
    ("wfa", -3.01, "Severely underweight"), ("wfa", -2.5, "Underweight"), ("wfa", -2.0, "Normal"),
    ("wfa", 2.0, "Normal"), ("wfa", 2.01, "Above normal range (> +2 SD)"),
    ("lhfa", -3.5, "Severely stunted"), ("lhfa", -2.2, "Stunted"), ("lhfa", 0, "Normal"),
    ("lhfa", 3.5, "Very tall (> +3 SD)"),
    ("bfa", -3.5, "Severely wasted (severe thinness)"), ("bfa", -2.5, "Wasted (thinness)"),
    ("bfa", 1.0, "Normal"), ("bfa", 1.5, "Possible risk of overweight"), ("bfa", 2.5, "Overweight"),
    ("bfa", 3.5, "Obese"),
    ("hcfa", -3.5, "Severe microcephaly"), ("hcfa", -2.5, "Microcephaly"), ("hcfa", 0, "Normal"),
    ("hcfa", 2.5, "Macrocephaly"), ("hcfa", 3.5, "Severe macrocephaly"),
])
def test_classification(ind, z, expected):
    assert W.classify(ind, z) == expected


# ----------------------------------------------------------------- recumbent vs standing
def test_length_height_adjustment_rules():
    assert W.adjust_length_height(80.0, W.RECUMBENT, 500) == (80.0, None)           # <2y, length: as is
    assert W.adjust_length_height(80.0, W.STANDING, 1000) == (80.0, None)           # >=2y, height: as is
    cm, note = W.adjust_length_height(80.0, W.STANDING, 500)                         # <2y, standing
    assert cm == pytest.approx(80.7) and note
    cm, note = W.adjust_length_height(100.0, W.RECUMBENT, 1000)                      # >=2y, recumbent
    assert cm == pytest.approx(99.3) and note
    with pytest.raises(ValueError):
        W.adjust_length_height(80.0, "Sitting", 500)


def test_same_child_measured_both_ways_gives_same_zscore():
    lying = W.analyse_child("Girl", 500, 10.0, 80.7, W.RECUMBENT, 46.0)
    standing = W.analyse_child("Girl", 500, 10.0, 80.0, W.STANDING, 46.0)
    for k in ("wfa", "lhfa", "bfa", "hcfa"):
        assert lying["results"][k]["z"] == pytest.approx(standing["results"][k]["z"])
    assert lying["bmi"] == pytest.approx(standing["bmi"])


# ----------------------------------------------------------------- sample children
# (sex, age in days, weight kg, length/height cm, measurement type, head cm)
SAMPLES = [
    ("Boy", 0, 3.3, 49.9, W.RECUMBENT, 34.5),
    ("Girl", 0, 3.2, 49.1, W.RECUMBENT, 33.9),
    ("Boy", 91, 6.4, 61.4, W.RECUMBENT, 40.0),
    ("Girl", 183, 7.3, 65.7, W.RECUMBENT, 52.0 - 8.0),
    ("Boy", 365, 9.6, 75.7, W.RECUMBENT, 46.1),
    ("Girl", 548, 10.8, 80.7, W.RECUMBENT, 46.2),
    ("Boy", 730, 12.2, 87.8, W.RECUMBENT, 48.3),
    ("Girl", 731, 11.5, 86.4, W.STANDING, 47.2),
    ("Boy", 1096, 14.3, 96.1, W.STANDING, 49.5),
    ("Girl", 1461, 16.3, 103.1, W.STANDING, 50.0),
    ("Boy", 1826, 18.3, 110.0, W.STANDING, 50.7),
    ("Girl", 1826, 18.2, 109.4, W.STANDING, 49.9),
]


@pytest.mark.parametrize("sample", SAMPLES, ids=lambda s: f"{s[0]}-{s[1]}d")
def test_sample_children_are_sane(sample):
    sex, days, wt, ht, mt, hc = sample
    out = W.analyse_child(sex, days, wt, ht, mt, hc)
    assert set(out["results"]) == {"wfa", "lhfa", "bfa", "hcfa"}
    for r in out["results"].values():
        assert -4 < r["z"] < 4 and not r["flag"]       # all samples are near-median children
        assert W.normalise_sex(sex) in r["table"]
    assert out["bmi"] == pytest.approx(wt / (out["adjusted_length_height"] / 100) ** 2)


def test_extreme_children_are_flagged_and_classified():
    out = W.analyse_child("Boy", 365, 4.0, 75.0, W.RECUMBENT, 46.0)   # a 1-year-old at 4 kg
    assert out["results"]["wfa"]["z"] < -3 and out["results"]["wfa"]["category"] == "Severely underweight"
    out = W.analyse_child("Girl", 365, 40.0, 75.0, W.RECUMBENT, 46.0)  # implausible 40 kg
    assert out["results"]["wfa"]["flag"]


def test_boy_and_girl_give_different_zscores_for_same_numbers():
    b = W.analyse_child("Boy", 600, 11.0, 82.0, W.RECUMBENT, 47.0)["results"]
    g = W.analyse_child("Girl", 600, 11.0, 82.0, W.RECUMBENT, 47.0)["results"]
    assert all(b[k]["z"] != g[k]["z"] for k in b)


# ----------------------------------------------------------------- WHO macro verification
def _macro_rows():
    d = pd.read_csv(REF_CSV)
    d = d[d["_agedays"] <= 1826]            # the app's scope: up to 5 y 0 m 0 d
    for _, r in d.iterrows():
        sex = "boys" if r.GENDER == 1 else "girls"
        days = int(r["_agedays"])
        mt = W.RECUMBENT if r.measure == "l" else W.STANDING
        if pd.isna(r.WEIGHT) or pd.isna(r.HEIGHT):
            continue
        out = W.analyse_child(sex, days, float(r.WEIGHT), float(r.HEIGHT), mt,
                              None if pd.isna(r.HEAD) else float(r.HEAD))
        yield r, out["results"]


@pytest.mark.parametrize("ind,col", [("wfa", "_ZWEI"), ("lhfa", "_ZLEN"), ("bfa", "_ZBMI"), ("hcfa", "_ZHC")])
def test_matches_who_macro_output(ind, col):
    """Compare with the z-scores from the WHO Anthro/igrowup macro for 498 survey children
    (both sexes, 0-5 y, length and height). Reference values are rounded to 2 decimals."""
    diffs, flag_mismatch = [], 0
    for r, res in _macro_rows():
        if ind not in res:
            continue
        if pd.isna(r[col]):
            continue                         # WHO macro blanks flagged / implausible values
        diffs.append(abs(r[col] - res[ind]["z"]))
    diffs = np.array(diffs)
    assert len(diffs) > 90
    assert diffs.max() < 0.05, diffs.max()
    assert np.mean(diffs <= 0.0101) > 0.95
