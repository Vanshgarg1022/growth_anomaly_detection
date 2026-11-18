"""
Synthetic longitudinal paediatric growth data (0-5 years).

How it works
------------
1. Each child gets a *trajectory type* (archetype) that defines how their weight-, length/height- and
   head-circumference z-scores move over time (17 types, 30% deteriorating, the rest stable, high/low,
   recovering, transient, or noisy but normal).
2. A visit schedule (5 patterns: regular well-child, sparse, dense during illness, late enrolment, 1-2 visits)
   decides at which ages the child is measured.
3. z-scores get correlated (AR(1)) noise, are converted to real kg / cm with the inverse WHO LMS formula,
   rounded like a clinic would record them, and sent back through the SAME WHO engine the app uses
   (who_zscore.analyse_child) so every stored z-score is exactly what the app would have stored.
4. Every visit is kept inside the WHO plausibility limits (the app's dynamic limits reject anything outside them).
5. Ground truth (archetype, onset visit, per-visit label) is stored beside the data, never inferred from z-scores.

Everything is reproducible from the master seed.  This is SYNTHETIC data for developing and testing the pipeline;
it is not clinical data and the parameters are project assumptions, not clinical facts.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Callable

import numpy as np
import pandas as pd

import who_zscore as who

ANCHOR_DATE = date(2026, 10, 31)          # no visit is later than this
SCHEDULE_MONTHS = [0, 1, 2, 4, 6, 9, 12, 15, 18, 24, 30, 36, 42, 48, 54, 60]
MIN_GAP_DAYS = 10                         # two visits of a child are at least this far apart
MAX_AGE_DAYS = 1826                       # 5 years
Z_CLIP = 4.3
NOISE_SD = {"w": 0.35, "h": 0.30, "c": 0.30}
MEASUREMENT_NOISE_SD = 0.12
NOISE_TAU_MONTHS = 4.0                    # memory of the AR(1) noise
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)      # train / validation / test, by child, stratified by archetype


# ----------------------------------------------------------------------------- archetypes
@dataclass(frozen=True)
class Archetype:
    name: str
    share: float                  # share of children
    deteriorating: bool           # label_emerging = 1 from the onset visit onward
    min_visits: int
    description: str
    fn: Callable                  # (rng, t, dt, n, k_on, base) -> (zw, zh, zc, params)
    pattern_override: str | None = None


def a_normal_typical(rng, t, dt, n, k, b):
    return (np.clip(np.full(n, b["w"]), -1.8, 1.8), np.clip(np.full(n, b["h"]), -1.8, 1.8),
            np.clip(np.full(n, b["c"]), -1.8, 1.8), {})


def a_normal_high(rng, t, dt, n, k, b):
    w, h, c = rng.uniform(1.4, 2.2), rng.uniform(0.8, 1.8), rng.uniform(0.3, 1.5)
    return np.full(n, w), np.full(n, h), np.full(n, c), {"w": round(w, 2), "h": round(h, 2)}


def a_normal_low(rng, t, dt, n, k, b):
    w, h, c = -rng.uniform(1.4, 1.9), -rng.uniform(1.0, 1.8), -rng.uniform(0.3, 1.2)
    return np.full(n, w), np.full(n, h), np.full(n, c), {"w": round(w, 2), "h": round(h, 2)}


def a_persistent_underweight(rng, t, dt, n, k, b):
    w, h = -rng.uniform(2.2, 3.0), -rng.uniform(0.8, 1.5)
    return np.full(n, w), np.full(n, h), np.full(n, rng.normal(-0.8, 0.4)), {"w": round(w, 2), "h": round(h, 2)}


def a_persistent_stunting(rng, t, dt, n, k, b):
    h = -rng.uniform(2.2, 3.2)
    w = h * rng.uniform(0.6, 0.9)
    return np.full(n, w), np.full(n, h), np.full(n, -rng.uniform(0.5, 1.5)), {"w": round(w, 2), "h": round(h, 2)}


def a_persistent_overweight(rng, t, dt, n, k, b):
    w, h = rng.uniform(1.6, 2.4), rng.normal(0.2, 0.5)
    return np.full(n, w), np.full(n, h), np.full(n, rng.normal(0.3, 0.5)), {"w": round(w, 2), "h": round(h, 2)}


def a_gradual_weight_faltering(rng, t, dt, n, k, b):
    s, dur = -rng.uniform(0.08, 0.20), rng.uniform(8, 20)
    d = np.minimum(dt, dur)
    return b["w"] + s * d, b["h"] + 0.3 * s * d, np.full(n, b["c"]), {"slope_w": round(s, 3), "duration_m": round(dur, 1)}


def a_rapid_weight_loss(rng, t, dt, n, k, b):
    s, dur = -rng.uniform(0.3, 0.7), rng.uniform(2.5, 5.0)
    d = np.minimum(dt, dur)
    return b["w"] + s * d, b["h"] + 0.15 * s * d, np.full(n, b["c"]), {"slope_w": round(s, 3), "duration_m": round(dur, 1)}


def a_progressive_stunting(rng, t, dt, n, k, b):
    s, dur = -rng.uniform(0.06, 0.15), rng.uniform(12, 24)
    d = np.minimum(dt, dur)
    return b["w"] + 0.4 * s * d, b["h"] + s * d, b["c"] + 0.2 * s * d, {"slope_h": round(s, 3), "duration_m": round(dur, 1)}


def a_rising_bmi(rng, t, dt, n, k, b):
    s, dur = rng.uniform(0.10, 0.25), rng.uniform(8, 20)
    w = np.minimum(b["w"] + s * np.minimum(dt, dur), 3.8)
    return w, np.full(n, b["h"]), np.full(n, b["c"]), {"slope_w": round(s, 3), "duration_m": round(dur, 1)}


def a_recovery_catchup(rng, t, dt, n, k, b):
    sw, rate = -rng.uniform(2.2, 3.0), rng.uniform(0.12, 0.25)
    sh, tw, th = sw * rng.uniform(0.4, 0.8), rng.normal(-0.3, 0.3), rng.normal(-0.5, 0.4)
    t_rec = t[min(1, n - 1)]
    el = np.maximum(t - t_rec, 0)
    w = np.minimum(sw + rate * el, tw)
    h = np.minimum(sh + 0.5 * rate * el, th)
    return w, h, np.full(n, rng.normal(-0.5, 0.5)), {"start_w": round(sw, 2), "rate": round(rate, 3)}


def a_hc_deceleration(rng, t, dt, n, k, b):
    s, dur = -rng.uniform(0.12, 0.25), rng.uniform(10, 24)
    return np.full(n, b["w"]), np.full(n, b["h"]), b["c"] + s * np.minimum(dt, dur), {"slope_c": round(s, 3), "duration_m": round(dur, 1)}


def a_hc_acceleration(rng, t, dt, n, k, b):
    s, dur = rng.uniform(0.12, 0.25), rng.uniform(10, 24)
    return np.full(n, b["w"]), np.full(n, b["h"]), b["c"] + s * np.minimum(dt, dur), {"slope_c": round(s, 3), "duration_m": round(dur, 1)}


def a_transient_dip(rng, t, dt, n, k, b):
    depth, down, up = rng.uniform(0.8, 1.4), rng.uniform(1.5, 2.5), rng.uniform(3.0, 5.0)
    f = np.where(dt < down, depth * dt / down, np.where(dt < down + up, depth * (1 - (dt - down) / up), 0.0))
    return b["w"] - f, b["h"] - 0.3 * f, np.full(n, b["c"]), {"depth": round(depth, 2), "down_m": round(down, 1), "up_m": round(up, 1)}


def a_isolated_outlier(rng, t, dt, n, k, b):
    n_sp = 1 if n < 6 or rng.random() < 0.6 else 2
    idx = rng.choice(n, size=min(n_sp, n), replace=False)
    zw, zh, zc = np.full(n, b["w"]), np.full(n, b["h"]), np.full(n, b["c"])
    which = []
    for i in idx:
        ind = rng.choice(["w", "h", "c"]); off = rng.choice([-1, 1]) * rng.uniform(1.2, 2.0)
        {"w": zw, "h": zh, "c": zc}[ind][i] += off
        which.append({"visit": int(i), "indicator": ind, "offset": round(off, 2)})
    return zw, zh, zc, {"spikes": which}


def a_multi_indicator_failure(rng, t, dt, n, k, b):
    s, dur = -rng.uniform(0.15, 0.30), rng.uniform(8, 16)
    d = np.minimum(dt, dur)
    return b["w"] + s * d, b["h"] + 0.4 * s * d, b["c"] + 0.25 * s * np.maximum(d - 3, 0), {"slope_w": round(s, 3), "duration_m": round(dur, 1)}


def a_catch_down(rng, t, dt, n, k, b):
    sw, tau = rng.uniform(1.3, 2.0), rng.uniform(2.0, 4.0)
    tw, th = rng.normal(0, 0.6), rng.normal(0, 0.6)
    w = tw + (sw - tw) * np.exp(-t / tau)
    h = th + (0.7 * sw - th) * np.exp(-t / tau)
    return w, h, np.full(n, b["c"]), {"start_w": round(sw, 2), "tau_m": round(tau, 1)}


ARCHETYPES: list[Archetype] = [
    Archetype("normal_typical", 0.34, False, 1, "Tracks one centile band with small correlated noise (reference 'healthy' child).", a_normal_typical),
    Archetype("normal_high_tracker", 0.04, False, 1, "Large but healthy: weight and height stay high (about +1.5 to +2.2) without worsening.", a_normal_high),
    Archetype("normal_low_tracker", 0.04, False, 1, "Small but healthy: weight and height stay low-normal (about -1.4 to -1.9).", a_normal_low),
    Archetype("persistent_underweight", 0.04, False, 1, "Weight stays below -2 from the first visit and does not worsen (already caught by the WHO cut-off).", a_persistent_underweight),
    Archetype("persistent_stunting", 0.04, False, 1, "Length/height stays below -2 from the first visit and does not worsen.", a_persistent_stunting),
    Archetype("persistent_overweight", 0.03, False, 1, "Weight high for height from the first visit; BMI z stays about +2 to +3, stable.", a_persistent_overweight),
    Archetype("gradual_weight_faltering", 0.07, True, 6, "After onset, weight z falls slowly (0.08-0.20 per month, 8-20 months); height follows weakly.", a_gradual_weight_faltering),
    Archetype("rapid_weight_loss", 0.04, True, 6, "After onset, weight z falls quickly (0.3-0.7 per month for 2.5-5 months) then stays low.", a_rapid_weight_loss),
    Archetype("progressive_stunting", 0.05, True, 6, "After onset, length/height z falls (0.06-0.15 per month, 12-24 months); weight and head follow weakly.", a_progressive_stunting),
    Archetype("rising_bmi", 0.06, True, 6, "After onset, weight z rises (0.10-0.25 per month) with height unchanged, so BMI z climbs.", a_rising_bmi),
    Archetype("multi_indicator_failure", 0.02, True, 6, "Weight falls, height follows, head circumference falls later: severe combined deterioration.", a_multi_indicator_failure),
    Archetype("hc_deceleration", 0.03, True, 6, "After onset, head-circumference z falls (0.12-0.25 per month) while other indicators are normal.", a_hc_deceleration),
    Archetype("hc_acceleration", 0.03, True, 6, "After onset, head-circumference z rises (0.12-0.25 per month) while other indicators are normal.", a_hc_acceleration),
    Archetype("recovery_catchup", 0.06, False, 5, "Starts low (about -2.2 to -3.0) then catches up towards normal: improving, must not be flagged as worsening.", a_recovery_catchup),
    Archetype("transient_illness_dip", 0.04, False, 5, "A temporary weight dip of 0.8-1.4 z over 2 months that rebounds: hard negative for decline detectors.", a_transient_dip),
    Archetype("isolated_outlier", 0.04, False, 4, "Normal child with 1-2 isolated visits shifted by 1.2-2.0 z (plausible measurement slip): robustness case.", a_isolated_outlier),
    Archetype("catch_down", 0.03, False, 4, "Large at birth (+1.3 to +2.0) and drifts down to the child's own line within months: a normal falling z-score.", a_catch_down, "early_start"),
]
ARCH_BY_NAME = {a.name: a for a in ARCHETYPES}
assert abs(sum(a.share for a in ARCHETYPES) - 1.0) < 1e-9


# ----------------------------------------------------------------------------- visit schedules
def _regular(rng, min_visits):
    start = int(rng.choice([0, 0, 0, 1, 2, 3, 4]))
    idx = [i for i in range(start, len(SCHEDULE_MONTHS)) if i == start or rng.random() < 0.85]
    n_max = int(rng.integers(max(6, min_visits), 15))
    idx = idx[:n_max]
    return np.array([SCHEDULE_MONTHS[i] for i in idx], float)


def _sparse(rng, min_visits):
    n = int(rng.integers(max(3, min_visits), max(6, min_visits + 1)))
    return np.sort(rng.uniform(0, 56, n))


def _short(rng, min_visits):
    return np.sort(rng.uniform(0, 58, int(rng.integers(1, 3))))


def _late(rng, min_visits):
    start = int(rng.integers(6, 10))
    ages = [SCHEDULE_MONTHS[i] for i in range(start, len(SCHEDULE_MONTHS))]
    n = int(rng.integers(max(4, min_visits), 9))
    return np.array(ages[:n], float)


def _dense(rng, min_visits):
    base = list(_regular(rng, max(6, min_visits)))
    if len(base) < 3:
        return np.array(base)
    j = int(rng.integers(1, len(base) - 1))
    extra = [base[j] + k * rng.uniform(0.7, 1.0) for k in range(1, int(rng.integers(3, 6)))]
    return np.sort(np.array(base + extra, float))


def _early_start(rng, min_visits):
    return np.array([0, 1, 2, 3.5, 5, 7, 9, 12, 15, 18, 24][: int(rng.integers(max(min_visits, 6), 12))], float)


PATTERNS = {"regular": _regular, "sparse": _sparse, "short": _short, "late_enrolment": _late, "dense_illness": _dense,
            "early_start": _early_start}


def pick_pattern(rng, arch: Archetype) -> str:
    if arch.pattern_override:
        return arch.pattern_override
    if arch.min_visits <= 1:
        names, p = ["regular", "sparse", "dense_illness", "late_enrolment", "short"], [0.56, 0.14, 0.08, 0.08, 0.14]
    elif arch.min_visits <= 4:
        names, p = ["regular", "sparse", "dense_illness", "late_enrolment"], [0.66, 0.14, 0.10, 0.10]
    else:
        names, p = ["regular", "sparse", "dense_illness", "late_enrolment"], [0.70, 0.10, 0.12, 0.08]
    return str(rng.choice(names, p=p))


def make_ages(rng, arch: Archetype):
    """Return (pattern, ages_days) satisfying the archetype's minimum number of visits."""
    for _ in range(40):
        pattern = pick_pattern(rng, arch)
        months = PATTERNS[pattern](rng, arch.min_visits)
        jitter = rng.normal(0, 0.12 + 0.03 * months)
        jitter = np.clip(jitter, -0.6, 0.6)
        months = np.where(months == 0, np.abs(jitter) * 0.5, months + jitter)
        days = np.round(np.clip(months, 0, 60) * who.DAYS_PER_MONTH).astype(int)
        days = np.sort(days)
        keep = [0]
        for i in range(1, len(days)):
            if days[i] - days[keep[-1]] >= MIN_GAP_DAYS and days[i] <= MAX_AGE_DAYS:
                keep.append(i)
        days = days[keep]
        if len(days) >= arch.min_visits:
            return pattern, days
    pattern = "regular"
    months = np.array(SCHEDULE_MONTHS[: max(arch.min_visits, 8)], float)
    return pattern, np.round(months * who.DAYS_PER_MONTH).astype(int)


# ----------------------------------------------------------------------------- noise and measurements
def ar1_noise(rng, t, sd):
    e = np.empty(len(t))
    e[0] = rng.normal(0, sd)
    for i in range(1, len(t)):
        rho = math.exp(-(t[i] - t[i - 1]) / NOISE_TAU_MONTHS)
        e[i] = rho * e[i - 1] + math.sqrt(1 - rho ** 2) * rng.normal(0, sd)
    return e


def measure_from_z(indicator, sex, age_days, z):
    """Inverse LMS: measurement that has this z-score at this age (same table the engine uses)."""
    key, _ = who.select_table(indicator, sex, age_days)
    L, M, S = who.lms_at(key, who.age_in_months(age_days))
    return who._cutoff(L, M, S, z)


def build_visit(rng, sex, age_days, zw, zh, zc):
    """Turn target z-scores into recorded measurements, re-score them with the engine and make sure
    nothing falls outside the WHO plausibility limits. Returns a dict or None."""
    if age_days < who.LENGTH_TO_HEIGHT_DAYS:
        mtype = who.STANDING if (age_days >= 365 and rng.random() < 0.08) else who.RECUMBENT
    else:
        mtype = who.RECUMBENT if rng.random() < 0.10 else who.STANDING
    for scale in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2):
        tw, th, tc = (float(np.clip(z * scale, -Z_CLIP, Z_CLIP)) for z in (zw, zh, zc))
        weight = round(measure_from_z("wfa", sex, age_days, tw), 2)
        table_height = measure_from_z("lhfa", sex, age_days, th)
        if age_days < who.LENGTH_TO_HEIGHT_DAYS and mtype == who.STANDING:
            recorded = table_height - who.LENGTH_HEIGHT_OFFSET_CM          # standing height reads 0.7 cm lower
        elif age_days >= who.LENGTH_TO_HEIGHT_DAYS and mtype == who.RECUMBENT:
            recorded = table_height + who.LENGTH_HEIGHT_OFFSET_CM          # recumbent length reads 0.7 cm higher
        else:
            recorded = table_height
        height = round(recorded, 1)
        hc = round(measure_from_z("hcfa", sex, age_days, tc), 1)
        if weight <= 0 or height <= 0 or hc <= 0:
            continue
        a = who.analyse_child(sex, age_days, weight, height, mtype, hc)
        if not any(r["flag"] for r in a["results"].values()):
            return {"weight": weight, "height": height, "hc": hc, "mtype": mtype, "analysis": a, "scaled": scale < 1.0}
    return None


# ----------------------------------------------------------------------------- one child
def generate_child(master_seed: int, child_no: int, arch: Archetype):
    rng = np.random.default_rng([master_seed, child_no])
    sex = "Boy" if rng.random() < 0.5 else "Girl"
    pattern, days = make_ages(rng, arch)
    n = len(days)
    t = days / who.DAYS_PER_MONTH
    b0 = rng.normal(0, 0.8)
    base = {"w": b0, "h": 0.6 * b0 + rng.normal(0, 0.5), "c": 0.4 * b0 + rng.normal(0, 0.6)}

    k_on, dt = None, np.zeros(n)
    if arch.deteriorating:
        k_on = int(rng.integers(2, n - 2))                       # >=2 visits before and >=2 from onset on
        dt = np.maximum(t - t[k_on], 0.0)
    elif arch.name == "transient_illness_dip":
        k_d = int(rng.integers(1, n - 3))
        dt = np.where(t >= t[k_d], t - t[k_d], 1e9)               # 1e9 = before the dip: no effect
    zw, zh, zc, params = arch.fn(rng, t, dt, n, k_on, base)
    zw, zh, zc = (np.asarray(z, float) * np.ones(n) for z in (zw, zh, zc))
    zw = zw + ar1_noise(rng, t, NOISE_SD["w"]) + rng.normal(0, MEASUREMENT_NOISE_SD, n)
    zh = zh + ar1_noise(rng, t, NOISE_SD["h"]) + rng.normal(0, MEASUREMENT_NOISE_SD, n)
    zc = zc + ar1_noise(rng, t, NOISE_SD["c"]) + rng.normal(0, MEASUREMENT_NOISE_SD, n)

    visits, scaled = [], 0
    for i in range(n):
        v = build_visit(rng, sex, int(days[i]), zw[i], zh[i], zc[i])
        if v is None:                                             # cannot be made plausible: stop the series here
            break
        scaled += int(v["scaled"])
        visits.append((int(days[i]), v))
    if len(visits) < arch.min_visits:                             # extremely unlikely; caller may retry
        return None
    if k_on is not None and k_on > len(visits) - 3:
        return None
    last_age = visits[-1][0]
    dob = ANCHOR_DATE - timedelta(days=last_age + int(rng.integers(0, 1500)))
    return {"sex": sex, "pattern": pattern, "visits": visits, "k_on": k_on, "params": params,
            "dob": dob, "scaled": scaled, "age_days": days}


# ----------------------------------------------------------------------------- whole dataset
def _counts(n_children: int):
    raw = np.array([a.share * n_children for a in ARCHETYPES])
    c = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - c))[: n_children - c.sum()]:
        c[i] += 1
    return c


def assign_splits(children: pd.DataFrame, seed: int) -> pd.Series:
    rng = np.random.default_rng([seed, 999_001])
    split = pd.Series("train", index=children.index)
    for _, g in children.groupby("archetype"):
        idx = rng.permutation(g.index.to_numpy())
        n_tr = int(round(SPLIT_FRACTIONS[0] * len(idx)))
        n_va = int(round(SPLIT_FRACTIONS[1] * len(idx)))
        split.loc[idx[n_tr:n_tr + n_va]] = "validation"
        split.loc[idx[n_tr + n_va:]] = "test"
    return split


def generate_dataset(n_children: int = 2000, seed: int = 42):
    """Return (visits_df, children_df)."""
    rng = np.random.default_rng([seed, 1])
    order = np.repeat(np.arange(len(ARCHETYPES)), _counts(n_children))
    rng.shuffle(order)

    vrows, crows = [], []
    for i, ai in enumerate(order, start=1):
        arch = ARCHETYPES[ai]
        child = None
        for attempt in range(25):
            child = generate_child(seed, i * 100 + attempt, arch)
            if child is not None:
                break
        if child is None:
            raise RuntimeError(f"could not generate child {i} ({arch.name})")
        code = f"S-{i:04d}"
        k_on = child["k_on"]
        for k, (days, v) in enumerate(child["visits"]):
            a = v["analysis"]; r = a["results"]
            label = int(arch.deteriorating and k >= k_on)
            vrows.append({
                "patient_code": code, "sex": child["sex"], "date_of_birth": child["dob"],
                "visit_date": child["dob"] + timedelta(days=days), "visit_index": k,
                "age_days": days, "age_months": round(days / who.DAYS_PER_MONTH, 3),
                "weight_kg": v["weight"], "height_cm": v["height"], "head_circumference_cm": v["hc"],
                "measurement_type": v["mtype"], "adjusted_length_height_cm": round(a["adjusted_length_height"], 2),
                "bmi": round(a["bmi"], 2),
                "waz": round(r["wfa"]["z"], 3), "haz": round(r["lhfa"]["z"], 3),
                "bmiz": round(r["bfa"]["z"], 3), "hcz": round(r["hcfa"]["z"], 3),
                "waz_category": r["wfa"]["category"], "haz_category": r["lhfa"]["category"],
                "bmiz_category": r["bfa"]["category"], "hcz_category": r["hcfa"]["category"],
                "label_emerging": label, "archetype": arch.name,
            })
        nv = len(child["visits"])
        crows.append({
            "patient_code": code, "sex": child["sex"], "date_of_birth": child["dob"], "archetype": arch.name,
            "deteriorating": int(arch.deteriorating), "pattern": child["pattern"], "n_visits": nv,
            "onset_visit_index": k_on if k_on is not None else "",
            "onset_age_months": round(child["visits"][k_on][0] / who.DAYS_PER_MONTH, 2) if k_on is not None else "",
            "first_age_months": round(child["visits"][0][0] / who.DAYS_PER_MONTH, 2),
            "last_age_months": round(child["visits"][-1][0] / who.DAYS_PER_MONTH, 2),
            "visits_rescaled_to_fit_limits": child["scaled"],
            "params_json": json.dumps(child["params"], sort_keys=True),
        })
    children = pd.DataFrame(crows)
    children["split"] = assign_splits(children, seed)
    visits = pd.DataFrame(vrows).merge(children[["patient_code", "split"]], on="patient_code")
    return visits, children


# ----------------------------------------------------------------------------- error injection
ERROR_TYPES = [
    "weight_in_pounds", "weight_in_grams", "weight_decimal_slip_up", "weight_decimal_slip_down",
    "height_in_inches", "height_in_mm", "weight_height_swapped", "hc_in_inches", "hc_copied_from_height",
    "missing_head_circumference", "zero_weight", "visit_before_birth", "future_visit_date",
    "duplicate_visit_date", "height_drop_plausible", "age_beyond_5_years",
]


def make_error_file(visits: pd.DataFrame, seed: int = 42, fraction: float = 0.05, today: date = date(2030, 1, 1)):
    """Corrupted copies of ~5% of the visits, never meant to be stored. Each row records which defence
    of the app (required field, input range, date check, duplicate check, WHO dynamic limits,
    previous-visit check) would stop it, or 'undetected'."""
    rng = np.random.default_rng([seed, 7])
    prev_by_row = visits.groupby("patient_code").shift(1)
    n_err = int(round(fraction * len(visits)))
    picks = rng.choice(len(visits), size=n_err, replace=False)
    out = []
    for e, ix in enumerate(sorted(picks), start=1):
        row = visits.iloc[ix].to_dict()
        prev = prev_by_row.iloc[ix]
        etype = ERROR_TYPES[(e - 1) % len(ERROR_TYPES)]
        if etype in ("height_drop_plausible", "duplicate_visit_date") and pd.isna(prev["visit_index"]):
            etype = "weight_in_pounds"                                  # needs an earlier visit; use another type
        if etype == "age_beyond_5_years" and row["age_months"] < 40:
            etype = "weight_in_pounds"
        w, h, c = row["weight_kg"], row["height_cm"], row["head_circumference_cm"]
        vdate, dob, days = row["visit_date"], row["date_of_birth"], int(row["age_days"])
        if etype == "weight_in_pounds":          w = round(w * 2.2046, 2)
        elif etype == "weight_in_grams":         w = round(w * 1000, 2)
        elif etype == "weight_decimal_slip_up":  w = round(w * 10, 2)
        elif etype == "weight_decimal_slip_down": w = round(w / 10, 2)
        elif etype == "height_in_inches":        h = round(h / 2.54, 1)
        elif etype == "height_in_mm":            h = round(h * 10, 1)
        elif etype == "weight_height_swapped":   w, h = h, w
        elif etype == "hc_in_inches":            c = round(c / 2.54, 1)
        elif etype == "hc_copied_from_height":   c = h
        elif etype == "missing_head_circumference": c = np.nan
        elif etype == "zero_weight":             w = 0.0
        elif etype == "visit_before_birth":      vdate = dob - timedelta(days=int(rng.integers(1, 30))); days = (vdate - dob).days
        elif etype == "future_visit_date":       vdate = today + timedelta(days=int(rng.integers(30, 400))); days = (vdate - dob).days
        elif etype == "duplicate_visit_date":    vdate = prev["visit_date"]; days = int(prev["age_days"])
        elif etype == "height_drop_plausible":   h = round(h - rng.uniform(1.5, 4.0), 1)
        elif etype == "age_beyond_5_years":      dob = dob - timedelta(days=730); days = (vdate - dob).days
        out.append({"error_id": e, "patient_code": row["patient_code"], "source_visit_index": int(row["visit_index"]),
                    "error_type": etype, "sex": row["sex"], "date_of_birth": dob, "visit_date": vdate, "age_days": days,
                    "weight_kg": w, "height_cm": h, "head_circumference_cm": c, "measurement_type": row["measurement_type"],
                    "_prev_height": prev["adjusted_length_height_cm"], "_prev_hc": prev["head_circumference_cm"]})
    err = pd.DataFrame(out)
    err["caught_by"] = [_first_defence(r) for r in err.to_dict("records")]
    return err.drop(columns=["_prev_height", "_prev_hc"])


def _first_defence(r) -> str:
    """Which layer of the app would reject this record first (order of the app's checks)."""
    w, h, c = r["weight_kg"], r["height_cm"], r["head_circumference_cm"]
    if pd.isna(c) or pd.isna(w) or pd.isna(h):                          return "required_field"
    if w <= 0 or h <= 0 or c <= 0:                                      return "required_field"
    if r["error_type"] == "duplicate_visit_date":                       return "duplicate_check"
    if r["visit_date"] < r["date_of_birth"] or r["visit_date"] > ANCHOR_DATE:  return "date_check"
    if r["age_days"] > MAX_AGE_DAYS:                                    return "age_scope"
    if w > 50 or h > 150 or c > 60:                                     return "input_range"
    a = who.analyse_child(r["sex"], int(r["age_days"]), float(w), float(h), r["measurement_type"], float(c))
    if any(x["flag"] for x in a["results"].values()):                   return "dynamic_who_limits"
    ph, pc = r["_prev_height"], r["_prev_hc"]
    if not pd.isna(ph) and float(ph) - a["adjusted_length_height"] > 1.0: return "previous_visit_check"
    if not pd.isna(pc) and float(pc) - c > 0.5:                          return "previous_visit_check"
    return "undetected"
