"""
Generate the synthetic dataset, the error-injection file, figures and docs/dataset_documentation.md.

    python scripts/make_synthetic.py --children 2000 --seed 42

Outputs (data/ is git-ignored):
    data/synthetic_visits.csv     one row per visit (what the app would store, plus ground truth columns)
    data/synthetic_children.csv   one row per child (archetype, onset, pattern, split, parameters)
    data/synthetic_errors.csv     corrupted copies of 5% of visits for testing the data-quality layer
    docs/dataset_documentation.md and docs/figures/*.png
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from growth import synth as S

KEY_INDICATOR = {
    "normal_typical": "waz", "normal_high_tracker": "waz", "normal_low_tracker": "waz",
    "persistent_underweight": "waz", "persistent_stunting": "haz", "persistent_overweight": "bmiz",
    "gradual_weight_faltering": "waz", "rapid_weight_loss": "waz", "progressive_stunting": "haz",
    "rising_bmi": "bmiz", "multi_indicator_failure": "waz", "hc_deceleration": "hcz", "hc_acceleration": "hcz",
    "recovery_catchup": "waz", "transient_illness_dip": "waz", "isolated_outlier": "multi", "catch_down": "waz",
}
IND_LABEL = {"waz": "weight-for-age z", "haz": "length/height-for-age z", "bmiz": "BMI-for-age z", "hcz": "head-circumference z",
             "multi": "weight, height and head z"}
BLUE, ORANGE, GREY, RED, GREEN = "#1f77b4", "#e07b00", "#8a949c", "#c0392b", "#2e8b57"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------ figures
def fig_archetypes(visits, children, path):
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(6, 3, figsize=(13, 17), sharey=True)
    for ax, arch in zip(axes.ravel(), S.ARCHETYPES):
        key = KEY_INDICATOR[arch.name]
        codes = children.loc[children.archetype == arch.name, "patient_code"].to_numpy()
        codes = rng.choice(codes[[len(visits[visits.patient_code == c]) >= 4 for c in codes]], size=7, replace=False)
        for j, code in enumerate(codes):
            g = visits[visits.patient_code == code].sort_values("visit_index")
            if key == "multi":
                if j >= 4:
                    continue
                for ind, col in (("waz", BLUE), ("haz", ORANGE), ("hcz", GREEN)):
                    ax.plot(g.age_months, g[ind], "-o", ms=2.5, lw=0.9, color=col, alpha=.8)
            else:
                ax.plot(g.age_months, g[key], "-o", ms=2.5, lw=0.9, color=BLUE, alpha=.75)
                on = children.loc[children.patient_code == code, "onset_visit_index"].iloc[0]
                if on != "":
                    ax.plot(g.age_months.iloc[int(on)], g[key].iloc[int(on)], "o", ms=6, color=RED, zorder=5)
        for y, ls in ((2, "--"), (-2, "--"), (0, "-")):
            ax.axhline(y, color=GREY, lw=0.8, ls=ls, alpha=.8)
        n = int((children.archetype == arch.name).sum())
        ax.set_title(f"{arch.name}  (n={n}){'  [deteriorating]' if arch.deteriorating else ''}", fontsize=9.5,
                     color=RED if arch.deteriorating else "black")
        ax.set_xlabel(f"age (months); {IND_LABEL[key]}", fontsize=7.5)
        ax.set_ylim(-5, 5); ax.set_xlim(-1, 61); ax.tick_params(labelsize=7.5)
    for ax in axes.ravel()[len(S.ARCHETYPES):]:
        ax.axis("off")
    fig.suptitle("Trajectory types: 7 example children each (red dot = onset of deterioration; dashed = -2 / +2 SD)", fontsize=11, y=0.995)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def fig_overview(visits, children, path):
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8))
    ax[0, 0].hist(children.n_visits, bins=np.arange(0.5, children.n_visits.max() + 1.5), color=BLUE, edgecolor="white")
    ax[0, 0].set(title="Visits per child", xlabel="visits", ylabel="children")
    ax[0, 1].hist(visits.age_months, bins=np.arange(0, 62, 2), color=BLUE, edgecolor="white")
    ax[0, 1].set(title="Age at visit", xlabel="months", ylabel="visits")
    order = [a.name for a in S.ARCHETYPES]
    cnt = children.groupby("archetype").size().reindex(order)
    det = [S.ARCH_BY_NAME[a].deteriorating for a in order]
    ax[1, 0].barh(range(len(order)), cnt.values, color=[RED if d else BLUE for d in det])
    ax[1, 0].set_yticks(range(len(order))); ax[1, 0].set_yticklabels(order, fontsize=7.5); ax[1, 0].invert_yaxis()
    ax[1, 0].set(title="Children per trajectory type (red = deteriorating)", xlabel="children")
    pat = children.pattern.value_counts()
    ax[1, 1].bar(pat.index, pat.values, color=BLUE)
    ax[1, 1].set(title="Visit-schedule pattern", ylabel="children"); ax[1, 1].tick_params(axis="x", labelrotation=25, labelsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ------------------------------------------------------------------------------------------ summaries
def baseline_flag(df):
    return (df.waz < -2) | (df.haz < -2) | (df.bmiz.abs() > 2) | (df.hcz.abs() > 2)


def baseline_coverage(visits, children):
    v = visits.assign(flag=baseline_flag(visits))
    rows = []
    for arch in S.ARCHETYPES:
        ch = children[children.archetype == arch.name]
        g = v[v.archetype == arch.name]
        first = g[g.flag].groupby("patient_code").visit_index.min()
        ever = round(100 * len(first) / len(ch), 1)
        if arch.deteriorating:
            on = ch.set_index("patient_code").onset_visit_index.astype(int)
            lead = (first - on.reindex(first.index))
            rows.append({"trajectory type": arch.name, "children ever flagged (%)": ever,
                         "flagged before onset (%)": round(100 * (lead < 0).sum() / len(ch), 1),
                         "median visits from onset to first flag": float(lead[lead >= 0].median()) if (lead >= 0).any() else "-",
                         "never flagged (%)": round(100 * (1 - len(first) / len(ch)), 1)})
        else:
            rows.append({"trajectory type": arch.name, "children ever flagged (%)": ever, "flagged before onset (%)": "-",
                         "median visits from onset to first flag": "-", "never flagged (%)": round(100 - ever, 1)})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--children", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    (ROOT / "data").mkdir(exist_ok=True); (ROOT / "docs/figures").mkdir(parents=True, exist_ok=True)
    visits, children = S.generate_dataset(args.children, args.seed)
    errors = S.make_error_file(visits, args.seed)
    visits.to_csv(ROOT / "data/synthetic_visits.csv", index=False)
    children.to_csv(ROOT / "data/synthetic_children.csv", index=False)
    errors.to_csv(ROOT / "data/synthetic_errors.csv", index=False)
    fig_archetypes(visits, children, ROOT / "docs/figures/trajectory_types.png")
    fig_overview(visits, children, ROOT / "docs/figures/dataset_overview.png")

    # ---- summary tables
    arch_rows = []
    for a in S.ARCHETYPES:
        g, ch = visits[visits.archetype == a.name], children[children.archetype == a.name]
        arch_rows.append({"trajectory type": a.name, "children": len(ch), "visits": len(g),
                          "mean visits/child": round(ch.n_visits.mean(), 1), "label": "deteriorating" if a.deteriorating else "not deteriorating",
                          "positive visits": int(g.label_emerging.sum()), "what it represents": a.description})
    arch_tbl = pd.DataFrame(arch_rows)
    bands = pd.cut(visits.age_months, [-0.01, 6, 12, 24, 36, 48, 60.01], labels=["0-6", "6-12", "12-24", "24-36", "36-48", "48-60"])
    age_tbl = visits.groupby(bands, observed=True).size().rename("visits").reset_index().rename(columns={"age_months": "age (months)"})
    nv = pd.cut(children.n_visits, [0, 1, 2, 4, 8, 12, 99], labels=["1", "2", "3-4", "5-8", "9-12", "13+"])
    nv_tbl = children.groupby(nv, observed=True).size().rename("children").reset_index().rename(columns={"n_visits": "visits per child"})
    pat_tbl = children.pattern.value_counts().rename_axis("pattern").reset_index(name="children")
    split_tbl = children.groupby("split").agg(children=("patient_code", "size")).join(
        visits.groupby("split").agg(visits=("patient_code", "size"), positive_visits=("label_emerging", "sum"))).reset_index()
    z_tbl = pd.DataFrame([{"indicator": n, "min": visits[c].min(), "median": visits[c].median(), "max": visits[c].max(),
                           "visits beyond -2 (%)": round(100 * (visits[c] < -2).mean(), 1), "visits beyond +2 (%)": round(100 * (visits[c] > 2).mean(), 1)}
                          for n, c in (("weight-for-age", "waz"), ("length/height-for-age", "haz"), ("BMI-for-age", "bmiz"), ("head circumference", "hcz"))])
    z_tbl[["min", "median", "max"]] = z_tbl[["min", "median", "max"]].round(2)
    mt = visits.assign(age_group=np.where(visits.age_days < 731, "under 2 years", "2 years and over")).groupby(["age_group", "measurement_type"]).size().rename("visits").reset_index()
    base_tbl = baseline_coverage(visits, children)
    err_tbl = errors.groupby(["error_type", "caught_by"]).size().unstack(fill_value=0)
    err_tbl.insert(0, "records", err_tbl.sum(axis=1))
    err_tbl = err_tbl.reset_index()
    detect = errors.caught_by.value_counts()
    undetected = int(detect.get("undetected", 0))

    pos_visits, n_det = int(visits.label_emerging.sum()), int(children.deteriorating.sum())
    doc = f"""# Synthetic longitudinal growth dataset: documentation

*Generated by `scripts/make_synthetic.py --children {args.children} --seed {args.seed}` (fully reproducible).*

**This is synthetic data.** It exists to develop, test and demonstrate the pipeline. It contains no real patients, the
trajectory parameters are project assumptions (not clinical facts), and any result obtained on it shows technical feasibility only,
not clinical validity.

## 1. What is in it

| item | value |
|---|---|
| children | {len(children)} ({int((children.sex == 'Boy').sum())} boys, {int((children.sex == 'Girl').sum())} girls) |
| visits | {len(visits)} (mean {len(visits) / len(children):.1f} per child, range {children.n_visits.min()} to {children.n_visits.max()}) |
| ages covered | 0 to 60 months (every visit is inside the 5-year project scope) |
| trajectory types | {len(S.ARCHETYPES)} ({sum(a.deteriorating for a in S.ARCHETYPES)} deteriorating, {sum(not a.deteriorating for a in S.ARCHETYPES)} not) |
| deteriorating children | {n_det} ({100 * n_det / len(children):.0f}%); positive visits (at or after onset): {pos_visits} ({100 * pos_visits / len(visits):.1f}% of visits) |
| patient codes | `S-0001` and up (the app's code generator only uses `A-####`, so real and synthetic children never clash) |
| reference date | no visit is later than {S.ANCHOR_DATE:%d %b %Y}; no child has two visits on the same date; visits are at least {S.MIN_GAP_DAYS} days apart |

Files: `data/synthetic_visits.csv`, `data/synthetic_children.csv`, `data/synthetic_errors.csv`, and the figures in `docs/figures/`.

![Trajectory types](figures/trajectory_types.png)

![Dataset overview](figures/dataset_overview.png)

## 2. How the data are produced

1. **Trajectory type per child.** The {len(S.ARCHETYPES)} types below fix how the weight, length/height and head-circumference z-scores
   evolve. Shares are exact (counts rounded to sum to {len(children)}).
2. **Child-level size factor.** Each child gets a baseline (weight z ~ N(0, 0.8); height z and head z correlated with it) so the three
   measurements move together as they do in real children.
3. **Onset.** For deteriorating types the onset visit is chosen so there are at least 2 visits before and at least 2 visits from onset on.
   The z-scores change after the onset age; the label switches on at the onset visit.
4. **Noise.** Correlated (AR(1)) noise with a memory of {S.NOISE_TAU_MONTHS:.0f} months and SD {S.NOISE_SD['w']}/{S.NOISE_SD['h']}/{S.NOISE_SD['c']} (weight/height/head),
   plus independent measurement noise SD {S.MEASUREMENT_NOISE_SD}. A child therefore tends to stay near their own line instead of jumping at random.
5. **Visit schedule** (5 patterns, see section 4), including children with only 1 or 2 visits so the "insufficient longitudinal history"
   behaviour can be tested.
6. **From z-score to measurement.** The target z-score is converted to kilograms or centimetres with the inverse WHO LMS formula,
   rounded the way a clinic records it (weight 0.01 kg, length/height and head 0.1 cm), and sent back through the same WHO engine the
   app uses (`who_zscore.analyse_child`). **Every stored z-score, BMI and category is therefore exactly what the app would have saved.**
7. **Length versus standing height.** Under 2 years the length table is used and from 731 days the height table. About 8% of children aged
   12-24 months are recorded as *standing* and about 10% of children aged 2+ as *recumbent*; their recorded value is shifted by 0.7 cm so
   the engine's WHO adjustment returns the intended z-score. This exercises the recumbent/standing logic.
8. **Plausibility.** If any indicator would fall outside the WHO plausibility range (WAZ -6 to +5, HAZ -6 to +6, BMIZ and HCZ -5 to +5),
   which is also the range your app's dynamic limits accept, that visit's z-scores are scaled toward zero until it fits
   ({int(children.visits_rescaled_to_fit_limits.sum())} visits, {100 * children.visits_rescaled_to_fit_limits.sum() / len(visits):.2f}%). No visit in the file is outside the limits.
9. **Split.** Children (never single visits) are assigned to train / validation / test = 70/15/15, stratified by trajectory type, so every
   type appears in every split.

## 3. Trajectory types

{md_table(arch_tbl)}

Rules for the label: `label_emerging = 1` only for visits **at or after the onset visit** of a deteriorating type. Persistent low or high
children, recovering children, catch-down, transient dips and isolated outliers are **0** by design: they are either already identified by the
conventional WHO cut-off, improving, or not sustained. `transient_illness_dip` and `isolated_outlier` are intentionally *hard negatives*: during the
dip itself a model may legitimately raise an alert, so expect some false alarms there and report them in the error analysis.

## 4. Visit patterns

{md_table(pat_tbl)}

- **regular**: well-child schedule 0, 1, 2, 4, 6, 9, 12, 15, 18, 24, 30, 36, 42, 48, 54, 60 months with random start, 15% of visits missed, and age jitter.
- **sparse**: 3-5 visits at random ages. **late_enrolment**: first visit at 12 months or later. **dense_illness**: regular plus 2-4 extra visits about monthly during a short window.
- **short**: 1-2 visits (only for the non-deteriorating types that allow it). **early_start**: newborn series for `catch_down`.

{md_table(nv_tbl)}

{md_table(age_tbl)}

{md_table(mt)}

## 5. Summary statistics

{md_table(z_tbl)}

{md_table(split_tbl)}

## 6. What the conventional rule does with this data

The simplest conventional rule (any indicator beyond the WHO cut-off: weight or height z below -2, BMI or head z beyond +/-2) was applied to every visit.
This is a property of the dataset, not a model result. It shows what is already caught by the cut-off and what is left for trajectory methods.

{md_table(base_tbl)}

Read it like this: children who are flagged before their onset (or from the first visit) are chronic cases, which the cut-off already finds;
the deteriorating children whose first flag comes several visits after onset are the room an earlier-warning method has to work in.

## 7. Error-injection file (`synthetic_errors.csv`)

{len(errors)} corrupted copies of real visits ({100 * len(errors) / len(visits):.0f}%), cycling through {len(S.ERROR_TYPES)} error types. **They are not loaded
into the database.** Column `caught_by` says which defence of the app stops the record first (in the order the app applies them);
`undetected` means that nothing in the current app catches it.

{md_table(err_tbl)}

Overall: {len(errors) - undetected} of {len(errors)} are stopped ({100 * (len(errors) - undetected) / len(errors):.0f}%); {undetected} are not
(check the rows with `caught_by == undetected` to see which error types currently need another layer, for example a previous-visit or a cross-field check).

## 8. Data dictionary

**synthetic_visits.csv** (one row per visit; the first block matches the app's `patients` and `visits` tables)

| column | meaning |
|---|---|
| patient_code, sex, date_of_birth | child identity (`Boy` / `Girl`) |
| visit_date | date of the visit |
| weight_kg, height_cm, head_circumference_cm | recorded measurements (`height_cm` is length or standing height as recorded) |
| measurement_type | `Recumbent Length` or `Standing Height` |
| adjusted_length_height_cm, bmi | value after the WHO +/-0.7 cm adjustment, and BMI computed from it |
| waz, haz, bmiz, hcz | WHO z-scores: weight-, length/height-, BMI-, head-circumference-for-age |
| *_category | WHO classification text for each z-score |
| visit_index, age_days, age_months | position in the child's series (0-based) and age |
| **label_emerging** | ground-truth label for the visit (0/1), described above |
| **archetype**, **split** | trajectory type and train/validation/test assignment |

> **Leakage warning.** `label_emerging`, `archetype`, `split`, `visit_index` and the child-level onset columns are ground truth or bookkeeping. They must **never** be
> used as model inputs. Compute features only from the measurements and z-scores up to the visit being scored.

**synthetic_children.csv**: patient_code, sex, date_of_birth, archetype, deteriorating (0/1), pattern, n_visits, onset_visit_index, onset_age_months,
first_age_months, last_age_months, visits_rescaled_to_fit_limits, split, params_json (the random parameters used for the child).

**synthetic_errors.csv**: error_id, patient_code, source_visit_index, error_type, the corrupted fields, and `caught_by`.

## 9. Limitations

- The trajectory shapes (linear drifts, plateaus, rebounds) and every parameter are assumptions; real children have messier patterns, illness clusters and unmeasured causes.
- Noise is Gaussian and the same for all children; there is no real-world missingness mechanism beyond skipped visits and no recording of gestational age or birth weight.
- Results on this dataset (including lead times) depend on how it was generated. Report them as technical feasibility, never as clinical evidence.
- The dataset deliberately over-represents unusual trajectories (30% deteriorating) compared with a real clinic population.

## 10. Reproduce and load

```
python scripts/make_synthetic.py --children {args.children} --seed {args.seed}      # regenerate everything
python scripts/load_synthetic_to_db.py                       # insert the S- children into MySQL (re-runnable)
pytest -q tests/test_synth.py                                # checks listed in the test file
```
"""
    (ROOT / "docs/dataset_documentation.md").write_text(doc, encoding="utf-8")
    print(f"children {len(children)}  visits {len(visits)}  positives {pos_visits}  errors {len(errors)} (undetected {undetected})")


if __name__ == "__main__":
    main()
