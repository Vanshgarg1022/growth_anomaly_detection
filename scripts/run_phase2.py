import sys
import os
import pandas as pd


# Allow importing from root and growth package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from growth.features import build_feature_matrix
from growth.baseline import evaluate_baseline_rules

def main():
    print("Loading synthetic datasets...")
    visits = pd.read_csv("data/synthetic_visits.csv")
    truth = pd.read_csv("data/synthetic_children.csv")

    print("Building time-aware feature matrix...")
    feat_df = build_feature_matrix(visits, truth)
    
    print("Evaluating conventional rule baseline...")
    def apply_baseline(row):
        z_dict = {ind: row[f"{ind}_current_z"] for ind in ("waz", "haz", "bmiz", "hcz")}
        status, rules = evaluate_baseline_rules(z_dict)
        return pd.Series([status, ",".join(rules)], index=["baseline_status", "fired_rules"])

    baseline_results = feat_df.apply(apply_baseline, axis=1)
    feat_df = pd.concat([feat_df, baseline_results], axis=1)

    out_path = "data/asof_features.csv"
    feat_df.to_csv(out_path, index=False)
    print(f"Phase 2 complete! Features saved to {out_path} ({feat_df.shape[0]} rows, {feat_df.shape[1]} cols).")

if __name__ == "__main__":
    main()
