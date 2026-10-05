#!/usr/bin/env python3
"""Verify the archived admission-control results against manuscript quantities."""
from pathlib import Path
import json
import math
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "reference_run"

def close(a, b, tol=5e-3):
    return math.isclose(float(a), float(b), abs_tol=tol, rel_tol=0.0)

cfg = json.loads((RESULTS / "run_config.json").read_text())
summary = pd.read_csv(RESULTS / "summary_raw.csv")
stats = pd.read_csv(RESULTS / "summary_statistics_ci.csv").set_index("policy")
ranking = pd.read_csv(RESULTS / "sla_constrained_policy_ranking.csv")
paired = pd.read_csv(RESULTS / "paired_significance_net_profit.csv")
improvement = pd.read_csv(RESULTS / "policy_improvement_vs_conservative.csv").set_index("policy")

assert cfg["monte_carlo_runs"] == 30
assert cfg["decision_epochs"] == 288
assert cfg["training_episodes"] == 120
assert cfg["workload_data_split"] == "test"
assert cfg["strict_workload_split"] is True
assert cfg["common_random_scenarios"] is True
assert close(cfg["q_fixed_opf"], 1.0, 1e-12)
assert 1.0 not in [float(x) for x in cfg["opf_set"]]
assert cfg.get("_run_metadata", {}).get("workload_sampling_used_full_profile_fallback") is False

seed_counts = summary.groupby("policy")["seed"].nunique()
assert len(summary) == 390, f"Expected 390 policy-run rows, found {len(summary)}"
assert (seed_counts == 30).all(), f"Expected 30 seeds per policy:\n{seed_counts}"

expected = {
    "static_opf_1.75_forecast": (-213.669347, 0.200140, 0.003009),
    "static_opf_2.00_forecast": (-304.220418, 0.218265, 0.011458),
    "opf_aware_q_learning": (-308.113561, 0.207392, 0.036690),
    "conservative": (-818.407339, 0.124937, 0.0),
    "q_learning": (-1228.892031, 0.115558, 0.0),
}
for policy, (reward, admission, sla) in expected.items():
    row = stats.loc[policy]
    assert close(row["net_profit_mean"], reward)
    assert close(row["admission_ratio_mean"], admission)
    assert close(row["sla_violation_rate_mean"], sla)

feasible = ranking[ranking["sla_feasible"] == True]
assert feasible.iloc[0]["policy"] == "static_opf_1.75_forecast"
assert close(feasible.iloc[0]["net_profit"], -213.669347)

imp = improvement.loc["static_opf_1.75_forecast"]
assert close(imp["net_profit_mean_improvement"], 604.737992)
assert close(imp["net_profit_relative_improvement_pct"], 73.892054)

mask = (
    ((paired["policy_a"] == "conservative") & (paired["policy_b"] == "static_opf_1.75_forecast")) |
    ((paired["policy_b"] == "conservative") & (paired["policy_a"] == "static_opf_1.75_forecast"))
)
row = paired.loc[mask].iloc[0]
assert close(row["paired_p_value"], 7.783171e-10, 1e-12)
assert close(row["wilcoxon_p_value"], 9.313226e-09, 1e-12)

print("Reference result verification passed.")
print(f"Policies: {seed_counts.size}")
print("Paired seeds per policy: 30")
print("Best policy at SLA violation rate <= 2%: Static OPF 1.75")
print("Mean reward: -213.67")
print("Admission ratio: 20.01%")
print("SLA violation rate: 0.30%")
