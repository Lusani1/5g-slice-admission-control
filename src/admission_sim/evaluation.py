"""Monte Carlo evaluation and result summarisation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable
import copy
import numpy as np
import pandas as pd
from scipy import stats

from .config import SimulatorConfig
from .data import MaternaProfileStore
from .environment import AdmissionControlSimulator, Scenario
from .forecaster import BaseForecaster, PersistenceForecaster
from .policies import make_default_policies, BasePolicy


def train_policy_if_needed(policy: BasePolicy, cfg: SimulatorConfig, profile_store: MaternaProfileStore, forecaster: BaseForecaster, seed: int) -> BasePolicy:
    """Train Q-learning policies; deterministic policies are returned unchanged.

    The training forecaster is selected by ``cfg.rl_train_forecast_mode`` and
    recorded on the policy object for run metadata. The reference experiment
    uses XGBoost during both policy training and evaluation.
    """
    if not hasattr(policy, "learn"):
        return policy
    train_cfg = copy.deepcopy(cfg)
    train_mode = cfg.rl_train_forecast_mode.lower()
    if train_cfg.forecast_mode.lower() == "xgboost" and train_mode == "persistence":
        train_cfg.forecast_mode = "persistence"
        train_forecaster: BaseForecaster = PersistenceForecaster()
    else:
        train_forecaster = forecaster
    policy.trained_with_forecast_mode = train_cfg.forecast_mode.lower()
    sim = AdmissionControlSimulator(cfg=train_cfg, profile_store=profile_store, forecaster=train_forecaster, seed=seed)
    for ep in range(cfg.training_episodes):
        train_scenario = sim.make_scenario(seed + 10_000 + ep)
        sim.run_episode(policy, seed=seed + 10_000 + ep, training=True, scenario=train_scenario)
    if hasattr(policy, "epsilon"):
        policy.epsilon = 0.0
    return policy


def _make_common_scenarios(cfg: SimulatorConfig, profile_store: MaternaProfileStore, forecaster: BaseForecaster, seeds: list[int]) -> dict[int, Scenario]:
    sim = AdmissionControlSimulator(cfg=cfg, profile_store=profile_store, forecaster=forecaster, seed=cfg.base_seed)
    return {seed: sim.make_scenario(seed) for seed in seeds}


def _write_run_config(cfg: SimulatorConfig, output_dir: Path, extra_metadata: dict | None = None) -> None:
    """Write the simulator configuration and reproducibility metadata.

    The metadata records the execution timestamp, git commit when available,
    command line, training-forecaster choice, and workload split.
    """
    import json
    import subprocess
    import sys
    import datetime

    serialisable = {}
    for k, v in cfg.__dict__.items():
        if k == "slice_classes":
            serialisable[k] = {name: sc.__dict__ for name, sc in v.items()}
        elif isinstance(v, tuple):
            serialisable[k] = list(v)
        else:
            serialisable[k] = v

    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, cwd=str(Path(__file__).resolve().parent)
        ).decode().strip()
    except Exception:
        git_hash = "unavailable (not a git checkout or git not installed)"

    metadata = {
        "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": git_hash,
        "command": " ".join(sys.argv),
        "rl_train_forecast_mode": cfg.rl_train_forecast_mode,
        "workload_data_split": cfg.workload_data_split,
        "workload_split_column": cfg.workload_split_column,
        "forecast_mode": cfg.forecast_mode,
        "q_fixed_opf": cfg.q_fixed_opf,
        "opf_set_phi": list(cfg.opf_set),
        "static_opfs": list(cfg.static_opfs),
        "capacity": list(cfg.capacity),
        "sla_penalty_weights": list(cfg.sla_penalty_weights),
        "overbooking_penalty_weights": list(cfg.overbooking_penalty_weights),
        "rejection_penalty_lambda": cfg.rejection_penalty_lambda,
        "sla_safe_bonus": cfg.sla_safe_bonus,
        "demand_pressure_multiplier": cfg.demand_pressure_multiplier,
        "degradation_probability": cfg.degradation_probability,
        "degradation_factor_range": list(cfg.degradation_factor_range),
        "degradation_duration_range": list(cfg.degradation_duration_range),
        "monte_carlo_runs": cfg.monte_carlo_runs,
        "training_episodes": cfg.training_episodes,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    serialisable["_run_metadata"] = metadata
    (output_dir / "run_config.json").write_text(json.dumps(serialisable, indent=2, default=str))


def run_policy_suite(
    cfg: SimulatorConfig,
    profile_store: MaternaProfileStore,
    forecaster: BaseForecaster,
    output_dir: str | Path,
    policies: Iterable[BasePolicy] | None = None,
    seeds: Iterable[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full Monte Carlo policy suite.

    If cfg.common_random_scenarios is True, every policy sees the same scenario
    for a given seed. This reduces variance and makes policy comparisons fair.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    policies = list(policies) if policies is not None else make_default_policies(cfg)
    seeds = list(seeds) if seeds is not None else [cfg.base_seed + i for i in range(cfg.monte_carlo_runs)]
    scenarios = _make_common_scenarios(cfg, profile_store, forecaster, seeds) if cfg.common_random_scenarios else {}

    summary_rows = []
    ts_frames = []
    decision_frames = []
    rl_train_modes: dict[str, str] = {}
    used_full_profile_fallback = False

    for policy_template in policies:
        # Train learning policies once per Monte Carlo experiment, then replay
        # the learned greedy policy over every common-random evaluation seed.
        # This avoids re-training per seed and makes the runtime practical.
        # Train/evaluate the policy object in place. Policies are created fresh
        # for a run; avoiding deepcopy prevents slow copying of dataclass objects
        # and learned Q-tables in long experiments.
        trained_template = train_policy_if_needed(policy_template, cfg, profile_store, forecaster, cfg.base_seed)
        if hasattr(trained_template, "trained_with_forecast_mode"):
            rl_train_modes[trained_template.name] = trained_template.trained_with_forecast_mode
        for seed in seeds:
            # Evaluation does not update policy parameters; run_episode resets
            # adaptive internal state at the start of each evaluation. Reusing
            # the trained object avoids expensive deep-copy operations.
            policy = trained_template
            sim = AdmissionControlSimulator(cfg=cfg, profile_store=profile_store, forecaster=forecaster, seed=seed)
            scenario = scenarios.get(seed)
            ts, decisions, summary = sim.run_episode(policy, seed=seed, training=False, scenario=scenario)
            if getattr(sim, "_last_scenario_used_full_profile_fallback", False):
                used_full_profile_fallback = True
            summary_rows.append(summary)
            ts_frames.append(ts.assign(policy=policy.name, seed=seed))
            if len(decisions):
                decision_frames.append(decisions.assign(policy=policy.name, seed=seed))
            print(
                f"[{policy.name}] seed={seed} "
                f"net_profit={summary['net_profit']:.2f} "
                f"sla={summary['sla_violation_rate']:.3f} "
                f"admission={summary['admission_ratio']:.3f} "
                f"req={summary['total_requests']}"
            )

    _write_run_config(cfg, output_dir, extra_metadata={
        "rl_train_forecast_mode_by_policy": rl_train_modes,
        "workload_sampling_used_full_profile_fallback": used_full_profile_fallback,
    })

    summary_df = pd.DataFrame(summary_rows)
    timeseries_df = pd.concat(ts_frames, ignore_index=True) if ts_frames else pd.DataFrame()
    decisions_df = pd.concat(decision_frames, ignore_index=True) if decision_frames else pd.DataFrame()

    summary_df.to_csv(output_dir / "summary_raw.csv", index=False)
    timeseries_df.to_csv(output_dir / "timeseries_raw.csv", index=False)
    decisions_df.to_csv(output_dir / "decisions_raw.csv", index=False)

    stats_df = summarise_results(summary_df)
    stats_df.to_csv(output_dir / "summary_statistics_ci.csv", index=False)
    pareto = pareto_frontier(summary_df)
    pareto.to_csv(output_dir / "pareto_frontier.csv", index=False)
    ranking = sla_constrained_ranking(summary_df, cfg.sla_violation_target)
    ranking.to_csv(output_dir / "sla_constrained_policy_ranking.csv", index=False)
    # Welch tests are retained for compatibility, but common-random scenarios
    # make paired tests the primary statistical comparison.
    write_pairwise_tests(summary_df, output_dir, metric="net_profit")
    write_pairwise_tests(summary_df, output_dir, metric="sla_violation_rate")
    write_paired_tests(summary_df, output_dir, metric="net_profit")
    write_paired_tests(summary_df, output_dir, metric="sla_violation_rate")
    write_improvement_vs_baseline(summary_df, output_dir, baseline="conservative")
    write_diagnostics(summary_df, timeseries_df, decisions_df, output_dir)
    return summary_df, timeseries_df, decisions_df


def summarise_results(summary_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "net_profit",
        "admission_ratio",
        "rejection_ratio",
        "sla_violation_rate",
        "mean_cpu_utilisation",
        "mean_mem_utilisation",
        "mean_bw_utilisation",
        "p95_cpu_utilisation",
        "p95_mem_utilisation",
        "p95_bw_utilisation",
        "mean_reserved_cpu_ratio",
        "mean_reserved_mem_ratio",
        "mean_reserved_bw_ratio",
        "mean_selected_opf",
        "total_admission_revenue",
        "total_sla_penalty",
        "total_rejection_penalty",
        "total_overbooking_penalty",
        "total_safe_bonus",
        "blocked_nominal_count",
        "blocked_forecast_count",
        "blocked_both_count",
        "policy_reject_count",
        "total_requests",
    ]
    rows = []
    for policy, grp in summary_df.groupby("policy"):
        row = {"policy": policy, "n_runs": int(len(grp))}
        for m in metrics:
            if m not in grp.columns:
                continue
            x = grp[m].astype(float).to_numpy()
            mean = float(np.mean(x))
            std = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
            se = float(std / np.sqrt(len(x))) if len(x) > 1 else 0.0
            # Use the Student-t critical value for the
            # observed n rather than the large-sample z-value 1.96. For the
            # manuscript's n=30 seeds (df=29) this is t~=2.045 vs 1.96, a ~4.3%
            # wider interval; the difference is small but the t-value is the
            # statistically correct one for finite n and is used unconditionally.
            t_crit = float(stats.t.ppf(0.975, len(x) - 1)) if len(x) > 1 else 0.0
            ci = float(t_crit * se)
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_se"] = se
            row[f"{m}_ci95_low"] = mean - ci
            row[f"{m}_ci95_high"] = mean + ci
        rows.append(row)
    return pd.DataFrame(rows).sort_values("net_profit_mean", ascending=False)


def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) == 0 or len(y) == 0:
        return np.nan
    gt = 0
    lt = 0
    for xi in x:
        gt += np.sum(xi > y)
        lt += np.sum(xi < y)
    return float((gt - lt) / (len(x) * len(y)))


def write_pairwise_tests(summary_df: pd.DataFrame, output_dir: Path, metric: str) -> pd.DataFrame:
    rows = []
    policies = sorted(summary_df["policy"].unique())
    for i, p1 in enumerate(policies):
        for p2 in policies[i + 1:]:
            x = summary_df.loc[summary_df.policy == p1, metric].astype(float).to_numpy()
            y = summary_df.loc[summary_df.policy == p2, metric].astype(float).to_numpy()
            if len(x) < 2 or len(y) < 2:
                t_stat, p_val = np.nan, np.nan
            else:
                t_stat, p_val = stats.ttest_ind(x, y, equal_var=False)
            rows.append({
                "metric": metric,
                "policy_a": p1,
                "policy_b": p2,
                "mean_a": float(np.mean(x)) if len(x) else np.nan,
                "mean_b": float(np.mean(y)) if len(y) else np.nan,
                "welch_t": float(t_stat) if np.isfinite(t_stat) else np.nan,
                "p_value": float(p_val) if np.isfinite(p_val) else np.nan,
                "cliffs_delta": cliffs_delta(x, y),
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"significance_{metric}.csv", index=False)
    return df


def paired_effect_size(diff: np.ndarray) -> float:
    """Paired Cohen's dz for common-random-seed experiments."""
    diff = np.asarray(diff, dtype=float)
    if len(diff) < 2:
        return np.nan
    sd = float(np.std(diff, ddof=1))
    if sd <= 1e-12:
        return np.nan
    return float(np.mean(diff) / sd)


def write_paired_tests(summary_df: pd.DataFrame, output_dir: Path, metric: str) -> pd.DataFrame:
    """Write paired statistical tests by Monte Carlo seed.

    Because policies are evaluated using common random scenarios, each policy's
    result at a given seed should be compared with the other policy at the same
    seed.  This gives a lower-variance and more valid test than Welch's
    independent-sample test.
    """
    rows = []
    policies = sorted(summary_df["policy"].unique())
    for i, p1 in enumerate(policies):
        for p2 in policies[i + 1:]:
            a = summary_df.loc[summary_df.policy == p1, ["seed", metric]].rename(columns={metric: "a"})
            b = summary_df.loc[summary_df.policy == p2, ["seed", metric]].rename(columns={metric: "b"})
            m = pd.merge(a, b, on="seed", how="inner").sort_values("seed")
            x = m["a"].astype(float).to_numpy()
            y = m["b"].astype(float).to_numpy()
            diff = x - y
            if len(diff) < 2:
                t_stat = p_val = w_stat = w_p = np.nan
            else:
                t_stat, p_val = stats.ttest_rel(x, y)
                try:
                    if np.allclose(diff, 0):
                        w_stat, w_p = 0.0, 1.0
                    else:
                        w_stat, w_p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                except Exception:
                    w_stat, w_p = np.nan, np.nan
            rows.append({
                "metric": metric,
                "policy_a": p1,
                "policy_b": p2,
                "n_paired_seeds": int(len(diff)),
                "mean_a": float(np.mean(x)) if len(x) else np.nan,
                "mean_b": float(np.mean(y)) if len(y) else np.nan,
                "mean_difference_a_minus_b": float(np.mean(diff)) if len(diff) else np.nan,
                "paired_t": float(t_stat) if np.isfinite(t_stat) else np.nan,
                "paired_p_value": float(p_val) if np.isfinite(p_val) else np.nan,
                "wilcoxon_statistic": float(w_stat) if np.isfinite(w_stat) else np.nan,
                "wilcoxon_p_value": float(w_p) if np.isfinite(w_p) else np.nan,
                "paired_cohens_dz": paired_effect_size(diff),
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / f"paired_significance_{metric}.csv", index=False)
    return df


def write_improvement_vs_baseline(summary_df: pd.DataFrame, output_dir: Path, baseline: str = "conservative") -> pd.DataFrame:
    """Compute paired improvements against the conservative baseline."""
    if baseline not in set(summary_df["policy"]):
        return pd.DataFrame()
    base = summary_df[summary_df.policy == baseline].set_index("seed")
    rows = []
    metrics = [
        "net_profit",
        "admission_ratio",
        "sla_violation_rate",
        "mean_cpu_utilisation",
        "mean_mem_utilisation",
        "mean_bw_utilisation",
        "total_admission_revenue",
        "total_rejection_penalty",
        "total_overbooking_penalty",
    ]
    for policy, grp in summary_df.groupby("policy"):
        if policy == baseline:
            continue
        g = grp.set_index("seed")
        common = base.index.intersection(g.index)
        row = {"policy": policy, "baseline": baseline, "n_paired_seeds": int(len(common))}
        for m in metrics:
            if m in base.columns and m in g.columns and len(common):
                diff = g.loc[common, m].astype(float).to_numpy() - base.loc[common, m].astype(float).to_numpy()
                row[f"{m}_mean_improvement"] = float(np.mean(diff))
                row[f"{m}_relative_improvement_pct"] = float(100.0 * np.mean(diff) / max(abs(float(base.loc[common, m].mean())), 1e-9))
        rows.append(row)
    df = pd.DataFrame(rows)
    if len(df) and "net_profit_mean_improvement" in df.columns:
        df = df.sort_values("net_profit_mean_improvement", ascending=False)
    df.to_csv(output_dir / "policy_improvement_vs_conservative.csv", index=False)
    return df


def pareto_frontier(summary_df: pd.DataFrame) -> pd.DataFrame:
    means = summary_df.groupby("policy", as_index=False).agg(
        net_profit=("net_profit", "mean"),
        sla_violation_rate=("sla_violation_rate", "mean"),
        admission_ratio=("admission_ratio", "mean"),
        mean_selected_opf=("mean_selected_opf", "mean"),
    )
    dominated = []
    for _, row in means.iterrows():
        other = means[means.policy != row.policy]
        is_dom = np.any(
            (other.net_profit >= row.net_profit) &
            (other.sla_violation_rate <= row.sla_violation_rate) &
            ((other.net_profit > row.net_profit) | (other.sla_violation_rate < row.sla_violation_rate))
        )
        dominated.append(bool(is_dom))
    means["pareto_efficient"] = [not d for d in dominated]
    return means.sort_values(["pareto_efficient", "net_profit"], ascending=[False, False])


def sla_constrained_ranking(summary_df: pd.DataFrame, sla_target: float) -> pd.DataFrame:
    means = summary_df.groupby("policy", as_index=False).agg(
        net_profit=("net_profit", "mean"),
        sla_violation_rate=("sla_violation_rate", "mean"),
        admission_ratio=("admission_ratio", "mean"),
        rejection_ratio=("rejection_ratio", "mean"),
        mean_selected_opf=("mean_selected_opf", "mean"),
    )
    means["sla_feasible"] = means["sla_violation_rate"] <= sla_target
    means = means.sort_values(["sla_feasible", "net_profit"], ascending=[False, False])
    means["rank"] = range(1, len(means) + 1)
    return means


def write_diagnostics(summary_df: pd.DataFrame, timeseries_df: pd.DataFrame, decisions_df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    if len(summary_df):
        common_counts = summary_df.groupby("seed")["total_requests"].nunique().reset_index(name="unique_request_counts")
        common_ok = bool((common_counts["unique_request_counts"] == 1).all())
    else:
        common_ok = False
    for policy, grp in summary_df.groupby("policy"):
        row = {
            "policy": policy,
            "common_random_scenarios_verified": common_ok,
            "mean_total_requests": float(grp["total_requests"].mean()) if "total_requests" in grp else np.nan,
            "mean_net_profit": float(grp["net_profit"].mean()),
            "mean_sla_violation_rate": float(grp["sla_violation_rate"].mean()),
            "mean_admission_ratio": float(grp["admission_ratio"].mean()),
            "mean_blocked_nominal": float(grp.get("blocked_nominal_count", pd.Series([0])).mean()),
            "mean_blocked_forecast": float(grp.get("blocked_forecast_count", pd.Series([0])).mean()),
            "mean_blocked_both": float(grp.get("blocked_both_count", pd.Series([0])).mean()),
        }
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("mean_net_profit", ascending=False)
    df.to_csv(output_dir / "diagnostics_policy_summary.csv", index=False)
    return df
