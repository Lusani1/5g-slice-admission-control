"""Limited one-factor-at-a-time sensitivity analysis utilities.

This module is designed for the journal robustness experiment. It reuses the
existing simulator, policy implementations, common-random-scenario logic, and
paired Monte Carlo seeds. Only one modelling assumption is varied at a time.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .config import SimulatorConfig
from .policies import (
    ConservativePolicy,
    StaticOPFPolicy,
    QLearningPolicy,
    OPFAwareQLearningPolicy,
)


POLICY_DISPLAY = {
    "conservative": "Conservative",
    "static_opf_2.00_forecast": "Static OPF 2.00",
    "q_learning": "Q-learning",
    "opf_aware_q_learning": "OPF-aware Q-learning",
}

PARAMETER_DISPLAY = {
    "sla_penalty": "SLA penalty multiplier",
    "overbooking_penalty": "Overbooking penalty multiplier",
    "demand": "Demand multiplier",
    "capacity": "Capacity",
    "vm_grouping": "VM grouping",
    "scaling": "Resource scaling vector (zeta)",
}

PARAMETER_UNITS = {
    "sla_penalty": r"$\times$",
    "overbooking_penalty": r"$\times$",
    "demand": r"$\times$",
    "capacity": "units",
    "vm_grouping": "VMs/profile",
    "scaling": "zeta = [cpu, mem, bw]",
}

BASELINE_LEVELS = {
    "sla_penalty": 1.0,
    "overbooking_penalty": 1.0,
    "demand": 1.0,
    "capacity": 100.0,
    "vm_grouping": 10.0,
    # NOTE: "scaling" levels are ordinal indices into --scaling-variants (not a
    # physical unit, since zeta is a 3-vector). 1.0 assumes the default
    # ordering "zeta_5_1_5,zeta_10_1_10,zeta_20_1_20", where index 1 is the
    # manuscript's zeta=[10,1,10] baseline. If --scaling-variants is
    # reordered, update this to match the new index of the zeta=[10,1,10]
    # variant.
    "scaling": 1.0,
}

METRICS = (
    "net_profit",
    "admission_ratio",
    "sla_violation_rate",
    "mean_cpu_utilisation",
    "mean_mem_utilisation",
    "mean_bw_utilisation",
)


def make_limited_policies(cfg: SimulatorConfig, include_opf_aware: bool = False, include_high_opf: bool = False):
    """Create the policy subset used by optional robustness experiments."""

    policies = [
        QLearningPolicy(
            fixed_opf=cfg.q_fixed_opf,
            learning_rate=cfg.learning_rate,
            discount_factor=cfg.discount_factor,
            epsilon_initial=cfg.epsilon_initial,
            epsilon_min=cfg.epsilon_min,
            epsilon_decay=cfg.epsilon_decay,
            min_visits_for_greedy=cfg.min_visits_for_greedy,
        )
    ]
    if include_opf_aware:
        policies.append(
            OPFAwareQLearningPolicy(
                fixed_opf=cfg.q_fixed_opf,
                learning_rate=cfg.learning_rate,
                discount_factor=cfg.discount_factor,
                epsilon_initial=cfg.epsilon_initial,
                epsilon_min=cfg.epsilon_min,
                epsilon_decay=cfg.epsilon_decay,
                min_visits_for_greedy=cfg.min_visits_for_greedy,
                opf_set=cfg.opf_set,
            )
        )
    policies.extend([
        ConservativePolicy(),
        StaticOPFPolicy(opf=2.00, use_forecast=True),
    ])
    if include_high_opf:
        policies.append(StaticOPFPolicy(opf=2.25, use_forecast=True))
    return policies


def t_confidence_interval(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    mean = float(np.mean(values))
    sem = float(stats.sem(values))
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, len(values) - 1))
    return mean - critical * sem, mean + critical * sem


def paired_cohens_dz(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) < 2:
        return np.nan
    sd = float(np.std(differences, ddof=1))
    if sd <= 1e-12:
        return np.nan
    return float(np.mean(differences) / sd)


def holm_adjust(p_values: Iterable[float]) -> list[float]:
    p = np.asarray(list(p_values), dtype=float)
    adjusted = np.full_like(p, np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    if len(finite) == 0:
        return adjusted.tolist()
    ordered = finite[np.argsort(p[finite])]
    running = 0.0
    m = len(ordered)
    for rank, idx in enumerate(ordered):
        candidate = min(1.0, (m - rank) * float(p[idx]))
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted.tolist()


def summarise_sensitivity(raw: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean, SD, SE and t-based 95% CI per setting and policy."""
    required = {"parameter", "level_value", "level_label", "policy", "seed"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Sensitivity data are missing columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    group_cols = ["parameter", "level_value", "level_label", "policy"]
    for key, group in raw.groupby(group_cols, sort=False):
        parameter, level_value, level_label, policy = key
        row: dict[str, object] = {
            "parameter": parameter,
            "parameter_display": PARAMETER_DISPLAY.get(parameter, parameter),
            "level_value": float(level_value),
            "level_label": str(level_label),
            "policy": policy,
            "policy_display": POLICY_DISPLAY.get(policy, policy),
            "n_runs": int(group["seed"].nunique()),
        }
        for metric in METRICS:
            if metric not in group.columns:
                continue
            values = group[metric].astype(float).to_numpy()
            low, high = t_confidence_interval(values)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            row[f"{metric}_se"] = float(stats.sem(values)) if len(values) > 1 else 0.0
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
        rows.append(row)
    return pd.DataFrame(rows)


def paired_level_tests(raw: pd.DataFrame) -> pd.DataFrame:
    """Compare each non-baseline level with its baseline using matched seeds."""
    rows: list[dict[str, object]] = []
    for parameter, parameter_frame in raw.groupby("parameter", sort=False):
        baseline_level = BASELINE_LEVELS[parameter]
        for policy, policy_frame in parameter_frame.groupby("policy", sort=False):
            baseline = policy_frame[
                np.isclose(policy_frame["level_value"].astype(float), baseline_level)
            ]
            if baseline.empty:
                continue
            for level in sorted(policy_frame["level_value"].astype(float).unique()):
                if math.isclose(level, baseline_level):
                    continue
                comparison = policy_frame[np.isclose(policy_frame["level_value"].astype(float), level)]
                for metric in ("net_profit", "admission_ratio", "sla_violation_rate"):
                    merged = (
                        baseline[["seed", metric]]
                        .rename(columns={metric: "baseline_value"})
                        .merge(
                            comparison[["seed", metric]].rename(columns={metric: "comparison_value"}),
                            on="seed",
                            how="inner",
                        )
                        .sort_values("seed")
                    )
                    if len(merged) < 2:
                        continue
                    difference = (
                        merged["comparison_value"].astype(float).to_numpy()
                        - merged["baseline_value"].astype(float).to_numpy()
                    )
                    t_result = stats.ttest_rel(
                        merged["comparison_value"],
                        merged["baseline_value"],
                        nan_policy="omit",
                    )
                    try:
                        if np.allclose(difference, 0.0):
                            wilcoxon_stat, wilcoxon_p = 0.0, 1.0
                        else:
                            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                                difference,
                                zero_method="wilcox",
                                alternative="two-sided",
                            )
                    except Exception:
                        wilcoxon_stat, wilcoxon_p = np.nan, np.nan
                    ci_low, ci_high = t_confidence_interval(difference)
                    rows.append({
                        "parameter": parameter,
                        "parameter_display": PARAMETER_DISPLAY.get(parameter, parameter),
                        "policy": policy,
                        "policy_display": POLICY_DISPLAY.get(policy, policy),
                        "metric": metric,
                        "baseline_level": baseline_level,
                        "comparison_level": float(level),
                        "n_paired_seeds": int(len(merged)),
                        "baseline_mean": float(merged["baseline_value"].mean()),
                        "comparison_mean": float(merged["comparison_value"].mean()),
                        "mean_difference_comparison_minus_baseline": float(np.mean(difference)),
                        "difference_ci95_low": ci_low,
                        "difference_ci95_high": ci_high,
                        "paired_t_statistic": float(t_result.statistic),
                        "paired_t_p_value": float(t_result.pvalue),
                        "wilcoxon_statistic": float(wilcoxon_stat),
                        "wilcoxon_p_value": float(wilcoxon_p),
                        "paired_cohens_dz": paired_cohens_dz(difference),
                    })
    result = pd.DataFrame(rows)
    if not result.empty:
        result["paired_t_p_holm"] = holm_adjust(result["paired_t_p_value"])
        result["wilcoxon_p_holm"] = holm_adjust(result["wilcoxon_p_value"])
    return result


def best_sla_target_feasible(summary: pd.DataFrame, sla_target: float) -> pd.DataFrame:
    """Select the highest-reward SLA-target-feasible policy per setting."""
    rows: list[pd.Series] = []
    group_cols = ["parameter", "parameter_display", "level_value", "level_label"]
    for _, group in summary.groupby(group_cols, sort=False):
        feasible = group[group["sla_violation_rate_mean"] <= sla_target].copy()
        if feasible.empty:
            selected = group.sort_values(
                ["sla_violation_rate_mean", "net_profit_mean"],
                ascending=[True, False],
            ).iloc[0].copy()
            selected["sla_target_feasible"] = False
        else:
            selected = feasible.sort_values("net_profit_mean", ascending=False).iloc[0].copy()
            selected["sla_target_feasible"] = True
        rows.append(selected)
    result = pd.DataFrame(rows)
    return result.sort_values(["parameter", "level_value"])


def _ordered_parameter_levels(parameter: str, frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("level_value")


def make_sensitivity_plots(summary: pd.DataFrame, output_dir: str | Path, sla_target: float) -> None:
    """Create separate journal-ready reward, admission and SLA plots per parameter."""
    output_dir = Path(output_dir)
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for parameter, parameter_frame in summary.groupby("parameter", sort=False):
        parameter_frame = _ordered_parameter_levels(parameter, parameter_frame)
        x_values = sorted(parameter_frame["level_value"].unique())
        if parameter == "scaling":
            label_lookup = (
                parameter_frame.drop_duplicates("level_value").set_index("level_value")["level_label"]
            )
            x_labels = [str(label_lookup.get(x, f"{x:g}")) for x in x_values]
        else:
            x_labels = [
                f"{x:g}" if parameter not in {"sla_penalty", "overbooking_penalty", "demand"} else f"{x:g}×"
                for x in x_values
            ]

        plot_specs = [
            ("net_profit", "Mean net reward", f"sensitivity_{parameter}_reward"),
            ("admission_ratio", "Admission ratio", f"sensitivity_{parameter}_admission"),
            ("sla_violation_rate", "SLA violation rate", f"sensitivity_{parameter}_sla"),
        ]
        for metric, ylabel, filename in plot_specs:
            plt.figure(figsize=(7.0, 4.5))
            for policy, policy_frame in parameter_frame.groupby("policy", sort=False):
                policy_frame = policy_frame.sort_values("level_value")
                means = policy_frame[f"{metric}_mean"].to_numpy(float)
                lower = means - policy_frame[f"{metric}_ci95_low"].to_numpy(float)
                upper = policy_frame[f"{metric}_ci95_high"].to_numpy(float) - means
                plt.errorbar(
                    policy_frame["level_value"].to_numpy(float),
                    means,
                    yerr=np.vstack([lower, upper]),
                    marker="o",
                    capsize=3,
                    label=POLICY_DISPLAY.get(policy, policy),
                )
            if metric == "sla_violation_rate":
                plt.axhline(sla_target, linestyle="--", label=f"SLA target ({100*sla_target:.0f}%)")
            plt.xticks(x_values, x_labels)
            plt.xlabel(PARAMETER_DISPLAY.get(parameter, parameter))
            plt.ylabel(ylabel)
            plt.grid(axis="y", alpha=0.25)
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(plot_dir / f"{filename}.png", dpi=300, bbox_inches="tight")
            plt.savefig(plot_dir / f"{filename}.pdf", bbox_inches="tight")
            plt.close()


def _tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "_": r"\_",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def write_latex_tables(
    summary: pd.DataFrame,
    best: pd.DataFrame,
    output_dir: str | Path,
    sla_target: float,
) -> None:
    output_dir = Path(output_dir)

    compact = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Policy robustness under selected parameter sensitivities.}",
        r"\label{tab:limited_parameter_sensitivity}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\renewcommand{\arraystretch}{1.10}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\textbf{Parameter} & \textbf{Level} & \textbf{Best SLA-target-feasible policy} & \textbf{Mean reward} & \textbf{Admission} & \textbf{SLA viol.} \\",
        r"\midrule",
    ]
    previous_parameter = None
    for _, row in best.iterrows():
        parameter = str(row["parameter"])
        if previous_parameter is not None and parameter != previous_parameter:
            compact.append(r"\midrule")
        feasible_label = row["policy_display"] if bool(row["sla_target_feasible"]) else "No policy met target; lowest-SLA shown"
        compact.append(
            f"{_tex_escape(row['parameter_display'])} & {_tex_escape(row['level_label'])} & "
            f"{_tex_escape(feasible_label)} & {row['net_profit_mean']:.2f} & "
            f"{100*row['admission_ratio_mean']:.2f}\\% & {100*row['sla_violation_rate_mean']:.2f}\\% \\\\" 
        )
        previous_parameter = parameter
    compact.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (output_dir / "limited_sensitivity_best_policy_table.tex").write_text("\n".join(compact), encoding="utf-8")

    full = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Complete limited parameter-sensitivity results.}",
        r"\label{tab:limited_parameter_sensitivity_full}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lllrrrr}",
        r"\toprule",
        r"\textbf{Parameter} & \textbf{Level} & \textbf{Policy} & \textbf{Mean reward} & \textbf{Admission} & \textbf{SLA viol.} & \textbf{SLA feasible} \\",
        r"\midrule",
    ]
    ordered = summary.sort_values(["parameter", "level_value", "net_profit_mean"], ascending=[True, True, False])
    for _, row in ordered.iterrows():
        feasible = "Yes" if row["sla_violation_rate_mean"] <= sla_target else "No"
        full.append(
            f"{_tex_escape(row['parameter_display'])} & {_tex_escape(row['level_label'])} & "
            f"{_tex_escape(row['policy_display'])} & {row['net_profit_mean']:.2f} & "
            f"{100*row['admission_ratio_mean']:.2f}\\% & {100*row['sla_violation_rate_mean']:.2f}\\% & {feasible} \\\\" 
        )
    full.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (output_dir / "limited_sensitivity_full_table.tex").write_text("\n".join(full), encoding="utf-8")


def write_methodology_snippet(cfg: SimulatorConfig, output_dir: str | Path, included_parameters: Iterable[str]) -> None:
    output_dir = Path(output_dir)
    parameters = set(included_parameters)
    parts = [
        r"\subsection{Parameter Sensitivity and Robustness Analysis}",
        r"\label{subsec:parameter_sensitivity_robustness}",
        "A one-factor-at-a-time sensitivity analysis was performed to determine whether the main policy conclusions depended on selected simulator assumptions. For every sensitivity setting, all parameters other than the factor under investigation were held at their baseline values. The same Monte Carlo seeds were used across parameter levels, preserving the arrival times, slice classes, lifetimes, revenue samples, workload-profile selections, burst periods, and degradation events wherever the underlying profile bank was unchanged.",
    ]
    descriptions = []
    if "sla_penalty" in parameters:
        descriptions.append(r"the SLA penalty vector was multiplied by $0.5$, $1.0$, and $2.0$")
    if "overbooking_penalty" in parameters:
        descriptions.append(r"the overbooking penalty vector was multiplied by $0.5$, $1.0$, and $2.0$")
    if "demand" in parameters:
        descriptions.append(r"all sampled slice-demand vectors were multiplied by $0.8$, $1.0$, and $1.2$")
    if "capacity" in parameters:
        descriptions.append(r"the normalised CPU, memory, and bandwidth capacities were set jointly to 80, 100, and 120 units")
    if "vm_grouping" in parameters:
        descriptions.append(r"the Materna-derived profile construction was repeated using 5, 10, and 15 VMs per profile, with XGBoost retrained separately for each profile bank")
    if "scaling" in parameters:
        descriptions.append(r"the CPU/bandwidth scaling vector $\zeta$ was varied while memory was left unscaled, with the full preprocessing pipeline (IQR capping, quantile normalisation, demand-regime thresholds) and the XGBoost forecaster recomputed separately for each $\zeta$")
    if descriptions:
        if len(descriptions) == 1:
            variation_sentence = descriptions[0]
        else:
            variation_sentence = ", ".join(descriptions[:-1]) + ", and " + descriptions[-1]
        parts.append("The evaluated variations were as follows: " + variation_sentence + ".")
    parts.append(
        f"The conservative baseline, Static OPF 2.00, and Q-learning policies were evaluated over {cfg.monte_carlo_runs} matched Monte Carlo runs per setting. Mean net reward, admission ratio, SLA violation rate, resource utilisation, and 95\\% confidence intervals were reported. Low and high parameter levels were compared with the baseline using paired $t$-tests and Wilcoxon signed-rank tests over common seeds."
    )
    (output_dir / "limited_sensitivity_methodology.tex").write_text("\n\n".join(parts), encoding="utf-8")


def write_analysis_outputs(
    raw: pd.DataFrame,
    output_dir: str | Path,
    sla_target: float,
    cfg: SimulatorConfig,
) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(output_dir / "limited_sensitivity_raw.csv", index=False)
    summary = summarise_sensitivity(raw)
    summary.to_csv(output_dir / "limited_sensitivity_summary_ci.csv", index=False)
    tests = paired_level_tests(raw)
    tests.to_csv(output_dir / "limited_sensitivity_paired_tests.csv", index=False)
    best = best_sla_target_feasible(summary, sla_target)
    best.to_csv(output_dir / "limited_sensitivity_best_sla_feasible.csv", index=False)
    make_sensitivity_plots(summary, output_dir, sla_target)
    write_latex_tables(summary, best, output_dir, sla_target)
    write_methodology_snippet(cfg, output_dir, raw["parameter"].unique())
    return {"raw": raw, "summary": summary, "tests": tests, "best": best}
