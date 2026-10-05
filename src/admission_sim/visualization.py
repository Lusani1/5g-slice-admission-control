"""Journal-style plotting for admission-control simulator outputs."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.png", dpi=600, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _short_label(policy: str) -> str:
    mapping = {
        "conservative": "Conservative",
        "adaptive_heuristic": "Adaptive",
        "q_learning": "Q-learning",
        "opf_aware_q_learning": "OPF-aware QL",
    }
    if policy in mapping:
        return mapping[policy]
    if policy.startswith("static_opf_"):
        return "Static OPF " + policy.split("_")[2]
    return policy


def make_all_plots(results_dir: str | Path, out_dir: str | Path | None = None) -> None:
    results_dir = Path(results_dir)
    out_dir = Path(out_dir) if out_dir is not None else results_dir
    summary = pd.read_csv(results_dir / "summary_raw.csv")
    ts = pd.read_csv(results_dir / "timeseries_raw.csv")
    stats = pd.read_csv(results_dir / "summary_statistics_ci.csv")

    policy_order = stats.sort_values("net_profit_mean", ascending=False)["policy"].tolist()
    labels = [_short_label(p) for p in policy_order]

    # Boxplot: net profit
    fig, ax = plt.subplots(figsize=(10, 5.2))
    data = [summary.loc[summary.policy == p, "net_profit"].astype(float).to_numpy() for p in policy_order]
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("Net reward")
    ax.set_xlabel("Admission-control policy")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=25)
    _save(fig, out_dir, "fig_boxplot_net_profit")

    # Boxplot: SLA violation rate
    fig, ax = plt.subplots(figsize=(10, 5.2))
    data = [summary.loc[summary.policy == p, "sla_violation_rate"].astype(float).to_numpy() for p in policy_order]
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.axhline(0.02, linestyle="--", linewidth=1, label="SLA target")
    ax.set_ylabel("SLA violation rate")
    ax.set_xlabel("Admission-control policy")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=25)
    _save(fig, out_dir, "fig_boxplot_sla_violation_rate")

    # Boxplot: admission ratio
    fig, ax = plt.subplots(figsize=(10, 5.2))
    data = [summary.loc[summary.policy == p, "admission_ratio"].astype(float).to_numpy() for p in policy_order]
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("Admission ratio")
    ax.set_xlabel("Admission-control policy")
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=25)
    _save(fig, out_dir, "fig_boxplot_admission_ratio")

    improvement_path = results_dir / "policy_improvement_vs_conservative.csv"
    if improvement_path.exists():
        imp = pd.read_csv(improvement_path)
        if len(imp) and "net_profit_mean_improvement" in imp.columns:
            imp = imp.sort_values("net_profit_mean_improvement", ascending=False)
            fig, ax = plt.subplots(figsize=(10, 5.2))
            ax.bar([_short_label(p) for p in imp["policy"]], imp["net_profit_mean_improvement"].astype(float))
            ax.axhline(0.0, linewidth=1)
            ax.set_ylabel("Net reward improvement vs conservative")
            ax.set_xlabel("Admission-control policy")
            ax.grid(axis="y", alpha=0.3)
            ax.tick_params(axis="x", rotation=25)
            _save(fig, out_dir, "fig_improvement_vs_conservative_net_reward")

    # Revenue-reliability trade-off scatter
    means = summary.groupby("policy", as_index=False).agg(
        net_profit=("net_profit", "mean"),
        sla_violation_rate=("sla_violation_rate", "mean"),
        admission_ratio=("admission_ratio", "mean"),
    ).sort_values("net_profit", ascending=False)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(means["sla_violation_rate"], means["net_profit"], s=70)
    offsets = [(6, 6), (6, -12), (6, 14), (6, -18), (6, 20), (6, -24), (6, 26), (6, -30), (6, 32)]
    for i, (_, row) in enumerate(means.iterrows()):
        ax.annotate(_short_label(row["policy"]), (row["sla_violation_rate"], row["net_profit"]),
                    xytext=offsets[i % len(offsets)], textcoords="offset points", fontsize=8)
    ax.axvline(0.02, linestyle="--", linewidth=1, label="SLA target")
    ax.set_xlabel("SLA violation rate")
    ax.set_ylabel("Net reward")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    _save(fig, out_dir, "fig_revenue_reliability_tradeoff")

    # OPF sensitivity curves for static OPF policies only
    static = means[means["policy"].str.contains("static_opf")].copy()
    if len(static):
        static["opf"] = static["policy"].str.extract(r"static_opf_(\d+\.\d+)").astype(float)
        static = static.sort_values("opf")
        for metric, ylabel, name in [
            ("net_profit", "Net reward", "fig_opf_net_profit_curve"),
            ("sla_violation_rate", "SLA violation rate", "fig_opf_sla_violation_rate_curve"),
            ("admission_ratio", "Admission ratio", "fig_opf_admission_ratio_curve"),
        ]:
            fig, ax = plt.subplots(figsize=(6.8, 4.6))
            ax.plot(static["opf"], static[metric], marker="o")
            if metric == "sla_violation_rate":
                ax.axhline(0.02, linestyle="--", linewidth=1, label="SLA target")
                ax.legend(frameon=False)
            ax.set_xlabel("Static OPF")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            _save(fig, out_dir, name)

    # Representative time series for best SLA-feasible policy by mean net profit
    if len(policy_order):
        means_rank = summary.groupby("policy", as_index=False).agg(
            net_profit=("net_profit", "mean"),
            sla_violation_rate=("sla_violation_rate", "mean"),
        )
        feasible = means_rank[means_rank["sla_violation_rate"] <= 0.02]
        best = (feasible if len(feasible) else means_rank).sort_values("net_profit", ascending=False)["policy"].iloc[0]
        seed = summary.loc[summary.policy == best].sort_values("net_profit", ascending=False)["seed"].iloc[0]
        sel = ts[(ts.policy == best) & (ts.seed == seed)].copy()
        if len(sel):
            fig, ax = plt.subplots(figsize=(9.5, 5))
            ax.plot(sel["t"], sel["actual_cpu"] / sel["capacity_cpu"], label="CPU utilisation")
            ax.plot(sel["t"], sel["actual_mem"] / sel["capacity_mem"], label="Memory utilisation")
            ax.plot(sel["t"], sel["actual_bw"] / sel["capacity_bw"], label="Bandwidth utilisation")
            ax.axhline(1.0, linestyle="--", linewidth=1, label="Capacity")
            ax.set_xlabel("Decision epoch")
            ax.set_ylabel("Utilisation ratio")
            ax.legend(frameon=False, loc="upper right")
            ax.grid(True, alpha=0.3)
            _save(fig, out_dir, "fig_timeseries_utilisation_best_policy")

            fig, ax = plt.subplots(figsize=(9.5, 5))
            ax.plot(sel["t"], sel["selected_opf"])
            ax.set_xlabel("Decision epoch")
            ax.set_ylabel("Selected OPF")
            ax.grid(True, alpha=0.3)
            _save(fig, out_dir, "fig_timeseries_selected_opf_best_policy")

            fig, ax = plt.subplots(figsize=(9.5, 5))
            ax.plot(sel["t"], sel["net_profit"].cumsum())
            ax.set_xlabel("Decision epoch")
            ax.set_ylabel("Cumulative net reward")
            ax.grid(True, alpha=0.3)
            _save(fig, out_dir, "fig_timeseries_cumulative_profit_best_policy")
