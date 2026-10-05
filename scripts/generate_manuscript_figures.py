#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

POLICY_LABELS = {
    'conservative': 'Conservative',
    'q_learning': 'Q-learning',
    'opf_aware_q_learning': 'OPF-aware Q-learning',
    'adaptive_heuristic': 'Adaptive heuristic',
}

def label(policy: str) -> str:
    if policy.startswith('static_opf_'):
        val = policy.split('_')[2]
        return f'Static OPF {float(val):.2f}'
    return POLICY_LABELS.get(policy, policy)

def save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results-dir', required=True)
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()
    r = Path(args.results_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    s = pd.read_csv(r/'summary_statistics_ci.csv')
    imp = pd.read_csv(r/'policy_improvement_vs_conservative.csv')
    rank = pd.read_csv(r/'sla_constrained_policy_ranking.csv')

    static = s[s.policy.str.startswith('static_opf_')].copy()
    static['opf'] = static.policy.str.extract(r'static_opf_([0-9.]+)_forecast')[0].astype(float)
    static = static.sort_values('opf')

    # Static OPF mean net reward with exact Student-t 95% CI from verified result CSV.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y = static.net_profit_mean.to_numpy()
    lo = static.net_profit_ci95_low.to_numpy()
    hi = static.net_profit_ci95_high.to_numpy()
    ax.errorbar(static.opf, y, yerr=np.vstack([y-lo, hi-y]), marker='o', capsize=4)
    ax.set_xlabel('Static OPF')
    ax.set_ylabel('Mean net reward')
    ax.grid(True, alpha=0.3)
    save(fig, out/'fig_static_opf_net_reward_ci_journal.png')

    # SLA violation rate in percent.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y = 100*static.sla_violation_rate_mean.to_numpy()
    lo = 100*static.sla_violation_rate_ci95_low.to_numpy()
    hi = 100*static.sla_violation_rate_ci95_high.to_numpy()
    ax.errorbar(static.opf, y, yerr=np.vstack([y-lo, hi-y]), marker='o', capsize=4)
    ax.axhline(2.0, linestyle='--', label='SLA target')
    ax.set_xlabel('Static OPF')
    ax.set_ylabel('SLA violation rate (%)')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    save(fig, out/'fig_static_opf_sla_violation_ci_journal.png')

    # Admission ratio in percent.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    y = 100*static.admission_ratio_mean.to_numpy()
    lo = 100*static.admission_ratio_ci95_low.to_numpy()
    hi = 100*static.admission_ratio_ci95_high.to_numpy()
    ax.errorbar(static.opf, y, yerr=np.vstack([y-lo, hi-y]), marker='o', capsize=4)
    ax.set_xlabel('Static OPF')
    ax.set_ylabel('Admission ratio (%)')
    ax.grid(True, alpha=0.3)
    save(fig, out/'fig_static_opf_admission_ratio_ci_journal.png')

    # Revenue-reliability tradeoff. Plot static sweep as a labelled curve and non-static policies separately.
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    ax.plot(100*static.sla_violation_rate_mean, static.net_profit_mean, marker='o', label='Static OPF sweep')
    for _, row in static.iterrows():
        ax.annotate(f"{row.opf:.2f}", (100*row.sla_violation_rate_mean, row.net_profit_mean),
                    xytext=(4, 4), textcoords='offset points', fontsize=8)
    nonstatic_order = ['conservative','q_learning','adaptive_heuristic','opf_aware_q_learning']
    markers = ['s','^','D','v']
    for p, m in zip(nonstatic_order, markers):
        row = s[s.policy.eq(p)].iloc[0]
        ax.scatter(100*row.sla_violation_rate_mean, row.net_profit_mean, marker=m, s=55, label=label(p))
    ax.axvline(2.0, linestyle='--', label='SLA target')
    ax.set_xlabel('SLA violation rate (%)')
    ax.set_ylabel('Mean net reward')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc='lower left')
    save(fig, out/'pareto_frontier.png')

    # Feasible policy ranking from verified CSV; show top 9 feasible policies in the authoritative ranking.
    feasible = rank[rank.sla_feasible.astype(bool)].copy().sort_values('rank')
    feasible['label'] = feasible.policy.map(label)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.barh(feasible.label[::-1], feasible.net_profit[::-1])
    ax.set_xlabel('Mean net reward')
    ax.set_ylabel('')
    ax.grid(True, axis='x', alpha=0.3)
    save(fig, out/'fig_sla_constrained_policy_ranking_journal.png')

    # Improvement relative to conservative from paired verified results.
    imp = imp.copy()
    imp['label'] = imp.policy.map(label)
    imp = imp.sort_values('net_profit_mean_improvement', ascending=True)
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.barh(imp.label, imp.net_profit_mean_improvement)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel('Mean net reward improvement')
    ax.set_ylabel('')
    ax.grid(True, axis='x', alpha=0.3)
    save(fig, out/'fig_improvement_vs_conservative_net_reward_journal.png')

if __name__ == '__main__':
    main()
