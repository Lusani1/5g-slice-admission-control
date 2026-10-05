# Manuscript evidence index

The table below maps the principal admission-control results reported in the manuscript to the corresponding files in `results/reference_run/`.

| Manuscript evidence | Primary source file | Supporting file/script |
|---|---|---|
| 30 paired policy evaluations | `summary_raw.csv` | `run_config.json` |
| Static OPF mean reward, admission and SLA violation values | `summary_statistics_ci.csv` | `summary_raw.csv` |
| 95% confidence intervals | `summary_statistics_ci.csv` | `src/admission_sim/evaluation.py` |
| SLA-constrained policy ranking | `sla_constrained_policy_ranking.csv` | `run_config.json` |
| Static OPF 1.75 best feasible policy at 2% threshold | `sla_constrained_policy_ranking.csv` | `scripts/verify_reference_results.py` |
| Improvement relative to conservative admission | `policy_improvement_vs_conservative.csv` | `summary_raw.csv` |
| Paired reward significance tests | `paired_significance_net_profit.csv` | `summary_raw.csv` |
| Paired SLA-rate significance tests | `paired_significance_sla_violation_rate.csv` | `summary_raw.csv` |
| Decision-block diagnostics | `diagnostics_policy_summary.csv` | decision records from a full experiment run |
| Revenue-reliability/Pareto data | `pareto_frontier.csv` | `summary_statistics_ci.csv` |
| Admission-control manuscript figures | `figures/` | `scripts/generate_manuscript_figures.py` |
| Simulator settings | `run_config.json` | `configs/manuscript_config.json` |
| Processed dataset fingerprint | `DATASET_FILES.sha256` | `DATA.md` |
| Compact result integrity | `REFERENCE_RESULTS.sha256` | SHA-256 |

## Principal reference values

| Policy | Mean reward | Admission ratio | SLA violation rate |
|---|---:|---:|---:|
| Static OPF 1.75 | -213.67 | 20.01% | 0.30% |
| Static OPF 2.00 | -304.22 | 21.83% | 1.15% |
| OPF-aware Q-learning | -308.11 | 20.74% | 3.67% |
| Conservative | -818.41 | 12.49% | 0.00% |
| Basic Q-learning | -1228.89 | 11.56% | 0.00% |

At the manuscript SLA-violation-rate constraint of 2%, Static OPF 1.75 has the highest mean reward. The mean reward improvement relative to conservative admission is 604.74 reward units, equivalent to 73.89% of the absolute conservative reward deficit. The paired reward comparison against conservative admission gives a paired t-test p-value of `7.78e-10` and a Wilcoxon signed-rank p-value of `9.31e-09`.

## Sensitivity material

The parameter-sensitivity implementation is located under `supplementary/sensitivity/`. Manuscript sensitivity figures are included in the same directory for correspondence with the submitted article. Sensitivity outputs are separated from the 30-run reference result set because they originate from distinct experiment sweeps.
