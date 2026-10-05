# Reproducibility procedure

## 1. Software environment

A recent CPython installation is required. Create an isolated environment and install the listed dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install pytest
```

Environment versions can be recorded with:

```bash
python3 scripts/print_environment.py
```

## 2. Implementation checks

```bash
python3 -m pytest tests -q
```

The test suite contains 18 implementation checks covering the principal equations, feasibility rules, OPF action space, reward terms, confidence intervals, and paired-scenario behaviour.

## 3. Dataset

The simulator accepts an extracted Materna-derived profile directory or an equivalent archive containing:

```text
profiles_clean/
  clean_profile_001_....csv.gz
  clean_profile_002_....csv.gz
  ...
```

Each processed profile contains normalized resource factors and the chronological split column `split_80_20`. Admission-control evaluation segments are sampled from rows labelled `test`, while XGBoost training uses rows labelled `train`.

The exact processed files used for the reference run are fingerprinted in:

```text
results/reference_run/DATASET_FILES.sha256
```

## 4. Main experiment

Run from the repository root:

```bash
./run_main_experiment.sh \
  /path/to/materna_synthetic_slice_profiles_forecasting_ready \
  results/reproduction
```

The manuscript configuration is:

```text
Monte Carlo runs:              30
Decision epochs per run:       288
Decision interval:             5 minutes
RL training episodes:          120
Base seed:                     42
Capacity:                      100, 100, 100
Basic Q-learning fixed OPF:    1.00
Static OPFs:                   1.10,1.20,1.30,1.40,1.50,1.75,2.00,2.25,2.50
OPF-aware accept actions:      1.10,1.20,1.30,1.40,1.50,1.75,2.00,2.25,2.50
Runtime forecaster:            XGBoost
Learning-policy forecaster:    XGBoost
Admission workload split:      test
Common random scenarios:       enabled
SLA ranking threshold:         2%
```

The complete execution command is defined in `run_main_experiment.sh`.

## 5. Main outputs

The experiment produces:

- `summary_raw.csv`: one row per policy and evaluation seed;
- `summary_statistics_ci.csv`: mean, standard deviation, standard error and Student-t 95% confidence intervals;
- `decisions_raw.csv`: decision-level records;
- `timeseries_raw.csv`: epoch-level records;
- `sla_constrained_policy_ranking.csv`: ranking under the 2% violation-rate limit;
- `paired_significance_net_profit.csv`: paired reward tests;
- `paired_significance_sla_violation_rate.csv`: paired SLA-rate tests;
- `policy_improvement_vs_conservative.csv`: paired improvement relative to conservative admission;
- `diagnostics_policy_summary.csv`: decision-block diagnostics;
- `run_config.json`: simulator configuration and run metadata.

## 6. Reference-result verification

```bash
python3 scripts/verify_reference_results.py
```

The verification script checks the 30 paired seeds for every policy and evaluates the principal manuscript quantities directly from the archived CSV files.

## 7. Figure regeneration

```bash
python3 scripts/generate_manuscript_figures.py \
  --results-dir results/reference_run \
  --out-dir results/reference_run/figures
```

Figures are generated from the archived CSV outputs.

## 8. Supplementary sensitivity analysis

The one-factor-at-a-time parameter-sensitivity runner is located at:

```text
supplementary/sensitivity/run_parameter_sensitivity.py
```

It compares conservative admission, Static OPF 1.75, and Static OPF 2.00 while varying capacity, demand pressure, SLA penalties, and overbooking penalties. Execution details and scope are documented in `supplementary/sensitivity/README.md`.
