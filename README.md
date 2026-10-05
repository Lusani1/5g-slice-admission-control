# SLA-Aware Resource Overbooking for 5G Core Network Slicing

Reproducibility materials accompanying the manuscript **“SLA-Aware Resource Overbooking for 5G Core Network Slicing Using Forecasting-Assisted Admission Control.”**

## Authors

- Lusani Mamushiane
- Dr. Mduduzi C.Hlophe

## Reviewers

- Dr. Albert A.Lysko
- Dr. Joyce Mwangama

## Scope

The repository contains the admission-control simulator, manuscript experiment configuration, implementation tests, a compact reference result set from the 30-run experiment, figure-generation utilities, and supplementary parameter-sensitivity code.

The reference experiment evaluates conservative admission, static overbooking factors (OPFs), an adaptive heuristic, basic Q-learning, and OPF-aware Q-learning under paired stochastic scenarios. CPU, memory, and bandwidth are modelled jointly. XGBoost is used as the runtime forecaster in the main admission-control experiment.

## Reference experiment

- 30 paired Monte Carlo seeds
- 288 decision epochs per run
- 5-minute decision interval
- 120 Q-learning training episodes
- Resource capacities: `[100, 100, 100]`
- XGBoost for learning-policy training and evaluation
- Admission workloads sampled from the held-out `test` split
- Common random scenarios across policies
- Basic Q-learning fixed OPF: `1.00`
- OPF-aware accept-action set: `1.10, 1.20, 1.30, 1.40, 1.50, 1.75, 2.00, 2.25, 2.50`
- SLA violation-rate constraint for policy ranking: `2%`

The reference results identify **Static OPF 1.75** as the highest-reward policy satisfying the 2% SLA-violation-rate constraint.

## Repository structure

```text
.
├── AUTHORS_AND_REVIEWERS.md
├── CITATION.cff
├── DATA.md
├── EVIDENCE_INDEX.md
├── LICENSE
├── README.md
├── REPRODUCIBILITY.md
├── configs/
│   └── manuscript_config.json
├── models/
├── results/
│   └── reference_run/
├── scripts/
│   ├── build_results_manifest.py
│   ├── generate_manuscript_figures.py
│   ├── print_environment.py
│   ├── run_main_experiment.py
│   ├── train_xgboost_runtime_forecaster.py
│   └── verify_reference_results.py
├── src/admission_sim/
├── supplementary/sensitivity/
└── tests/
```

## Verification

Create an isolated Python environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install pytest
```

Run the implementation tests:

```bash
python3 -m pytest tests -q
```

Verify the archived reference results:

```bash
python3 scripts/verify_reference_results.py
```

## Main experiment

From the repository root:

```bash
./run_main_experiment.sh \
  /path/to/materna_synthetic_slice_profiles_forecasting_ready \
  results/reproduction
```

The experiment runner performs data preflight checks, trains the XGBoost runtime forecaster, executes the 30 paired Monte Carlo runs, verifies configuration and seed counts, creates result hashes, and regenerates the admission-control figures used in the manuscript.

Detailed execution instructions are provided in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). The mapping between manuscript results and repository evidence is given in [EVIDENCE_INDEX.md](EVIDENCE_INDEX.md). Complete decision-level and epoch-level outputs are distributed separately in `IEEE_Access_Full_Run_Evidence.zip` to keep the repository compact.

## Data availability

The raw Materna workload traces are not redistributed. The expected processed-profile structure and dataset fingerprint are documented in [DATA.md](DATA.md). Profile hashes for the reference run are listed in `results/reference_run/DATASET_FILES.sha256`.
