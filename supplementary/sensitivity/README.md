# Parameter sensitivity

This directory contains the one-factor-at-a-time parameter-sensitivity execution path corresponding to the manuscript robustness analysis.

The comparison uses:

- Conservative admission
- Static OPF 1.75
- Static OPF 2.00

The tested levels are:

- Capacity: 80, 100, 120 units per resource
- Demand multiplier: 0.8, 1.0, 1.2
- SLA penalty multiplier: 0.5, 1.0, 2.0
- Overbooking penalty multiplier: 0.5, 1.0, 2.0

Run from the repository root:

```bash
python3 supplementary/sensitivity/run_parameter_sensitivity.py \
  --data-root /path/to/materna_synthetic_slice_profiles_forecasting_ready \
  --output-dir results/parameter_sensitivity \
  --parameters sla_penalty,overbooking_penalty,demand,capacity \
  --runs 30 \
  --epochs 288 \
  --training-episodes 120
```

The figure files used in the manuscript sensitivity section are stored in `manuscript_figures/` for correspondence with the submitted article. The parameter-sensitivity runner can be used to generate a fresh set of numeric outputs from the processed data.
