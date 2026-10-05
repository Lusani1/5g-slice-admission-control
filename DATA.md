# Data

## Source

The admission-control workload is derived from the GWA-T-13 Materna workload traces distributed through the Grid Workloads Archive. The official source is:

- **@Large Research, GWA-T-13 Materna:** https://atlarge-research.com/gwa-t-13/

The archive identifies Materna GmbH Information & Communications, Dortmund, Germany, as the trace provider and requests acknowledgement of the data source in publications that use the trace. The raw third-party traces are not redistributed in this repository.

## Processed profile format

Processed compressed CSV files are expected under `profiles_clean/` with names matching:

```text
clean_profile_*.csv.gz
```

Fields used by the admission-control implementation include:

- `cpu_factor`
- `mem_factor`
- `bw_factor`
- `demand_score`
- `split_80_20`

The reference experiment uses chronological `train` rows for runtime-forecaster fitting and samples admission-control evaluation segments from held-out `test` rows.

## Reference-run fingerprint

`results/reference_run/DATASET_FILES.sha256` records the SHA-256 hash of each processed profile used in the reference experiment. Paths are stored relative to `profiles_clean/` so that the fingerprint is machine-independent.

A local profile directory can be checked by computing SHA-256 hashes and comparing them with the supplied fingerprint.

## Data licensing

The source workload traces remain subject to the terms of their original provider. This repository contains code and derived result evidence, not the raw third-party dataset.
