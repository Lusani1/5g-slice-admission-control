#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/materna_synthetic_slice_profiles_forecasting_ready[.zip] [output_dir]"
  exit 2
fi

DATA_ROOT="$1"
OUT="${2:-results/reproduction}"
MODEL="models/xgboost_runtime_forecaster.joblib"

if [[ ! -e "$DATA_ROOT" ]]; then
  echo "ERROR: data path does not exist: $DATA_ROOT" >&2
  exit 1
fi

for f in scripts/run_main_experiment.py scripts/build_results_manifest.py requirements.txt; do
  [[ -f "$f" ]] || { echo "ERROR: run this script from the repository root; missing $f" >&2; exit 1; }
done

mkdir -p "$(dirname "$OUT")" models
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "=== Tests ==="
python3 -m pytest tests -q

echo "=== Dataset preflight ==="
python3 - "$DATA_ROOT" <<'PY'
import sys
sys.path.insert(0, 'src')
from admission_sim.data import MaternaProfileStore
store = MaternaProfileStore(sys.argv[1], max_profiles=147, min_rows=1000)
if not store.profile_paths:
    raise SystemExit("No clean profiles found")
df = store.load_profile(store.profile_paths[0])
required = {"cpu_factor", "mem_factor", "bw_factor", "demand_score", "split_80_20"}
missing = required - set(df.columns)
if missing:
    raise SystemExit(f"Missing required columns: {sorted(missing)}")
labels = set(df["split_80_20"].astype(str).str.lower())
if not {"train", "test"}.issubset(labels):
    raise SystemExit("Expected both train and test rows in split_80_20")
print(f"Profiles available (capped at 147): {len(store.profile_paths)}")
print("Dataset preflight passed")
PY

rm -f "$MODEL" "${MODEL%.joblib}.metadata.json"

echo "=== Main experiment ==="
python3 scripts/run_main_experiment.py \
  --data-root "$DATA_ROOT" \
  --output-dir "$OUT" \
  --model-path "$MODEL" \
  --runs 30 \
  --epochs 288 \
  --training-episodes 120 \
  --max-profiles 147 \
  --max-train-windows 250000 \
  --seed 42 \
  --static-opfs 1.10,1.20,1.30,1.40,1.50,1.75,2.00,2.25,2.50 \
  --opf-set 1.10,1.20,1.30,1.40,1.50,1.75,2.00,2.25,2.50 \
  --q-fixed-opf 1.00 \
  --rl-train-forecast-mode xgboost \
  --workload-data-split test \
  --base-arrival-rate 3.0 \
  --rejection-penalty-lambda 0.10 \
  --forecast-buffer-kappa 0.50 \
  --demand-pressure-multiplier 1.15 \
  --force-train-model \
  --make-plots

if [[ -f "$DATA_ROOT" ]]; then
  sha256sum "$DATA_ROOT" > "$OUT/DATASET.sha256"
else
  python3 - "$DATA_ROOT" "$OUT/DATASET_FILES.sha256" <<'PYHASH'
from pathlib import Path
import hashlib, sys
root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
files = sorted(root.rglob("clean_profile_*.csv.gz"))
with out.open("w", encoding="utf-8") as f:
    for path in files:
        h = hashlib.sha256()
        with path.open("rb") as src:
            for block in iter(lambda: src.read(1024 * 1024), b""):
                h.update(block)
        rel = path.relative_to(root).as_posix()
        f.write(f"{h.hexdigest()}  {rel}\n")
PYHASH
fi
python3 scripts/build_results_manifest.py --results-dir "$OUT"

python3 - "$OUT" <<'PY'
from pathlib import Path
import json, sys
import pandas as pd
out = Path(sys.argv[1])
cfg = json.loads((out / "run_config.json").read_text())
assert cfg["monte_carlo_runs"] == 30
assert cfg["decision_epochs"] == 288
assert cfg["training_episodes"] == 120
assert cfg["workload_data_split"] == "test"
assert cfg["strict_workload_split"] is True
assert cfg["common_random_scenarios"] is True
assert cfg.get("_run_metadata", {}).get("workload_sampling_used_full_profile_fallback") is False
summary = pd.read_csv(out / "summary_raw.csv")
counts = summary.groupby("policy")["seed"].nunique()
if (counts != 30).any():
    raise SystemExit(f"Expected 30 unique seeds per policy:\n{counts}")
print("Configuration and seed checks passed")
PY

echo "Results written to: $OUT"
