#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admission_sim import SimulatorConfig
from admission_sim.forecaster import train_xgboost_from_profiles


def main():
    ap = argparse.ArgumentParser(description="Train XGBoost runtime forecaster for admission-control simulator")
    ap.add_argument("--data-root", required=True, help="Dataset ZIP or extracted dataset directory")
    ap.add_argument("--model-path", default="models/xgboost_runtime_forecaster.joblib")
    ap.add_argument("--max-train-windows", type=int, default=250000)
    ap.add_argument("--max-profiles", type=int, default=147)
    ap.add_argument("--n-estimators", type=int, default=None, help="Override number of trees; use e.g. 100 for smoke tests")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = SimulatorConfig()
    forecaster = train_xgboost_from_profiles(
        data_root=args.data_root,
        model_path=args.model_path,
        cfg=cfg,
        max_train_windows=args.max_train_windows,
        max_profiles=args.max_profiles,
        n_estimators_override=args.n_estimators,
        seed=args.seed,
    )
    print(f"Saved XGBoost forecaster to {args.model_path}")

if __name__ == "__main__":
    main()
