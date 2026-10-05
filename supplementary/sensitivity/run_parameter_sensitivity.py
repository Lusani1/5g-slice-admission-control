#!/usr/bin/env python3
"""Run the one-factor-at-a-time parameter sensitivity analysis reported in the manuscript."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from admission_sim import SimulatorConfig, MaternaProfileStore
from admission_sim.evaluation import run_policy_suite
from admission_sim.forecaster import load_or_train_xgboost, PersistenceForecaster
from admission_sim.sensitivity import make_limited_policies, write_analysis_outputs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-factor-at-a-time parameter sensitivity")
    p.add_argument("--data-root", required=True, help="Processed Materna-derived profile directory or ZIP")
    p.add_argument("--output-dir", default="results/parameter_sensitivity")
    p.add_argument("--model-path", default="models/xgboost_runtime_forecaster.joblib")
    p.add_argument("--forecast-mode", choices=("xgboost", "persistence"), default="xgboost")
    p.add_argument("--runs", type=int, default=30)
    p.add_argument("--epochs", type=int, default=288)
    p.add_argument("--training-episodes", type=int, default=120)
    p.add_argument("--max-profiles", type=int, default=147)
    p.add_argument("--max-train-windows", type=int, default=250000)
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="Rerun conditions even when cached summaries exist")
    return p.parse_args()


def annotate(df: pd.DataFrame, parameter: str, level_value: float, level_label: str, key: str) -> pd.DataFrame:
    out = df.copy()
    out["parameter"] = parameter
    out["level_value"] = float(level_value)
    out["level_label"] = level_label
    out["condition_key"] = key
    return out


def run_condition(key: str, cfg: SimulatorConfig, store: MaternaProfileStore, forecaster,
                  output_root: Path, force: bool, metadata: dict) -> pd.DataFrame:
    condition_dir = output_root / "conditions" / key
    summary_path = condition_dir / "summary_raw.csv"
    if summary_path.exists() and not force:
        return pd.read_csv(summary_path)
    condition_dir.mkdir(parents=True, exist_ok=True)
    summary, _, _ = run_policy_suite(
        cfg,
        store,
        forecaster,
        output_dir=condition_dir,
        policies=make_limited_policies(cfg),
    )
    (condition_dir / "sensitivity_condition.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    cfg = SimulatorConfig()
    cfg.monte_carlo_runs = args.runs
    cfg.decision_epochs = args.epochs
    cfg.training_episodes = args.training_episodes
    cfg.base_seed = args.seed
    cfg.forecast_mode = args.forecast_mode
    cfg.static_opfs = (1.75, 2.00)
    cfg.common_random_scenarios = True
    cfg.workload_data_split = "test"
    cfg.strict_workload_split = True
    cfg.allow_unseen_state_fallback = False

    store = MaternaProfileStore(args.data_root, max_profiles=args.max_profiles, min_rows=1000)
    if args.forecast_mode == "persistence":
        forecaster = PersistenceForecaster()
    else:
        forecaster = load_or_train_xgboost(
            data_root=args.data_root,
            model_path=args.model_path,
            cfg=cfg,
            max_train_windows=args.max_train_windows,
            max_profiles=args.max_profiles,
            n_estimators_override=args.n_estimators,
            seed=args.seed,
        )

    base_sla = np.asarray(cfg.sla_penalty_weights, dtype=float)
    base_overbooking = np.asarray(cfg.overbooking_penalty_weights, dtype=float)
    base_demand = float(cfg.demand_pressure_multiplier)

    baseline = run_condition(
        "baseline", copy.deepcopy(cfg), store, forecaster, out, args.force,
        {
            "capacity": list(cfg.capacity),
            "demand_pressure_multiplier": base_demand,
            "sla_penalty_weights": list(cfg.sla_penalty_weights),
            "overbooking_penalty_weights": list(cfg.overbooking_penalty_weights),
        },
    )

    frames = [
        annotate(baseline, "capacity", 100.0, "100", "baseline"),
        annotate(baseline, "demand", 1.0, "1.0x", "baseline"),
        annotate(baseline, "sla_penalty", 1.0, "1.0x", "baseline"),
        annotate(baseline, "overbooking_penalty", 1.0, "1.0x", "baseline"),
    ]

    for capacity in (80.0, 120.0):
        c = copy.deepcopy(cfg)
        c.capacity = (capacity, capacity, capacity)
        key = f"capacity_{capacity:g}"
        result = run_condition(key, c, store, forecaster, out, args.force, {"capacity": list(c.capacity)})
        frames.append(annotate(result, "capacity", capacity, f"{capacity:g}", key))

    for multiplier in (0.8, 1.2):
        c = copy.deepcopy(cfg)
        c.demand_pressure_multiplier = base_demand * multiplier
        key = f"demand_{multiplier:g}x"
        result = run_condition(
            key, c, store, forecaster, out, args.force,
            {"relative_level": multiplier, "demand_pressure_multiplier": c.demand_pressure_multiplier},
        )
        frames.append(annotate(result, "demand", multiplier, f"{multiplier:g}x", key))

    for multiplier in (0.5, 2.0):
        c = copy.deepcopy(cfg)
        c.sla_penalty_weights = tuple((base_sla * multiplier).tolist())
        key = f"sla_penalty_{multiplier:g}x"
        result = run_condition(
            key, c, store, forecaster, out, args.force,
            {"multiplier": multiplier, "sla_penalty_weights": list(c.sla_penalty_weights)},
        )
        frames.append(annotate(result, "sla_penalty", multiplier, f"{multiplier:g}x", key))

    for multiplier in (0.5, 2.0):
        c = copy.deepcopy(cfg)
        c.overbooking_penalty_weights = tuple((base_overbooking * multiplier).tolist())
        key = f"overbooking_penalty_{multiplier:g}x"
        result = run_condition(
            key, c, store, forecaster, out, args.force,
            {"multiplier": multiplier, "overbooking_penalty_weights": list(c.overbooking_penalty_weights)},
        )
        frames.append(annotate(result, "overbooking_penalty", multiplier, f"{multiplier:g}x", key))

    raw = pd.concat(frames, ignore_index=True)
    write_analysis_outputs(raw, out, sla_target=cfg.sla_violation_target, cfg=cfg)

    manifest = {
        "policies": ["conservative", "static_opf_1.75_forecast", "static_opf_2.00_forecast"],
        "runs": args.runs,
        "epochs": args.epochs,
        "training_episodes": args.training_episodes,
        "seed": args.seed,
        "forecast_mode": args.forecast_mode,
        "capacity_levels": [80, 100, 120],
        "demand_multipliers": [0.8, 1.0, 1.2],
        "sla_penalty_multipliers": [0.5, 1.0, 2.0],
        "overbooking_penalty_multipliers": [0.5, 1.0, 2.0],
        "baseline_sla_penalty_weights": list(cfg.sla_penalty_weights),
        "baseline_overbooking_penalty_weights": list(cfg.overbooking_penalty_weights),
        "baseline_demand_pressure_multiplier": base_demand,
    }
    (out / "parameter_sensitivity_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Sensitivity results written to {out}")


if __name__ == "__main__":
    main()
