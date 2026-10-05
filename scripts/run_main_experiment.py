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

from admission_sim import SimulatorConfig, MaternaProfileStore
from admission_sim.forecaster import load_or_train_xgboost
from admission_sim.evaluation import run_policy_suite


def parse_float_list(s: str):
    return tuple(float(x.strip()) for x in s.split(",") if x.strip())


def main():
    ap = argparse.ArgumentParser(description="Run full Monte Carlo admission-control evaluation")
    ap.add_argument("--data-root", required=True, help="Dataset ZIP or extracted dataset directory")
    ap.add_argument("--output-dir", default="results/full_admission_control_xgboost")
    ap.add_argument("--model-path", default="models/xgboost_runtime_forecaster.joblib")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=288)
    ap.add_argument("--training-episodes", type=int, default=120)
    ap.add_argument("--max-profiles", type=int, default=147)
    ap.add_argument("--max-train-windows", type=int, default=250000)
    ap.add_argument("--n-estimators", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--static-opfs", default="1.10,1.20,1.30,1.40,1.50,1.75,2.00,2.25,2.50")
    ap.add_argument("--opf-set", default="1.10,1.20,1.30,1.40,1.50,1.75,2.00,2.25,2.50",
                     help="OPF-aware Q-learning accept-action set Phi. The value 1.00 is reserved for rejection and should not appear here.")
    ap.add_argument("--q-fixed-opf", type=float, default=1.00,
                     help="Fixed overbooking factor used by basic accept/reject Q-learning. The manuscript configuration uses 1.00.")
    ap.add_argument("--rl-train-forecast-mode", choices=["persistence", "xgboost"], default="xgboost",
                     help="Forecaster used while training Q-learning policies. The manuscript configuration uses xgboost.")
    ap.add_argument("--workload-data-split", choices=["train", "val", "test", "all"], default="test",
                     help="Profile split sampled by the admission-control simulator. The manuscript configuration uses the held-out test split.")
    ap.add_argument("--base-arrival-rate", type=float, default=3.0)
    ap.add_argument("--rejection-penalty-lambda", type=float, default=0.10)
    ap.add_argument("--forecast-buffer-kappa", type=float, default=0.50)
    ap.add_argument("--demand-pressure-multiplier", type=float, default=1.15)
    ap.add_argument("--allow-unseen-state-fallback", action="store_true", help="Allow Q-learning policies to use the disclosed safety fallback for unseen evaluation states. Default is pure greedy/reject unseen states.")
    ap.add_argument("--allow-split-fallback", action="store_true", help="Allow workload sampling to fall back from the requested split to full profiles. Not used by the manuscript configuration.")
    ap.add_argument("--force-train-model", action="store_true", help="Retrain the runtime XGBoost forecaster even if --model-path exists.")
    ap.add_argument("--allow-smoke-model", action="store_true", help="Permit loading a smoke-test XGBoost model. Intended only for quick development checks.")
    ap.add_argument("--make-plots", action="store_true", help="Generate journal figures after the simulation; figures may also be generated separately.")
    args = ap.parse_args()

    cfg = SimulatorConfig()
    cfg.monte_carlo_runs = args.runs
    cfg.decision_epochs = args.epochs
    cfg.training_episodes = args.training_episodes
    cfg.base_seed = args.seed
    cfg.forecast_mode = "xgboost"
    cfg.static_opfs = parse_float_list(args.static_opfs)
    cfg.opf_set = parse_float_list(args.opf_set)
    cfg.adaptive_max_opf = max(cfg.opf_set)
    cfg.q_fixed_opf = args.q_fixed_opf
    cfg.rl_train_forecast_mode = args.rl_train_forecast_mode
    cfg.workload_data_split = args.workload_data_split
    cfg.strict_workload_split = not args.allow_split_fallback
    cfg.allow_unseen_state_fallback = bool(args.allow_unseen_state_fallback)
    cfg.base_arrival_rate = args.base_arrival_rate
    cfg.rejection_penalty_lambda = args.rejection_penalty_lambda
    cfg.forecast_buffer_kappa = args.forecast_buffer_kappa
    cfg.demand_pressure_multiplier = args.demand_pressure_multiplier

    store = MaternaProfileStore(args.data_root, max_profiles=args.max_profiles, min_rows=1000)
    forecaster = load_or_train_xgboost(
        data_root=args.data_root,
        model_path=args.model_path,
        cfg=cfg,
        max_train_windows=args.max_train_windows,
        max_profiles=args.max_profiles,
        n_estimators_override=args.n_estimators,
        seed=args.seed,
        force_train=args.force_train_model,
        allow_smoke_model=args.allow_smoke_model,
    )
    out = Path(args.output_dir)
    run_policy_suite(cfg, store, forecaster, output_dir=out)
    if args.make_plots:
        from admission_sim.visualization import make_all_plots
        make_all_plots(out)
    print(f"\nFull Monte Carlo evaluation complete. Results written to {out}")
    if not args.make_plots:
        print("Generate figures with: python3 scripts/generate_journal_figures.py --results-dir " + str(out))

if __name__ == "__main__":
    main()
