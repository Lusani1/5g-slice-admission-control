"""Configuration objects for the forecasting-assisted 5G core slice admission simulator."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import numpy as np

RESOURCES = ("cpu", "mem", "bw")


@dataclass(frozen=True)
class SliceClassConfig:
    name: str
    demand_mean: Tuple[float, float, float]
    lifetime_range: Tuple[int, int]
    revenue_range: Tuple[float, float]
    expected_utilisation: Tuple[float, float, float]
    sla_weight: float
    arrival_weight: float
    demand_std_fraction: float = 0.15
    rejection_weight: float | None = None

    def rejection_weight_value(self) -> float:
        return self.sla_weight if self.rejection_weight is None else self.rejection_weight


@dataclass
class SimulatorConfig:
    """Central configuration for reproducible admission-control experiments.

    The defaults are calibrated to expose the revenue--reliability trade-off that
    the paper studies: conservative and moderate policies should remain SLA-safe,
    while aggressive OPF values should increase admission and revenue but can
    create SLA risk under bursts/degradation.
    """

    # Time and capacity
    decision_epochs: int = 288
    sampling_minutes: int = 5
    capacity: Tuple[float, float, float] = (100.0, 100.0, 100.0)

    # Forecasting
    lookback_window: int = 12
    horizon: int = 1
    forecast_mode: str = "xgboost"  # xgboost, persistence, oracle, no_forecast, synthetic
    synthetic_forecast_bias: float = 0.0
    synthetic_forecast_std: float = 0.04
    forecast_buffer_kappa: float = 0.50
    sigma_min_fraction: float = 0.02

    # Arrivals and load pressure
    base_arrival_rate: float = 3.0
    max_arrivals_per_epoch: int = 10
    low_regime_multiplier: float = 0.65
    high_regime_multiplier: float = 1.65
    burst_multiplier: float = 2.40
    demand_pressure_multiplier: float = 1.15
    request_ordering: str = "revenue_density"

    # Degradation / capacity uncertainty
    degradation_probability: float = 0.035
    degradation_factor_range: Tuple[float, float] = (0.60, 0.90)
    degradation_duration_range: Tuple[int, int] = (6, 30)

    # OPF and policies. The extended upper range is intentional: it exposes the
    # point where overbooking stops being safe and starts creating SLA risk.
    # OPF-aware Q-learning action set Phi. The manuscript defines
    # A^OPF = {(reject, 1.00)} U {(accept, phi): phi in Phi}; therefore 1.00
    # is reserved for rejection and is not included as an accept action.
    opf_set: Tuple[float, ...] = (1.10, 1.20, 1.30, 1.40, 1.50, 1.75, 2.00, 2.25, 2.50)
    static_opfs: Tuple[float, ...] = (1.10, 1.20, 1.30, 1.40, 1.50, 1.75, 2.00, 2.25, 2.50)
    adaptive_initial_opf: float = 1.20
    adaptive_min_opf: float = 1.00
    adaptive_max_opf: float = 2.50
    adaptive_step: float = 0.05
    adaptive_high_sla_threshold: float = 0.02
    adaptive_low_headroom_threshold: float = 0.08
    adaptive_rejection_threshold: float = 0.30

    # Reward / penalties. Rejection penalty is an opportunity-cost term; setting
    # it too high can dominate economics and hide the overbooking trade-off.
    sla_penalty_weights: Tuple[float, float, float] = (70.0, 55.0, 65.0)
    overbooking_penalty_weights: Tuple[float, float, float] = (4.0, 3.0, 4.0)
    rejection_penalty_lambda: float = 0.10
    sla_safe_bonus: float = 2.0
    sla_violation_target: float = 0.02

    # Q-learning
    learning_rate: float = 0.12
    discount_factor: float = 0.95
    epsilon_initial: float = 0.40
    epsilon_min: float = 0.03
    epsilon_decay: float = 0.992
    min_visits_for_greedy: int = 3
    training_episodes: int = 120
    # Basic Q-learning uses accept/reject actions at a fixed OPF of 1.00,
    # matching the manuscript definition A^QL = {reject, accept}.
    q_fixed_opf: float = 1.00
    allow_unseen_state_fallback: bool = False  # False = pure learned greedy policy; True = disclosed safety fallback ablation

    # Forecaster used while training the learning-based policies. The reference
    # experiment uses XGBoost for both training and evaluation.
    rl_train_forecast_mode: str = "xgboost"

    # Workload rows sampled by the admission-control simulator. The reference
    # experiment uses the held-out test split to keep evaluation segments
    # separate from rows used to fit the runtime forecaster.
    workload_data_split: str = "test"  # "test", "train", "val", or "all"
    workload_split_column: str = "split_80_20"
    strict_workload_split: bool = True  # fail rather than fall back to the full profile

    # Experiment design
    monte_carlo_runs: int = 30
    base_seed: int = 42
    common_random_scenarios: bool = True

    # Slice classes
    slice_classes: Dict[str, SliceClassConfig] = field(default_factory=lambda: {
        "URLLC": SliceClassConfig(
            name="URLLC",
            demand_mean=(9.0, 6.0, 7.0),
            lifetime_range=(8, 34),
            revenue_range=(30, 48),
            expected_utilisation=(0.82, 0.78, 0.75),
            sla_weight=2.30,
            arrival_weight=0.18,
        ),
        "eMBB": SliceClassConfig(
            name="eMBB",
            demand_mean=(12.0, 9.0, 17.0),
            lifetime_range=(12, 58),
            revenue_range=(24, 40),
            expected_utilisation=(0.72, 0.68, 0.82),
            sla_weight=1.35,
            arrival_weight=0.36,
        ),
        "mMTC": SliceClassConfig(
            name="mMTC",
            demand_mean=(3.5, 4.0, 4.5),
            lifetime_range=(18, 80),
            revenue_range=(7, 18),
            expected_utilisation=(0.48, 0.55, 0.46),
            sla_weight=0.95,
            arrival_weight=0.26,
        ),
        "Elastic video": SliceClassConfig(
            name="Elastic video",
            demand_mean=(8.0, 7.0, 23.0),
            lifetime_range=(10, 44),
            revenue_range=(18, 34),
            expected_utilisation=(0.58, 0.60, 0.84),
            sla_weight=0.85,
            arrival_weight=0.20,
        ),
    })

    def capacity_array(self) -> np.ndarray:
        return np.asarray(self.capacity, dtype=float)

    def sla_penalty_array(self) -> np.ndarray:
        return np.asarray(self.sla_penalty_weights, dtype=float)

    def overbooking_penalty_array(self) -> np.ndarray:
        return np.asarray(self.overbooking_penalty_weights, dtype=float)

    def class_names(self) -> List[str]:
        return list(self.slice_classes.keys())


def smoke_config() -> SimulatorConfig:
    cfg = SimulatorConfig()
    cfg.decision_epochs = 80
    cfg.monte_carlo_runs = 2
    cfg.training_episodes = 4
    cfg.static_opfs = (1.20, 1.75, 2.25)
    cfg.opf_set = (1.20, 1.75, 2.25)
    return cfg
