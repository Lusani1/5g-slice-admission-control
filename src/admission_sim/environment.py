"""Discrete-time forecasting-assisted slice admission-control environment.

The simulator implements the overbooking model in the paper:

* nominal OPF feasibility: R + d <= phi C
* forecast-aware SLA feasibility: U_hat + buffer + eta d <= C
* realised SLA violation after decisions: U(t) > C(t)
* reward: admission revenue - SLA penalty - rejection penalty - overbooking penalty + safe bonus

The implementation uses common random scenarios for fair policy comparison. For a
fixed seed, every policy sees the same arrivals, slice classes, demands, revenue,
lifetimes, profile seeds, and capacity-degradation sequence. This is essential for
reliable Monte Carlo policy comparisons.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd

from .config import SimulatorConfig
from .data import MaternaProfileStore, ProfileSegment
from .forecaster import BaseForecaster, PersistenceForecaster


def utilisation_multiplier(resource_factors: np.ndarray) -> np.ndarray:
    """Map normalised Materna resource factors to utilisation multipliers.

    The factors are in [0, 1]. The mapping keeps typical utilisation below the
    nominal reservation while still allowing bursty periods to push realised
    utilisation close to or above physical capacity under aggressive OPF.
    """
    f = np.asarray(resource_factors, dtype=float)
    return np.clip(0.35 + 0.95 * f, 0.15, 1.30)


@dataclass
class SliceRequest:
    request_id: int
    t: int
    cls_name: str
    demand: np.ndarray
    lifetime: int
    revenue: float
    eta: np.ndarray
    sla_weight: float
    rejection_weight: float
    profile_seed: int

    @property
    def revenue_density(self) -> float:
        return float(self.revenue / max(float(np.sum(self.demand)), 1e-9))


@dataclass
class ActiveSlice:
    request: SliceRequest
    start_t: int
    end_t: int
    profile: ProfileSegment
    forecast_factors: np.ndarray | None = None

    def offset(self, t: int) -> int:
        return max(0, min(t - self.start_t, len(self.profile.frame) - 1))

    def history_features(self, t: int, window: int) -> np.ndarray:
        off = self.offset(t)
        start = max(0, off - window + 1)
        return self.profile.features[start:off + 1]

    def actual_utilisation(self, t: int) -> np.ndarray:
        off = self.offset(t)
        factor = self.profile.resource_factors[off]
        return self.request.demand * self.request.eta * utilisation_multiplier(factor)

    def true_future_utilisation(self, t: int, horizon: int) -> np.ndarray:
        off = self.offset(t + horizon)
        factor = self.profile.resource_factors[off]
        return self.request.demand * self.request.eta * utilisation_multiplier(factor)

    def forecast_utilisation(self, t: int, forecaster: BaseForecaster, window: int) -> np.ndarray:
        off = self.offset(t)
        if self.forecast_factors is not None and off < len(self.forecast_factors):
            pred_factor = self.forecast_factors[off]
        else:
            hist = self.history_features(t, window)
            pred_factor = forecaster.forecast(hist)
        return self.request.demand * self.request.eta * utilisation_multiplier(pred_factor)


@dataclass
class StepContext:
    t: int
    capacity: np.ndarray
    reserved_load: np.ndarray
    actual_utilisation: np.ndarray
    forecast_utilisation: np.ndarray
    buffered_forecast: np.ndarray
    forecast_buffer: np.ndarray
    recent_sla_rate: float
    recent_rejection_rate: float
    demand_regime: str
    is_burst: bool


@dataclass
class DecisionResult:
    accept: bool
    selected_opf: float
    reason: str = ""
    use_forecast: bool = True


@dataclass
class Scenario:
    seed: int
    run_profile: ProfileSegment
    requests_by_t: list[list[SliceRequest]]
    capacities: list[np.ndarray]
    degradation_factors: list[float]
    degradation_events: int


class AdmissionControlSimulator:
    def __init__(
        self,
        cfg: SimulatorConfig,
        profile_store: MaternaProfileStore,
        forecaster: BaseForecaster | None = None,
        seed: int = 42,
    ):
        self.cfg = cfg
        self.profile_store = profile_store
        self.forecaster = forecaster or PersistenceForecaster()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.base_capacity = cfg.capacity_array()
        self._scenario: Scenario | None = None
        self._last_scenario_used_full_profile_fallback = False
        self.reset(seed=seed)

    # ------------------------------------------------------------------
    # Scenario generation
    # ------------------------------------------------------------------
    def make_scenario(self, seed: int) -> Scenario:
        """Create one common random scenario for all policies for a seed."""
        rng = np.random.default_rng(seed)
        length = self.cfg.decision_epochs + self.cfg.lookback_window + self.cfg.horizon + 5
        split_value = None if self.cfg.workload_data_split.lower() == "all" else self.cfg.workload_data_split
        run_profile, used_fallback = self.profile_store.sample_segment(
            rng, length=length, min_length=length,
            split_column=self.cfg.workload_split_column, split_value=split_value,
            allow_fallback=not self.cfg.strict_workload_split,
        )
        self._last_scenario_used_full_profile_fallback = used_fallback

        requests_by_t: list[list[SliceRequest]] = []
        capacities: list[np.ndarray] = []
        degradation_factors: list[float] = []
        degradation_remaining = 0
        degradation_factor = 1.0
        degradation_events = 0
        request_counter = 0

        for t in range(self.cfg.decision_epochs):
            if degradation_remaining > 0:
                degradation_remaining -= 1
            else:
                degradation_factor = 1.0
                if rng.random() < self.cfg.degradation_probability:
                    degradation_factor = float(rng.uniform(*self.cfg.degradation_factor_range))
                    degradation_remaining = int(rng.integers(
                        self.cfg.degradation_duration_range[0],
                        self.cfg.degradation_duration_range[1] + 1,
                    ))
                    degradation_events += 1
            capacities.append(self.base_capacity * degradation_factor)
            degradation_factors.append(float(degradation_factor))

            regime, burst, demand_score = self._regime_from_profile(run_profile, t)
            lam = self._arrival_rate_from_state(regime, burst)
            n = int(min(rng.poisson(lam), self.cfg.max_arrivals_per_epoch))
            reqs: list[SliceRequest] = []
            if n > 0:
                names, weights = self._class_probabilities_from_state(regime, burst)
                for _ in range(n):
                    cls_name = str(rng.choice(names, p=weights))
                    c = self.cfg.slice_classes[cls_name]
                    mean = np.asarray(c.demand_mean, dtype=float)
                    std = np.maximum(mean * c.demand_std_fraction, 0.05)
                    demand = rng.normal(mean, std)
                    demand = np.clip(demand, mean * 0.35, mean * 1.85)

                    # Demand pressure represents practical enterprise pressure on
                    # a shared 5G core during high-load/bursty periods.
                    scale = self.cfg.demand_pressure_multiplier * (0.85 + 0.40 * float(np.clip(demand_score, 0.0, 1.0)))
                    if regime.lower() == "high":
                        scale *= 1.10
                    if burst:
                        scale *= 1.12
                    demand = demand * scale

                    lifetime = int(rng.integers(c.lifetime_range[0], c.lifetime_range[1] + 1))
                    revenue = float(rng.uniform(c.revenue_range[0], c.revenue_range[1]))
                    profile_seed = int(rng.integers(0, 2**31 - 1))
                    reqs.append(SliceRequest(
                        request_id=request_counter,
                        t=t,
                        cls_name=cls_name,
                        demand=demand.astype(float),
                        lifetime=lifetime,
                        revenue=revenue,
                        eta=np.asarray(c.expected_utilisation, dtype=float),
                        sla_weight=float(c.sla_weight),
                        rejection_weight=float(c.rejection_weight_value()),
                        profile_seed=profile_seed,
                    ))
                    request_counter += 1
            if self.cfg.request_ordering == "revenue_density":
                reqs.sort(key=lambda r: r.revenue_density, reverse=True)
            requests_by_t.append(reqs)

        return Scenario(
            seed=seed,
            run_profile=run_profile,
            requests_by_t=requests_by_t,
            capacities=capacities,
            degradation_factors=degradation_factors,
            degradation_events=degradation_events,
        )

    # ------------------------------------------------------------------
    # Episode state management
    # ------------------------------------------------------------------
    def reset(self, seed: int | None = None, scenario: Scenario | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.seed = seed
        self._scenario = scenario
        if scenario is None:
            scenario = self.make_scenario(self.seed)
            self._scenario = scenario
        self.run_profile = scenario.run_profile
        self.active_slices: list[ActiveSlice] = []
        self.request_counter = 0
        self.degradation_events = scenario.degradation_events
        self.degradation_factor = 1.0
        self.epoch_records: list[dict[str, Any]] = []
        self.decision_records: list[dict[str, Any]] = []

    def _run_index(self, t: int) -> int:
        return min(t + self.cfg.lookback_window, len(self.run_profile.frame) - 1)

    def _regime_from_profile(self, profile: ProfileSegment, t: int) -> tuple[str, bool, float]:
        idx = min(t + self.cfg.lookback_window, len(profile.frame) - 1)
        row = profile.frame.iloc[idx]
        regime = str(row["demand_regime"]) if "demand_regime" in row else "medium"
        burst = bool(row["is_burst"]) if "is_burst" in row else False
        demand_score = float(row["demand_score"])
        return regime, burst, demand_score

    def current_regime(self, t: int) -> tuple[str, bool, float]:
        return self._regime_from_profile(self.run_profile, t)

    def _arrival_rate_from_state(self, regime: str, burst: bool) -> float:
        rate = self.cfg.base_arrival_rate
        if regime.lower() == "low":
            rate *= self.cfg.low_regime_multiplier
        elif regime.lower() == "high":
            rate *= self.cfg.high_regime_multiplier
        if burst:
            rate *= self.cfg.burst_multiplier
        return rate

    def _class_probabilities_from_state(self, regime: str, burst: bool) -> tuple[list[str], np.ndarray]:
        names = self.cfg.class_names()
        weights = np.array([self.cfg.slice_classes[n].arrival_weight for n in names], dtype=float)
        if regime.lower() == "high":
            for i, n in enumerate(names):
                if n in ("eMBB", "URLLC"):
                    weights[i] *= 1.25
                if n == "mMTC":
                    weights[i] *= 0.75
        if burst:
            for i, n in enumerate(names):
                if n == "Elastic video":
                    weights[i] *= 1.6
        return names, weights / weights.sum()

    def remove_expired(self, t: int) -> None:
        self.active_slices = [s for s in self.active_slices if s.end_t > t]

    def reserved_load(self) -> np.ndarray:
        if not self.active_slices:
            return np.zeros(3, dtype=float)
        return np.sum([s.request.demand for s in self.active_slices], axis=0)

    def actual_utilisation(self, t: int) -> np.ndarray:
        if not self.active_slices:
            return np.zeros(3, dtype=float)
        return np.sum([s.actual_utilisation(t) for s in self.active_slices], axis=0)

    def forecast_active_utilisation(self, t: int) -> np.ndarray:
        if not self.active_slices:
            return np.zeros(3, dtype=float)
        mode = self.cfg.forecast_mode.lower()
        if mode == "no_forecast":
            return self.actual_utilisation(t)
        if mode == "oracle":
            return np.sum([s.true_future_utilisation(t, self.cfg.horizon) for s in self.active_slices], axis=0)
        if mode == "synthetic":
            true = np.sum([s.true_future_utilisation(t, self.cfg.horizon) for s in self.active_slices], axis=0)
            noise = self.rng.normal(self.cfg.synthetic_forecast_bias, self.cfg.synthetic_forecast_std, size=3)
            return np.clip(true * (1.0 + noise), 0.0, None)
        utils = []
        missing_histories = []
        missing_slices = []
        for s in self.active_slices:
            off = s.offset(t)
            if s.forecast_factors is not None and off < len(s.forecast_factors):
                factor = s.forecast_factors[off]
                utils.append(s.request.demand * s.request.eta * utilisation_multiplier(factor))
            else:
                missing_histories.append(s.history_features(t, self.cfg.lookback_window))
                missing_slices.append(s)
        if missing_histories:
            pred_factors = self.forecaster.batch_forecast(missing_histories)
            for s, factor in zip(missing_slices, pred_factors):
                utils.append(s.request.demand * s.request.eta * utilisation_multiplier(factor))
        return np.sum(utils, axis=0) if utils else np.zeros(3, dtype=float)

    def forecast_buffer(self, capacity: np.ndarray) -> np.ndarray:
        return self.cfg.forecast_buffer_kappa * self.cfg.sigma_min_fraction * capacity

    def recent_rates(self, n: int = 24) -> tuple[float, float]:
        if not self.epoch_records:
            return 0.0, 0.0
        recent = self.epoch_records[-n:]
        sla = float(np.mean([r["sla_violation"] for r in recent])) if recent else 0.0
        req = sum(r["arrivals"] for r in recent)
        rej = sum(r["rejected"] for r in recent)
        rejection = float(rej / req) if req > 0 else 0.0
        return sla, rejection

    def build_context(self, t: int, capacity: np.ndarray) -> StepContext:
        reserved = self.reserved_load()
        actual = self.actual_utilisation(t)
        forecast = self.forecast_active_utilisation(t)
        buffer = self.forecast_buffer(capacity)
        regime, burst, _ = self.current_regime(t)
        recent_sla, recent_rejection = self.recent_rates()
        return StepContext(
            t=t,
            capacity=capacity,
            reserved_load=reserved,
            actual_utilisation=actual,
            forecast_utilisation=forecast,
            buffered_forecast=forecast + buffer,
            forecast_buffer=buffer,
            recent_sla_rate=recent_sla,
            recent_rejection_rate=recent_rejection,
            demand_regime=regime,
            is_burst=burst,
        )

    def check_feasibility(self, context: StepContext, req: SliceRequest, opf: float, use_forecast: bool = True) -> tuple[bool, bool, bool]:
        """General nominal-OPF + forecast-aware feasibility check (eqs. 19, 21).

        When ``use_forecast`` is false, this method applies nominal OPF
        feasibility only. The conservative baseline is handled separately by
        ``check_conservative_feasibility`` so that it follows the manuscript's
        strict reservation condition without an actual-utilisation or forecast
        term.
        """
        opf_vec = np.full(3, float(opf))
        nominal_ok = np.all(context.reserved_load + req.demand <= opf_vec * context.capacity + 1e-9)
        if use_forecast:
            mode = self.cfg.forecast_mode.lower()
            if mode == "no_forecast":
                # Reactive no-forecast baseline: use current measured utilisation
                # as the best available estimate of near-future utilisation.
                forecast_ok = np.all(context.actual_utilisation + req.eta * req.demand <= context.capacity + 1e-9)
            else:
                forecast_ok = np.all(context.buffered_forecast + req.eta * req.demand <= context.capacity + 1e-9)
        else:
            # Nominal-only feasibility, used by the conservative baseline and
            # explicit nominal-only ablations.
            forecast_ok = True
        return bool(nominal_ok and forecast_ok), bool(nominal_ok), bool(forecast_ok)

    def check_conservative_feasibility(self, context: StepContext, req: SliceRequest) -> bool:
        """Conservative-reservation feasibility, eq. (10) / eq. (32).

        R_r(t) + d_{i,r} <= C_r for all r in R. No actual-utilisation term and
        no forecast term. This is intentionally independent of ``cfg.forecast_mode``
        so the conservative baseline cannot silently pick up a different
        feasibility mechanism when the global forecast mode changes.
        """
        return bool(np.all(context.reserved_load + req.demand <= context.capacity + 1e-9))

    def _precompute_forecast_factors(self, profile: ProfileSegment) -> np.ndarray | None:
        if self.cfg.forecast_mode.lower() != "xgboost":
            return None
        histories = []
        features = profile.features
        for off in range(len(profile.frame)):
            start = max(0, off - self.cfg.lookback_window + 1)
            histories.append(features[start:off + 1])
        return self.forecaster.batch_forecast(histories)

    def admit(self, req: SliceRequest, t: int) -> ActiveSlice:
        seg_len = req.lifetime + self.cfg.lookback_window + self.cfg.horizon + 2
        profile_rng = np.random.default_rng(req.profile_seed)
        split_value = None if self.cfg.workload_data_split.lower() == "all" else self.cfg.workload_data_split
        profile, used_fallback = self.profile_store.sample_segment(
            profile_rng, length=seg_len, min_length=seg_len,
            split_column=self.cfg.workload_split_column, split_value=split_value,
            allow_fallback=not self.cfg.strict_workload_split,
        )
        if used_fallback:
            self._last_scenario_used_full_profile_fallback = True
        forecast_factors = self._precompute_forecast_factors(profile)
        active = ActiveSlice(request=req, start_t=t, end_t=t + req.lifetime, profile=profile, forecast_factors=forecast_factors)
        self.active_slices.append(active)
        return active

    def compute_penalties(self, capacity: np.ndarray, rejected_requests: list[SliceRequest], admission_revenue: float) -> dict[str, float | np.ndarray | bool]:
        actual = self.actual_utilisation(self._current_t)
        reserved = self.reserved_load()
        overload = np.maximum(0.0, (actual - capacity) / np.maximum(capacity, 1e-9))
        sla_penalty = float(np.sum(self.cfg.sla_penalty_array() * overload))
        sla_violation = bool(np.any(actual > capacity + 1e-9))
        rejection_penalty = float(sum(
            self.cfg.rejection_penalty_lambda * r.revenue * r.rejection_weight
            for r in rejected_requests
        ))
        overbook = np.maximum(0.0, (reserved - capacity) / np.maximum(capacity, 1e-9))
        overbooking_penalty = float(np.sum(self.cfg.overbooking_penalty_array() * overbook))
        safe_bonus = float(self.cfg.sla_safe_bonus if not sla_violation else 0.0)
        reward = float(admission_revenue - sla_penalty - rejection_penalty - overbooking_penalty + safe_bonus)
        return {
            "actual": actual,
            "reserved": reserved,
            "sla_penalty": sla_penalty,
            "sla_violation": sla_violation,
            "rejection_penalty": rejection_penalty,
            "overbooking_penalty": overbooking_penalty,
            "safe_bonus": safe_bonus,
            "reward": reward,
        }

    def state_key(self, context: StepContext, req: SliceRequest | None = None, opf: float | None = None) -> tuple:
        """Compact tabular-RL state.

        The first simulator version used three separate buckets for each
        resource, which created many rarely visited states.  For tabular
        learning this caused OPF-aware Q-learning to fall back too often.  This
        compact state keeps the operational signals that matter for admission
        control while improving generalisation across Monte Carlo scenarios.
        """
        util_ratio = context.actual_utilisation / np.maximum(context.capacity, 1e-9)
        reserved_ratio = context.reserved_load / np.maximum(context.capacity, 1e-9)
        forecast_ratio = context.buffered_forecast / np.maximum(context.capacity, 1e-9)
        headroom_ratio = 1.0 - forecast_ratio

        def bucket(v: float, cuts: list[float]) -> int:
            return int(np.digitize([float(v)], cuts)[0])

        util_state = bucket(np.max(util_ratio), [0.25, 0.50, 0.75, 0.95, 1.05])
        reserved_state = bucket(np.max(reserved_ratio), [0.50, 0.90, 1.20, 1.50, 2.00])
        head_state = bucket(np.min(headroom_ratio), [-0.10, 0.00, 0.05, 0.15, 0.30, 0.50])
        sla_state = bucket(context.recent_sla_rate, [0.0, 0.005, self.cfg.sla_violation_target, 0.05])
        rej_state = bucket(context.recent_rejection_rate, [0.25, 0.50, 0.70, 0.85])
        burst_state = int(context.is_burst)
        regime_state = {"low": 0, "medium": 1, "high": 2}.get(context.demand_regime.lower(), 1)
        cls_state = req.cls_name if req is not None else "none"
        opf_state = round(float(opf if opf is not None else 1.0), 2)
        return (util_state, reserved_state, head_state, sla_state, rej_state, burst_state, regime_state, cls_state, opf_state)

    def run_episode(self, policy, seed: int | None = None, training: bool = False, scenario: Scenario | None = None) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        self.reset(seed=seed if seed is not None else self.seed, scenario=scenario)
        # Deterministic/adaptive policies should reset at the start of an
        # evaluation episode. During Q-learning training, do not reset epsilon
        # every episode; otherwise exploration never decays.
        if hasattr(policy, "reset") and not training:
            policy.reset()

        assert self._scenario is not None
        for t in range(self.cfg.decision_epochs):
            self._current_t = t
            self.remove_expired(t)
            capacity = self._scenario.capacities[t].copy()
            self.degradation_factor = self._scenario.degradation_factors[t]
            context = self.build_context(t, capacity)
            requests = list(self._scenario.requests_by_t[t])
            accepted: list[SliceRequest] = []
            rejected: list[SliceRequest] = []
            admission_revenue = 0.0
            selected_opfs: list[float] = []
            q_transitions = []

            for req in requests:
                if hasattr(policy, "state_key"):
                    state_before = policy.state_key(self, context, req)
                else:
                    state_before = self.state_key(context, req=req, opf=getattr(policy, "current_opf", 1.0))
                decision = policy.decide(self, context, req, training=training)
                feasible, nominal_ok, forecast_ok = self.check_feasibility(
                    context, req, decision.selected_opf, use_forecast=decision.use_forecast
                )
                accept = bool(decision.accept and feasible)
                reason = decision.reason
                if decision.accept and not feasible:
                    if not nominal_ok and forecast_ok:
                        reason = "blocked_nominal_opf"
                    elif nominal_ok and not forecast_ok:
                        reason = "blocked_forecast_sla"
                    elif not nominal_ok and not forecast_ok:
                        reason = "blocked_nominal_and_forecast"
                    else:
                        reason = "blocked_by_feasibility"
                elif not decision.accept:
                    reason = reason or "policy_reject"

                if accept:
                    active = self.admit(req, t)
                    accepted.append(req)
                    admission_revenue += req.revenue
                    selected_opfs.append(decision.selected_opf)
                    # Incrementally update context so the next request in the
                    # same epoch sees the new slice without repeatedly
                    # recomputing forecasts for all active slices.
                    added_actual = active.actual_utilisation(t)
                    if decision.use_forecast and self.cfg.forecast_mode.lower() not in ("no_forecast",):
                        if self.cfg.forecast_mode.lower() == "oracle":
                            added_forecast = active.true_future_utilisation(t, self.cfg.horizon)
                        elif self.cfg.forecast_mode.lower() == "synthetic":
                            added_forecast = added_actual
                        else:
                            added_forecast = active.forecast_utilisation(t, self.forecaster, self.cfg.lookback_window)
                    else:
                        added_forecast = added_actual
                    context.reserved_load = context.reserved_load + req.demand
                    context.actual_utilisation = context.actual_utilisation + added_actual
                    context.forecast_utilisation = context.forecast_utilisation + added_forecast
                    context.buffered_forecast = context.forecast_utilisation + context.forecast_buffer
                else:
                    rejected.append(req)
                    selected_opfs.append(decision.selected_opf)

                if hasattr(policy, "state_key"):
                    state_after = policy.state_key(self, context, req)
                else:
                    state_after = self.state_key(context, req=req, opf=decision.selected_opf)
                q_transitions.append((state_before, decision, state_after))
                self.decision_records.append({
                    "seed": self.seed,
                    "t": t,
                    "request_id": req.request_id,
                    "class": req.cls_name,
                    "demand_cpu": req.demand[0],
                    "demand_mem": req.demand[1],
                    "demand_bw": req.demand[2],
                    "revenue": req.revenue,
                    "lifetime": req.lifetime,
                    "selected_opf": decision.selected_opf,
                    "use_forecast": int(decision.use_forecast),
                    "accepted": accept,
                    "nominal_ok": nominal_ok,
                    "forecast_ok": forecast_ok,
                    "reason": reason,
                })

            pen = self.compute_penalties(capacity, rejected, admission_revenue)
            reward = float(pen["reward"])
            if hasattr(policy, "learn") and training:
                for s_before, decision, s_after in q_transitions:
                    policy.learn(s_before, decision, reward, s_after)
            if hasattr(policy, "after_epoch"):
                policy.after_epoch(self, context, reward, bool(pen["sla_violation"]), len(requests), len(rejected))

            mean_opf = float(np.mean(selected_opfs)) if selected_opfs else float(getattr(policy, "current_opf", 1.0))
            actual = np.asarray(pen["actual"], dtype=float)
            reserved = np.asarray(pen["reserved"], dtype=float)
            record = {
                "seed": self.seed,
                "t": t,
                "policy": policy.name,
                "arrivals": len(requests),
                "accepted": len(accepted),
                "rejected": len(rejected),
                "active_slices": len(self.active_slices),
                "selected_opf": mean_opf,
                "capacity_cpu": capacity[0],
                "capacity_mem": capacity[1],
                "capacity_bw": capacity[2],
                "reserved_cpu": reserved[0],
                "reserved_mem": reserved[1],
                "reserved_bw": reserved[2],
                "actual_cpu": actual[0],
                "actual_mem": actual[1],
                "actual_bw": actual[2],
                "forecast_cpu": context.forecast_utilisation[0],
                "forecast_mem": context.forecast_utilisation[1],
                "forecast_bw": context.forecast_utilisation[2],
                "buffer_cpu": context.forecast_buffer[0],
                "buffer_mem": context.forecast_buffer[1],
                "buffer_bw": context.forecast_buffer[2],
                "admission_revenue": admission_revenue,
                "sla_penalty": float(pen["sla_penalty"]),
                "rejection_penalty": float(pen["rejection_penalty"]),
                "overbooking_penalty": float(pen["overbooking_penalty"]),
                "safe_bonus": float(pen["safe_bonus"]),
                "net_profit": reward,
                "sla_violation": int(bool(pen["sla_violation"])),
                "demand_regime": context.demand_regime,
                "is_burst": int(context.is_burst),
                "degradation_factor": self.degradation_factor,
            }
            self.epoch_records.append(record)

        ts = pd.DataFrame(self.epoch_records)
        decisions = pd.DataFrame(self.decision_records)
        summary = self.summarise_episode(ts, decisions, policy.name)
        return ts, decisions, summary

    def summarise_episode(self, ts: pd.DataFrame, decisions: pd.DataFrame, policy_name: str) -> dict[str, Any]:
        total_requests = int(ts["arrivals"].sum()) if len(ts) else 0
        total_admitted = int(ts["accepted"].sum()) if len(ts) else 0
        total_rejected = int(ts["rejected"].sum()) if len(ts) else 0
        accepted_decisions = decisions[decisions["accepted"].astype(bool)] if len(decisions) and "accepted" in decisions.columns else pd.DataFrame()
        mean_selected_opf_all = float(ts["selected_opf"].mean()) if len(ts) and "selected_opf" in ts.columns else np.nan
        mean_selected_opf_accepted = (
            float(accepted_decisions["selected_opf"].astype(float).mean())
            if len(accepted_decisions) else mean_selected_opf_all
        )
        summary = {
            "policy": policy_name,
            "seed": self.seed,
            "epochs": int(len(ts)),
            "total_requests": total_requests,
            "total_admitted": total_admitted,
            "total_rejected": total_rejected,
            "admission_ratio": total_admitted / total_requests if total_requests else 0.0,
            "rejection_ratio": total_rejected / total_requests if total_requests else 0.0,
            "mean_cpu_utilisation": float((ts["actual_cpu"] / ts["capacity_cpu"]).mean()),
            "mean_mem_utilisation": float((ts["actual_mem"] / ts["capacity_mem"]).mean()),
            "mean_bw_utilisation": float((ts["actual_bw"] / ts["capacity_bw"]).mean()),
            "p95_cpu_utilisation": float((ts["actual_cpu"] / ts["capacity_cpu"]).quantile(0.95)),
            "p95_mem_utilisation": float((ts["actual_mem"] / ts["capacity_mem"]).quantile(0.95)),
            "p95_bw_utilisation": float((ts["actual_bw"] / ts["capacity_bw"]).quantile(0.95)),
            "mean_reserved_cpu_ratio": float((ts["reserved_cpu"] / ts["capacity_cpu"]).mean()),
            "mean_reserved_mem_ratio": float((ts["reserved_mem"] / ts["capacity_mem"]).mean()),
            "mean_reserved_bw_ratio": float((ts["reserved_bw"] / ts["capacity_bw"]).mean()),
            "mean_selected_opf": mean_selected_opf_accepted,
            "mean_selected_opf_accepted": mean_selected_opf_accepted,
            "mean_selected_opf_all_decisions": mean_selected_opf_all,
            "sla_violation_rate": float(ts["sla_violation"].mean()),
            "total_admission_revenue": float(ts["admission_revenue"].sum()),
            "total_sla_penalty": float(ts["sla_penalty"].sum()),
            "total_rejection_penalty": float(ts["rejection_penalty"].sum()),
            "total_overbooking_penalty": float(ts["overbooking_penalty"].sum()),
            "total_safe_bonus": float(ts["safe_bonus"].sum()),
            "net_profit": float(ts["net_profit"].sum()),
            "degradation_events": int(self.degradation_events),
        }
        if len(decisions):
            rej = decisions[~decisions["accepted"].astype(bool)].copy()
            nominal_ok = rej["nominal_ok"].astype(bool)
            forecast_ok = rej["forecast_ok"].astype(bool)
            summary["blocked_nominal_count"] = int(((~nominal_ok) & forecast_ok).sum())
            summary["blocked_forecast_count"] = int((nominal_ok & (~forecast_ok)).sum())
            summary["blocked_both_count"] = int(((~nominal_ok) & (~forecast_ok)).sum())
            # Pure policy rejection means the request was technically feasible
            # under the selected OPF/forecast mode but the learned policy chose
            # to reject it.
            summary["policy_reject_count"] = int((nominal_ok & forecast_ok).sum())
            summary["q_unseen_reject_count"] = int(rej["reason"].astype(str).str.contains("unseen_state_reject", regex=False).sum())
            summary["q_unseen_fallback_accept_count"] = int(
                decisions[decisions["accepted"].astype(bool)]["reason"].astype(str).str.contains("unseen_state", regex=False).sum()
            )
        else:
            summary["blocked_nominal_count"] = 0
            summary["blocked_forecast_count"] = 0
            summary["blocked_both_count"] = 0
            summary["policy_reject_count"] = 0
            summary["q_unseen_reject_count"] = 0
            summary["q_unseen_fallback_accept_count"] = 0
        summary["sla_feasible"] = bool(summary["sla_violation_rate"] <= self.cfg.sla_violation_target)
        return summary
