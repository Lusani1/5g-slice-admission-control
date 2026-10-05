"""Admission-control policies for the slice overbooking simulator."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

from .environment import StepContext, SliceRequest, DecisionResult


class BasePolicy:
    name: str = "base"

    def reset(self) -> None:
        pass

    def state_key(self, env, context: StepContext, req: SliceRequest):
        return env.state_key(context, req=req, opf=getattr(self, "current_opf", 1.0))

    def decide(self, env, context: StepContext, req: SliceRequest, training: bool = False) -> DecisionResult:
        raise NotImplementedError


@dataclass
class ConservativePolicy(BasePolicy):
    """No-overbooking baseline with nominal reservation feasibility only."""
    name: str = "conservative"

    @property
    def current_opf(self) -> float:
        return 1.0

    def decide(self, env, context: StepContext, req: SliceRequest, training: bool = False) -> DecisionResult:
        feasible = env.check_conservative_feasibility(context, req)
        return DecisionResult(accept=feasible, selected_opf=1.0, reason="conservative", use_forecast=False)


@dataclass
class StaticOPFPolicy(BasePolicy):
    opf: float = 1.10
    use_forecast: bool = True

    @property
    def name(self) -> str:
        suffix = "forecast" if self.use_forecast else "no_forecast"
        return f"static_opf_{self.opf:.2f}_{suffix}"

    @property
    def current_opf(self) -> float:
        return self.opf

    def decide(self, env, context: StepContext, req: SliceRequest, training: bool = False) -> DecisionResult:
        feasible, _, _ = env.check_feasibility(context, req, opf=self.opf, use_forecast=self.use_forecast)
        return DecisionResult(accept=feasible, selected_opf=self.opf, reason="static_opf", use_forecast=self.use_forecast)


@dataclass
class AdaptiveHeuristicPolicy(BasePolicy):
    """Forecast-assisted adaptive OPF policy.

    The policy increases OPF when rejection is high and forecast headroom is safe,
    and reduces OPF after SLA violations or when forecast headroom becomes small.
    """
    initial_opf: float = 1.20
    min_opf: float = 1.00
    max_opf: float = 2.00
    step: float = 0.05
    current_opf: float = field(init=False)

    @property
    def name(self) -> str:
        return "adaptive_heuristic"

    def reset(self) -> None:
        self.current_opf = self.initial_opf

    def __post_init__(self) -> None:
        self.reset()

    def decide(self, env, context: StepContext, req: SliceRequest, training: bool = False) -> DecisionResult:
        feasible, _, _ = env.check_feasibility(context, req, opf=self.current_opf, use_forecast=True)
        return DecisionResult(accept=feasible, selected_opf=self.current_opf, reason="adaptive_heuristic", use_forecast=True)

    def after_epoch(self, env, context: StepContext, reward: float, sla_violation: bool, arrivals: int, rejected: int) -> None:
        rejection_rate = rejected / arrivals if arrivals else 0.0
        forecast_ratio = context.buffered_forecast / np.maximum(context.capacity, 1e-9)
        min_headroom = float(np.min(1.0 - forecast_ratio))
        if sla_violation or context.recent_sla_rate > env.cfg.adaptive_high_sla_threshold or min_headroom < env.cfg.adaptive_low_headroom_threshold:
            self.current_opf = max(self.min_opf, self.current_opf - self.step)
        elif rejection_rate > env.cfg.adaptive_rejection_threshold and min_headroom > 0.18:
            self.current_opf = min(self.max_opf, self.current_opf + self.step)
        self.current_opf = round(float(self.current_opf), 2)


@dataclass
class QLearningPolicy(BasePolicy):
    fixed_opf: float = 1.00
    learning_rate: float = 0.12
    discount_factor: float = 0.95
    epsilon_initial: float = 0.40
    epsilon_min: float = 0.03
    epsilon_decay: float = 0.992
    min_visits_for_greedy: int = 3
    allow_unseen_state_fallback: bool = False
    name: str = "q_learning"
    q_table: defaultdict = field(default_factory=lambda: defaultdict(lambda: np.zeros(2, dtype=float)))
    visit_counts: defaultdict = field(default_factory=lambda: defaultdict(lambda: np.zeros(2, dtype=int)))
    epsilon: float = field(init=False)
    actions: tuple[str, str] = ("reject", "accept")

    @property
    def current_opf(self) -> float:
        return self.fixed_opf

    def reset(self) -> None:
        self.epsilon = self.epsilon_initial

    def __post_init__(self) -> None:
        self.reset()

    def _select_action_idx(self, state, training: bool, rng) -> int:
        if training and rng.random() < self.epsilon:
            return int(rng.integers(0, len(self.actions)))
        q = self.q_table[state]
        visits = self.visit_counts[state]
        if visits.sum() < self.min_visits_for_greedy and training:
            return int(rng.integers(0, len(self.actions)))
        return int(np.argmax(q))

    def decide(self, env, context: StepContext, req: SliceRequest, training: bool = False) -> DecisionResult:
        state = env.state_key(context, req=req, opf=self.fixed_opf)
        if (not training) and self.visit_counts[state].sum() == 0 and not self.allow_unseen_state_fallback:
            return DecisionResult(accept=False, selected_opf=self.fixed_opf, reason="q_unseen_state_reject", use_forecast=True)
        if (not training) and self.visit_counts[state].sum() == 0 and self.allow_unseen_state_fallback:
            feasible, _, _ = env.check_feasibility(context, req, opf=self.fixed_opf, use_forecast=True)
            return DecisionResult(accept=feasible, selected_opf=self.fixed_opf, reason="q_unseen_state_feasibility_fallback", use_forecast=True)
        idx = self._select_action_idx(state, training=training, rng=env.rng)
        action = self.actions[idx]
        return DecisionResult(accept=(action == "accept"), selected_opf=self.fixed_opf, reason=f"q_action_{action}", use_forecast=True)

    def learn(self, state, decision: DecisionResult, reward: float, next_state) -> None:
        idx = 1 if decision.accept else 0
        self.visit_counts[state][idx] += 1
        best_next = float(np.max(self.q_table[next_state]))
        old = self.q_table[state][idx]
        target = reward + self.discount_factor * best_next
        self.q_table[state][idx] = old + self.learning_rate * (target - old)

    def after_epoch(self, env, context: StepContext, reward: float, sla_violation: bool, arrivals: int, rejected: int) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


@dataclass
class OPFAwareQLearningPolicy(QLearningPolicy):
    opf_set: tuple[float, ...] = (1.10, 1.20, 1.30, 1.40, 1.50, 1.75, 2.00, 2.25, 2.50)
    name: str = "opf_aware_q_learning"

    def __post_init__(self) -> None:
        self.actions = ("reject",) + tuple(f"accept_{opf:.2f}" for opf in self.opf_set)
        n_actions = len(self.actions)
        self.q_table = defaultdict(lambda: np.zeros(n_actions, dtype=float))
        self.visit_counts = defaultdict(lambda: np.zeros(n_actions, dtype=int))
        self.reset()

    def _decode_action(self, idx: int) -> tuple[bool, float, str]:
        if idx == 0:
            return False, 1.0, "reject"
        opf = float(self.opf_set[idx - 1])
        return True, opf, f"accept_{opf:.2f}"

    def state_key(self, env, context: StepContext, req: SliceRequest):
        # OPF is an action in this policy, not part of the state.  Keeping it
        # out of the state prevents fragmentation and makes learned OPF choices
        # transferable across feasible OPF levels.
        return env.state_key(context, req=req, opf=1.0)

    def _fallback_decision(self, env, context: StepContext, req: SliceRequest) -> DecisionResult:
        """Operational fallback for genuinely unseen states.

        The previous implementation always selected the highest feasible OPF,
        which made OPF-aware Q-learning indistinguishable from the maximum
        static-OPF policy when many states were unseen.  This fallback is still
        safe, but it is workload-aware: it uses higher OPF only when forecast
        headroom is comfortable and recent SLA exposure is low.
        """
        forecast_ratio = context.buffered_forecast / np.maximum(context.capacity, 1e-9)
        min_headroom = float(np.min(1.0 - forecast_ratio))
        recent_sla = float(context.recent_sla_rate)
        if recent_sla > env.cfg.sla_violation_target or min_headroom < 0.05:
            opf_cap = 1.20
        elif min_headroom < 0.15:
            opf_cap = 1.50
        elif context.is_burst or context.demand_regime.lower() == "high":
            opf_cap = 1.75
        elif context.recent_rejection_rate > env.cfg.adaptive_rejection_threshold and min_headroom > 0.25:
            opf_cap = max(self.opf_set)
        else:
            opf_cap = 2.00
        candidates = [float(o) for o in self.opf_set if float(o) <= opf_cap + 1e-9]
        for opf in sorted(candidates, reverse=True):
            feasible, _, _ = env.check_feasibility(context, req, opf=opf, use_forecast=True)
            if feasible:
                return DecisionResult(accept=True, selected_opf=float(opf), reason="opf_q_unseen_state_safe_fallback", use_forecast=True)
        return DecisionResult(accept=False, selected_opf=1.0, reason="opf_q_unseen_state_reject", use_forecast=True)

    def decide(self, env, context: StepContext, req: SliceRequest, training: bool = False) -> DecisionResult:
        state = self.state_key(env, context, req)
        if (not training) and self.visit_counts[state].sum() == 0 and not self.allow_unseen_state_fallback:
            return DecisionResult(accept=False, selected_opf=1.0, reason="opf_q_unseen_state_reject", use_forecast=True)
        if (not training) and self.visit_counts[state].sum() == 0 and self.allow_unseen_state_fallback:
            return self._fallback_decision(env, context, req)
        idx = self._select_action_idx(state, training=training, rng=env.rng)
        accept, opf, label = self._decode_action(idx)
        return DecisionResult(accept=accept, selected_opf=opf, reason=f"opf_q_action_{label}", use_forecast=True)

    def learn(self, state, decision: DecisionResult, reward: float, next_state) -> None:
        if not decision.accept:
            idx = 0
        else:
            idx = 1 + int(np.argmin(np.abs(np.asarray(self.opf_set) - decision.selected_opf)))
        self.visit_counts[state][idx] += 1
        best_next = float(np.max(self.q_table[next_state]))
        old = self.q_table[state][idx]
        target = reward + self.discount_factor * best_next
        self.q_table[state][idx] = old + self.learning_rate * (target - old)


def make_default_policies(cfg) -> list[BasePolicy]:
    # Put learning policies first to avoid unnecessary deep-copy/garbage-
    # collection overhead after large batches of deterministic-policy results.
    policies: list[BasePolicy] = [
        QLearningPolicy(
            fixed_opf=cfg.q_fixed_opf,
            learning_rate=cfg.learning_rate,
            discount_factor=cfg.discount_factor,
            epsilon_initial=cfg.epsilon_initial,
            epsilon_min=cfg.epsilon_min,
            epsilon_decay=cfg.epsilon_decay,
            min_visits_for_greedy=cfg.min_visits_for_greedy,
            allow_unseen_state_fallback=cfg.allow_unseen_state_fallback,
        ),
        OPFAwareQLearningPolicy(
            fixed_opf=cfg.q_fixed_opf,
            learning_rate=cfg.learning_rate,
            discount_factor=cfg.discount_factor,
            epsilon_initial=cfg.epsilon_initial,
            epsilon_min=cfg.epsilon_min,
            epsilon_decay=cfg.epsilon_decay,
            min_visits_for_greedy=cfg.min_visits_for_greedy,
            allow_unseen_state_fallback=cfg.allow_unseen_state_fallback,
            opf_set=cfg.opf_set,
        ),
        ConservativePolicy(),
    ]
    for opf in cfg.static_opfs:
        policies.append(StaticOPFPolicy(opf=opf, use_forecast=True))
    policies.append(AdaptiveHeuristicPolicy(
        initial_opf=cfg.adaptive_initial_opf,
        min_opf=cfg.adaptive_min_opf,
        max_opf=cfg.adaptive_max_opf,
        step=cfg.adaptive_step,
    ))
    return policies
