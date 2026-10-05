"""Unit tests for equation-critical functions in the admission-control simulator.

Each test uses a small, hand-computed example so the expected answer is
easy to inspect and can be run with:

    cd admission_control_xgboost_simulator_with_sensitivity
    python3 -m pytest tests/test_equations.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from admission_sim.config import SimulatorConfig
from admission_sim.environment import AdmissionControlSimulator, StepContext, SliceRequest
from admission_sim.policies import ConservativePolicy, StaticOPFPolicy, QLearningPolicy, OPFAwareQLearningPolicy


def make_context(capacity, reserved, actual, forecast, buffer):
    capacity = np.asarray(capacity, dtype=float)
    reserved = np.asarray(reserved, dtype=float)
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    buffer = np.asarray(buffer, dtype=float)
    return StepContext(
        t=0, capacity=capacity, reserved_load=reserved, actual_utilisation=actual,
        forecast_utilisation=forecast, buffered_forecast=forecast + buffer, forecast_buffer=buffer,
        recent_sla_rate=0.0, recent_rejection_rate=0.0, demand_regime="medium", is_burst=False,
    )


def make_request(demand, eta=(1.0, 1.0, 1.0), revenue=10.0, rejection_weight=1.0, cls_name="eMBB"):
    return SliceRequest(
        request_id=0, t=0, cls_name=cls_name, demand=np.asarray(demand, dtype=float),
        lifetime=10, revenue=revenue, eta=np.asarray(eta, dtype=float),
        sla_weight=1.0, rejection_weight=rejection_weight, profile_seed=0,
    )


class DummyProfileStore:
    """Minimal stand-in; not used by the unit-level checks below."""
    pass


def make_sim(cfg=None):
    cfg = cfg or SimulatorConfig()
    sim = AdmissionControlSimulator.__new__(AdmissionControlSimulator)
    sim.cfg = cfg
    sim.rng = np.random.default_rng(0)
    return sim


# ---------------------------------------------------------------------------
# 1. Conservative admission feasibility (eq. 10 / 32): R_r(t)+d_i,r <= C_r
# ---------------------------------------------------------------------------
def test_conservative_feasibility_accept():
    sim = make_sim()
    # Capacity 100 each; already reserved 80; new request demands 15 -> 95 <= 100: accept.
    ctx = make_context(capacity=[100, 100, 100], reserved=[80, 80, 80],
                       actual=[10, 10, 10], forecast=[10, 10, 10], buffer=[0, 0, 0])
    req = make_request(demand=[15, 15, 15])
    assert sim.check_conservative_feasibility(ctx, req) is True


def test_conservative_feasibility_reject_on_nominal_only():
    sim = make_sim()
    # 80 + 25 = 105 > 100 -> reject, regardless of low actual utilisation.
    ctx = make_context(capacity=[100, 100, 100], reserved=[80, 80, 80],
                       actual=[5, 5, 5], forecast=[5, 5, 5], buffer=[0, 0, 0])
    req = make_request(demand=[25, 25, 25])
    assert sim.check_conservative_feasibility(ctx, req) is False


def test_conservative_policy_ignores_actual_utilisation_gate():
    """Regression test for conservative-feasibility behaviour.

    The conservative policy must not additionally require
    actual_utilisation + eta*d <= C, which could reject nominally-feasible
    requests. Construct a case where the nominal condition is satisfied but
    a (now-removed) actual-utilisation condition would not be, and confirm
    the policy still accepts.
    """
    sim = make_sim()
    cfg = sim.cfg
    policy = ConservativePolicy()
    # Nominal: 50 + 10 = 60 <= 100 (feasible).
    # Actual utilisation is deliberately set above capacity so that the old,
    # buggy "actual_utilisation + eta*d <= C" branch would have rejected.
    ctx = make_context(capacity=[100, 100, 100], reserved=[50, 50, 50],
                       actual=[99, 99, 99], forecast=[99, 99, 99], buffer=[0, 0, 0])
    req = make_request(demand=[10, 10, 10], eta=[1.0, 1.0, 1.0])
    decision = policy.decide(sim, ctx, req)
    assert decision.accept is True, (
        "Conservative baseline must depend only on nominal reservation (eq. 32), "
        "not on current actual utilisation."
    )


# ---------------------------------------------------------------------------
# 2. Static OPF nominal feasibility (eq. 19): R_r(t)+d_i,r <= phi * C_r
# ---------------------------------------------------------------------------
def test_static_opf_nominal_feasibility():
    cfg = SimulatorConfig()
    cfg.forecast_mode = "xgboost"
    sim = make_sim(cfg)
    # Capacity 100, reserved 100 already (at nominal limit). New demand 20.
    # phi=1.10 -> effective cap 110. 100+20=120 > 110 -> nominal infeasible.
    ctx = make_context(capacity=[100, 100, 100], reserved=[100, 100, 100],
                       actual=[50, 50, 50], forecast=[50, 50, 50], buffer=[0, 0, 0])
    req = make_request(demand=[20, 20, 20])
    feasible, nominal_ok, forecast_ok = sim.check_feasibility(ctx, req, opf=1.10, use_forecast=False)
    assert nominal_ok is False
    # phi=1.30 -> effective cap 130. 100+20=120 <= 130 -> nominal feasible.
    feasible, nominal_ok, forecast_ok = sim.check_feasibility(ctx, req, opf=1.30, use_forecast=False)
    assert nominal_ok is True


# ---------------------------------------------------------------------------
# 3. Forecast-aware feasibility (eq. 21): U_tilde(t+h) + eta*d <= C
# ---------------------------------------------------------------------------
def test_forecast_aware_feasibility():
    cfg = SimulatorConfig()
    cfg.forecast_mode = "xgboost"
    sim = make_sim(cfg)
    # Buffered forecast already at 90 (=forecast 85 + buffer 5). New request
    # contributes eta*d = 0.5*20 = 10 -> 90+10=100 <= 100: feasible (boundary).
    ctx = make_context(capacity=[100, 100, 100], reserved=[60, 60, 60],
                       actual=[60, 60, 60], forecast=[85, 85, 85], buffer=[5, 5, 5])
    req = make_request(demand=[20, 20, 20], eta=[0.5, 0.5, 0.5])
    _, _, forecast_ok = sim.check_feasibility(ctx, req, opf=2.00, use_forecast=True)
    assert forecast_ok is True
    # Increase demand contribution beyond headroom -> infeasible.
    req2 = make_request(demand=[40, 40, 40], eta=[0.5, 0.5, 0.5])
    _, _, forecast_ok2 = sim.check_feasibility(ctx, req2, opf=2.00, use_forecast=True)
    assert forecast_ok2 is False


def test_no_forecast_mode_does_not_impose_hidden_condition():
    """Regression test for nominal-only feasibility: when use_forecast is
    False, the SLA feasibility condition should simply be skipped (True),
    not silently replaced by an actual-utilisation gate."""
    cfg = SimulatorConfig()
    cfg.forecast_mode = "xgboost"
    sim = make_sim(cfg)
    ctx = make_context(capacity=[100, 100, 100], reserved=[10, 10, 10],
                       actual=[95, 95, 95], forecast=[95, 95, 95], buffer=[0, 0, 0])
    req = make_request(demand=[5, 5, 5])
    _, nominal_ok, forecast_ok = sim.check_feasibility(ctx, req, opf=1.00, use_forecast=False)
    assert nominal_ok is True
    assert forecast_ok is True  # not gated on the (very high) actual utilisation


# ---------------------------------------------------------------------------
# 4. SLA violation indicator (eq. 14/15): v_r(t) = 1 if U_r(t) > C_r
# ---------------------------------------------------------------------------
def test_sla_violation_indicator():
    cfg = SimulatorConfig()
    sim = make_sim(cfg)
    sim.active_slices = []
    sim._current_t = 0
    capacity = np.array([100.0, 100.0, 100.0])

    class _FakeSlice:
        def __init__(self, util, demand=(0, 0, 0)):
            self._util = np.asarray(util, dtype=float)
            self.request = type("R", (), {"demand": np.asarray(demand, dtype=float)})()
        def actual_utilisation(self, t):
            return self._util

    sim.active_slices = [_FakeSlice([60, 60, 60]), _FakeSlice([50, 30, 30])]  # cpu total 110 > 100
    pen = sim.compute_penalties(capacity, rejected_requests=[], admission_revenue=0.0)
    assert pen["sla_violation"] is True

    sim.active_slices = [_FakeSlice([40, 40, 40]), _FakeSlice([30, 30, 30])]  # all <= 100
    pen2 = sim.compute_penalties(capacity, rejected_requests=[], admission_revenue=0.0)
    assert pen2["sla_violation"] is False


# ---------------------------------------------------------------------------
# 5. SLA penalty (eq. 16): P_sla = sum_r alpha_r * max(0, (U_r-C_r)/C_r)
# ---------------------------------------------------------------------------
def test_sla_penalty_hand_computed():
    cfg = SimulatorConfig()
    cfg.sla_penalty_weights = (70.0, 55.0, 65.0)
    sim = make_sim(cfg)

    class _FakeSlice:
        def __init__(self, util, demand=(0, 0, 0)):
            self._util = np.asarray(util, dtype=float)
            self.request = type("R", (), {"demand": np.asarray(demand, dtype=float)})()
        def actual_utilisation(self, t):
            return self._util

    # capacity 100 each; utilisation cpu=120 (20% over), mem=80 (no overload), bw=130 (30% over)
    sim.active_slices = [_FakeSlice([120, 80, 130])]
    sim._current_t = 0
    capacity = np.array([100.0, 100.0, 100.0])
    pen = sim.compute_penalties(capacity, rejected_requests=[], admission_revenue=0.0)
    expected = 70.0 * 0.20 + 55.0 * 0.0 + 65.0 * 0.30
    assert pen["sla_penalty"] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 6. Overbooking penalty (eq. 22): P_ob = sum_r gamma_r * max(0, (R_r-C_r)/C_r)
# ---------------------------------------------------------------------------
def test_overbooking_penalty_hand_computed():
    cfg = SimulatorConfig()
    cfg.overbooking_penalty_weights = (4.0, 3.0, 4.0)
    sim = make_sim(cfg)

    class _FakeSlice:
        def __init__(self, util, demand):
            self._util = np.asarray(util, dtype=float)
            self.request = type("R", (), {"demand": np.asarray(demand, dtype=float)})()
        def actual_utilisation(self, t):
            return self._util

    # reserved (nominal demand) cpu=150 (50% over), mem=90 (no overload), bw=110 (10% over)
    sim.active_slices = [_FakeSlice(util=[50, 50, 50], demand=[150, 90, 110])]
    sim._current_t = 0
    capacity = np.array([100.0, 100.0, 100.0])
    pen = sim.compute_penalties(capacity, rejected_requests=[], admission_revenue=0.0)
    expected = 4.0 * 0.50 + 3.0 * 0.0 + 4.0 * 0.10
    assert pen["overbooking_penalty"] == pytest.approx(expected, rel=1e-6)


# ---------------------------------------------------------------------------
# 7. Net reward (eq. 23): g(t) = R_adm - P_sla - P_rej - P_ob + B_safe
# ---------------------------------------------------------------------------
def test_net_reward_with_sla_safe_bonus():
    cfg = SimulatorConfig()
    cfg.sla_penalty_weights = (70.0, 55.0, 65.0)
    cfg.overbooking_penalty_weights = (4.0, 3.0, 4.0)
    cfg.sla_safe_bonus = 2.0
    cfg.rejection_penalty_lambda = 0.10
    sim = make_sim(cfg)

    class _FakeSlice:
        def __init__(self, util, demand):
            self._util = np.asarray(util, dtype=float)
            self.request = type("R", (), {"demand": np.asarray(demand, dtype=float)})()
        def actual_utilisation(self, t):
            return self._util

    # No SLA violation, no overbooking -> reward = revenue - 0 - rejection_penalty - 0 + bonus
    sim.active_slices = [_FakeSlice(util=[40, 40, 40], demand=[40, 40, 40])]
    sim._current_t = 0
    capacity = np.array([100.0, 100.0, 100.0])
    rejected = [make_request(demand=[1, 1, 1], revenue=20.0, rejection_weight=1.0)]
    pen = sim.compute_penalties(capacity, rejected_requests=rejected, admission_revenue=50.0)
    expected_rejection_penalty = 0.10 * 20.0 * 1.0
    expected_reward = 50.0 - 0.0 - expected_rejection_penalty - 0.0 + 2.0
    assert pen["reward"] == pytest.approx(expected_reward, rel=1e-6)
    assert pen["safe_bonus"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# 8. Q-learning update rule (eq. 36)
# ---------------------------------------------------------------------------
def test_q_learning_update_rule_hand_computed():
    policy = QLearningPolicy(fixed_opf=1.00, learning_rate=0.5, discount_factor=0.9)
    state = ("s0",)
    next_state = ("s1",)
    # Q(s0, accept)=0 initially; next-state Q values = [0, 0] -> best_next=0.
    decision = type("D", (), {"accept": True, "selected_opf": 1.00})()
    reward = 10.0
    policy.learn(state, decision, reward, next_state)
    # target = reward + gamma*best_next = 10 + 0.9*0 = 10
    # new Q = old + alpha*(target-old) = 0 + 0.5*(10-0) = 5.0
    assert policy.q_table[state][1] == pytest.approx(5.0)

    # Second update from the same state with a different next-state value.
    policy.q_table[next_state][1] = 4.0  # so max over next_state actions = 4.0
    policy.learn(state, decision, reward=10.0, next_state=next_state)
    # old=5.0, target=10+0.9*4=13.6, new=5.0+0.5*(13.6-5.0)=9.3
    assert policy.q_table[state][1] == pytest.approx(9.3)


# ---------------------------------------------------------------------------
# 9. OPF-aware action set (eq. 37): A^OPF = {(reject,1.00)} U {(accept,phi): phi in Phi}
# ---------------------------------------------------------------------------
def test_opf_aware_action_set_excludes_accept_at_1_00_by_default():
    """Regression test for the OPF-aware action set."""
    cfg = SimulatorConfig()
    policy = OPFAwareQLearningPolicy(opf_set=cfg.opf_set)
    assert "accept_1.00" not in policy.actions, (
        "Default Phi must not include 1.00; accept_1.00 must not be a reachable "
        "action unless the manuscript explicitly redefines Phi to include it."
    )
    assert policy.actions[0] == "reject"
    assert all(a.startswith("accept_") for a in policy.actions[1:])
    assert min(cfg.opf_set) >= 1.10 - 1e-9


def test_opf_aware_action_set_can_still_be_configured_with_1_00_explicitly():
    """If opf_set explicitly includes 1.00 (e.g. for an ablation
    comparing against the paper's action space), the policy should still work,
    confirming the exclusion above is a *default*, not a hard constraint."""
    policy = OPFAwareQLearningPolicy(opf_set=(1.00, 1.50))
    assert "accept_1.00" in policy.actions


# ---------------------------------------------------------------------------
# 10. q_fixed_opf behaviour for basic Q-learning
# ---------------------------------------------------------------------------
def test_basic_q_learning_default_fixed_opf_is_1_00():
    cfg = SimulatorConfig()
    assert cfg.q_fixed_opf == pytest.approx(1.00), (
        "Basic (accept/reject) Q-learning must default to q_fixed_opf=1.00 to "
        "match the paper's A^QL={reject,accept} action space with no implicit "
        "overbooking headroom."
    )


def test_basic_q_learning_uses_configured_fixed_opf_for_feasibility():
    cfg = SimulatorConfig()
    cfg.forecast_mode = "xgboost"
    sim = make_sim(cfg)
    policy_100 = QLearningPolicy(fixed_opf=1.00)
    policy_130 = QLearningPolicy(fixed_opf=1.30)
    # Reserved 100 (at nominal cap), demand 20: infeasible at OPF 1.00, feasible at OPF 1.30.
    ctx = make_context(capacity=[100, 100, 100], reserved=[100, 100, 100],
                       actual=[50, 50, 50], forecast=[50, 50, 50], buffer=[0, 0, 0])
    req = make_request(demand=[20, 20, 20])
    feasible_100, _, _ = sim.check_feasibility(ctx, req, opf=policy_100.fixed_opf, use_forecast=True)
    feasible_130, _, _ = sim.check_feasibility(ctx, req, opf=policy_130.fixed_opf, use_forecast=True)
    assert feasible_100 is False
    assert feasible_130 is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))


def test_no_forecast_mode_uses_current_actual_utilisation_when_enabled():
    """No-forecast forecast-mode sensitivity should be reactive, not nominal-only.

    Conservative admission is nominal-only, but a no-forecast static/admission
    mode should use current measured utilisation as its best estimate of near-
    future SLA feasibility, consistent with the manuscript description.
    """
    cfg = SimulatorConfig()
    cfg.forecast_mode = "no_forecast"
    sim = make_sim(cfg)
    ctx = make_context(capacity=[100, 100, 100], reserved=[10, 10, 10],
                       actual=[98, 98, 98], forecast=[0, 0, 0], buffer=[0, 0, 0])
    req = make_request(demand=[10, 10, 10], eta=[0.5, 0.5, 0.5])
    feasible, nominal_ok, forecast_ok = sim.check_feasibility(ctx, req, opf=2.00, use_forecast=True)
    assert nominal_ok is True
    assert forecast_ok is False
    assert feasible is False


def test_basic_q_learning_rejects_unseen_state_by_default():
    cfg = SimulatorConfig()
    cfg.forecast_mode = "xgboost"
    sim = make_sim(cfg)
    policy = QLearningPolicy(fixed_opf=1.00, allow_unseen_state_fallback=False)
    ctx = make_context(capacity=[100, 100, 100], reserved=[0, 0, 0],
                       actual=[0, 0, 0], forecast=[0, 0, 0], buffer=[0, 0, 0])
    req = make_request(demand=[1, 1, 1])
    decision = policy.decide(sim, ctx, req, training=False)
    assert decision.accept is False
    assert decision.reason == "q_unseen_state_reject"


def test_opf_aware_q_learning_rejects_unseen_state_by_default():
    cfg = SimulatorConfig()
    sim = make_sim(cfg)
    policy = OPFAwareQLearningPolicy(opf_set=cfg.opf_set, allow_unseen_state_fallback=False)
    ctx = make_context(capacity=[100, 100, 100], reserved=[0, 0, 0],
                       actual=[0, 0, 0], forecast=[0, 0, 0], buffer=[0, 0, 0])
    req = make_request(demand=[1, 1, 1])
    decision = policy.decide(sim, ctx, req, training=False)
    assert decision.accept is False
    assert decision.reason == "opf_q_unseen_state_reject"
