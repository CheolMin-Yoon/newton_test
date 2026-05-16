"""Observation terms.

V1 핵심: SRBD-MPC plan 을 obs 로 주입 → RL(WBC)이 "MPC 출력을 고려"하는 지점.
"""

from __future__ import annotations


def base_state(env):
  """로봇 기본 상태 (base lin/ang vel, proj gravity, joint pos/vel ...)."""
  raise NotImplementedError


def mpc_plan_obs(env):
  """env._mpc_plan (SrbdPlan) 을 flatten 해 관측으로. mdp.mpc.update_mpc_plan 선행."""
  raise NotImplementedError("env._mpc_plan grf/wrench/contact_sched flatten")
