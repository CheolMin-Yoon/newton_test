"""SRBD-MPC 배치 솔버 — 공존 설계의 핵심 순수 경계.

★ 규율: solve_batch 는 pure 함수 (state in → plan out). side-effect 금지.
  ActionTerm/ObservationTerm 어디서 불러도 안전해야 V1·V2 공존 비용이 최소.
  MPC 로직을 adapter(actions.py/observations.py)에 inline 하지 말 것.

배치 QP 백엔드는 batch_qp.py (codegen .so 또는 torch/warp ADMM).
"""

from __future__ import annotations

from typing import TypedDict


class SrbdPlan(TypedDict):
  """SRBD-MPC 1-step 출력 (모두 배치: leading dim = num_envs)."""
  grf: object            # desired ground reaction forces  (N, 4, 3)
  wrench: object         # desired base wrench              (N, 6)
  contact_sched: object  # 접촉 스케줄                       (N, 4)


def solve_batch(state: dict, *, dt: float, horizon: int) -> SrbdPlan:
  """배치 SRBD-MPC 1-step. env step 안에서 num_envs 만큼 호출됨.

  Args:
    state: quadruped.state_adapter 가 만든 배치 SRBD 입력
           (base p/rpy/v/omega, r_feet_com, R_WB, I_W, mass ...).
    dt, horizon: MPC 이산화/지평.
  Returns:
    SrbdPlan (순수 출력, env 변형 없음).
  """
  raise NotImplementedError(
    "srbd_model.build_srbd_AB → batch_qp.solve 조립. 순수 유지."
  )
