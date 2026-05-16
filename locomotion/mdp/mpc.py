"""배치 SRBD-MPC 브리지 — env step 당 1회 호출, plan 을 env 에 캐시.

흐름 (V1):
  Entity.data --state_adapter--> SRBD state --mpc.solve_batch--> SrbdPlan
  → env._mpc_plan 에 캐시 → observations.py/rewards.py 가 읽음.

RL 은 mpc/ 를 직접 import 하지 않음 — 이 브리지(=mdp term)를 통해서만.
solve_batch 는 순수(srbd_mpc.py) → 여기서 캐시/주기(decimation) 같은 부수효과 관리.
"""

from __future__ import annotations

from mpc import solve_batch
from quadruped.state_adapter import to_srbd_state


def update_mpc_plan(env) -> None:
  """env step hook: SRBD-MPC 풀어 env._mpc_plan 갱신 (EventTerm/observation 선행)."""
  # terms = env._dyn_terms  # dynamics/.so go2_all_terms 출력 (별도 term 에서 채움)
  # state = to_srbd_state(env.scene["robot"], terms)
  # env._mpc_plan = solve_batch(state, dt=..., horizon=...)
  raise NotImplementedError("dynamics .so 연동 + decimation 정책 확정 후 구현")
