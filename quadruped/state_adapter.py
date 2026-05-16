"""mjlab Entity.data → SRBD/Pinocchio 입력 변환 (배치).

핵심 글루: MuJoCo qpos/qvel 순서 ↔ Pinocchio nq/nv 순서,
floating-base 쿼터니언 컨벤션 MuJoCo (w,x,y,z) ↔ Pinocchio (x,y,z,w).
mpc/ 와 dynamics/.so 호출 직전에 적용. RAL2025 레퍼런스의 remap 패턴.
"""

from __future__ import annotations


def mjlab_to_pin_qdq(entity):
  """mjlab Entity → (q[N,19], dq[N,18]) Pinocchio 순서·쿼터니언 컨벤션."""
  raise NotImplementedError("joint 순서 매핑 + quat (wxyz→xyzw) 변환")


def to_srbd_state(entity, terms: dict, contact) -> dict:
  """Entity + go2_all_terms 출력 + 접촉상태 → mpc.solve_batch 가 받는 SRBD state dict.

  Args:
    entity: mjlab robot Entity (base p/rpy/v/omega 추출).
    terms: dynamics/.so go2_all_terms 출력 (r_feet_com, R_WB, I_W, mass ...).
    contact: mjlab ContactSensor 판독 (어느 발이 접지 — SRBD 접촉스케줄 필수 입력.
             pin terms 만으로 도출 불가하므로 별도 인자).
  """
  raise NotImplementedError(
    "base p/rpy/v/omega + r_feet_com + R_WB + I_W + mass + contact mask 조립"
  )
