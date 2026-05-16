"""SRBD (Single Rigid Body Dynamics) 선형 모델 A, B 구성.

상태 x = [p(3), rpy(3), v(3), omega(3)] (+gravity term = 13).
입력 u = 발 접촉력 (3 * n_contact).

필요한 full-model 항(질량, I_W, R_WB/R_WH, p_com, r_feet_com)은
dynamics/casadi_fns/go2_all_terms 의 출력 부분집합 → 그걸 받아서 A,B 구성.
배치(N envs) 텐서 연산 기준.
"""

from __future__ import annotations


def build_srbd_AB(terms: dict, dt: float):
  """go2_all_terms 출력(batched) → 이산 SRBD (A, B).

  Args:
    terms: {"I_W","R_WB","r_feet_com","mass",...} 배치 텐서.
    dt: MPC 이산화 timestep.
  Returns:
    (A, B) 배치 텐서.
  """
  raise NotImplementedError("SRBD A,B 구성 — terms 키/shape 확정 후 구현")
