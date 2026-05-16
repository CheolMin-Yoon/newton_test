"""배치 QP 백엔드.

env step 내 num_envs 동시 solve 필요 → per-env CPU 솔버 금지.
옵션:
  (a) casadi-on-gpu codegen: SRBD-MPC QP/ADMM 를 .casadi → .so → cog.launch (Mythos.md A8b)
  (b) torch/warp 배치 ADMM/box-QP

(a) 가 Mythos.md 철학과 정합 (dynamics/casadi_fns/srbd_mpc.casadi 추가 codegen).
"""

from __future__ import annotations


def solve(H, g, A, lb, ub):
  """배치 QP: min 0.5 xᵀHx + gᵀx  s.t. lb ≤ Ax ≤ ub. (leading dim = N)"""
  raise NotImplementedError("배치 QP 백엔드 미선택 (codegen .so vs warp ADMM)")
