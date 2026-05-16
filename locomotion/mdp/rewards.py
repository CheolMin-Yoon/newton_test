"""Reward terms.

추종(속도/자세) + 안정(CAM 등) + (옵션) MPC plan 일치 보상.
CMM 분해/CAM reward 개념은 Mythos.md Part C 참조 — mjlab 매니저 API 로 재작성.
"""

from __future__ import annotations


def tracking_velocity(env):
  raise NotImplementedError


def mpc_consistency(env):
  """정책 실현 결과 vs env._mpc_plan 편차 보상 (선택)."""
  raise NotImplementedError
