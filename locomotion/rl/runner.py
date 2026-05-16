"""On-policy 러너 (mjlab tasks/velocity/rl/runner.py 대응).

기본은 mjlab MjlabOnPolicyRunner 재사용. WBC 특화(예: MPC plan 로깅)가 필요하면
여기서 subclass. cartpole 은 MjlabOnPolicyRunner 를 그대로 register 에 넘김.
"""

from __future__ import annotations

from mjlab.rl.runner import MjlabOnPolicyRunner


class LocomotionOnPolicyRunner(MjlabOnPolicyRunner):
  """현재는 동작 동일. WBC 진단 로깅 추가 지점."""
  pass
