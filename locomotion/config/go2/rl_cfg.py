"""Go2 V1(WBC) PPO 러너 cfg (cartpole rl_cfg.py 패턴).

V1: action = 관절토크(~12) → 정책망/스케일은 그에 맞춤.
정책 신경망은 RslRlModelCfg(hidden_dims=...) 'config' 로 선언 (직접 코딩 아님).
"""

from __future__ import annotations


def go2_wbc_ppo_runner_cfg():
  # from mjlab.rl.config import (RslRlOnPolicyRunnerCfg, RslRlModelCfg,
  #                              RslRlPpoAlgorithmCfg)
  raise NotImplementedError(
    "RslRlOnPolicyRunnerCfg(actor/critic=RslRlModelCfg(hidden_dims=...), "
    "algorithm=RslRlPpoAlgorithmCfg(...), experiment_name='go2_wbc', logger='tensorboard')"
  )
