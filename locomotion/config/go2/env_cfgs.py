"""Go2 V1(WBC) 구체 env cfg — base(locomotion_env_cfg)를 override.

V1: action = JointEffortAction(관절토크, RL=WBC), obs 에 mdp.mpc_plan_obs 포함.
(V2 추가 시 같은 base 위에 env_cfgs 만 분기 — mpc/dynamics/quadruped 재사용)
"""

from __future__ import annotations

from locomotion.locomotion_env_cfg import locomotion_env_cfg


def go2_wbc_env_cfg(play: bool = False):
  cfg = locomotion_env_cfg(play=play)
  # TODO: scene entity = quadruped.get_go2_robot_cfg()
  # TODO: actions = JointEffortAction(관절 12)
  # TODO: observations.actor += mdp.mpc_plan_obs (SRBD-MPC plan)
  # TODO: events: dynamics .so terms + mdp.mpc.update_mpc_plan (plan 캐시)
  return cfg
