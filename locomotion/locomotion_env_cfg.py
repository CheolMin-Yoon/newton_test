"""BASE 환경 cfg (mjlab velocity_env_cfg.py 대응).

robot-agnostic 공통 골격. config/go2/env_cfgs.py 가 robot 별로 override.
velocity_env_cfg 처럼 **모든 manager 슬롯을 구조적으로 노출**한다
(본문은 NotImplementedError stub — 구조 계약만 확정).

V1(WBC): action=JointEffortAction(관절토크=RL), obs에 SRBD-MPC plan 포함.
contact sensor 는 RL obs 겸 **SRBD-MPC 접촉스케줄 입력**으로 공유.
"""

from __future__ import annotations


# ── Scene: robot + terrain + sensors ─────────────────────────────────────────
def _scene_cfg(play: bool):
  """SceneCfg 골격.

  entities : quadruped.get_go2_robot_cfg()  (config/go2 에서 주입)
  terrain  : TerrainEntityCfg + ROUGH_TERRAINS_CFG (floating-base 기본 rough)
  sensors  : contact(feet_ground_contact) ─ RL obs + SRBD-MPC 접촉입력 공유
             imu(lin_vel/ang_vel), terrain_scan / foot_height_scan (RayCast/Height)
  """
  raise NotImplementedError("SceneCfg(entities, terrain, sensors[contact/imu/scan])")


# ── Observations: actor(noisy) / critic(privileged) + MPC plan ───────────────
def _observations_cfg(play: bool):
  """비대칭 actor-critic.

  actor  : base lin/ang vel, projected gravity, joint pos/vel, last action,
           command, (V1) mdp.mpc_plan_obs   ← RL 이 MPC 출력을 고려하는 지점
  critic : actor + privileged (height_scan, foot_contact_forces ...)
  """
  raise NotImplementedError("ObservationGroupCfg actor/critic (+mpc_plan_obs)")


# ── Actions: V1 = 관절토크(RL=WBC) ───────────────────────────────────────────
def _actions_cfg():
  raise NotImplementedError("JointEffortActionCfg (관절 12, RL=WBC)")


# ── Commands: twist (RL obs 겸 SRBD-MPC reference) ───────────────────────────
def _commands_cfg():
  raise NotImplementedError("UniformVelocityCommandCfg('twist') — MPC reference 겸용")


# ── Events: reset / push / domain randomization ──────────────────────────────
def _events_cfg():
  raise NotImplementedError(
    "reset_base, reset_joints, push_robot, foot_friction/base_com/encoder_bias(DR)"
  )


# ── Curriculum ───────────────────────────────────────────────────────────────
def _curriculum_cfg():
  raise NotImplementedError("terrain_levels CurriculumTermCfg")


# ── Rewards / Terminations ───────────────────────────────────────────────────
def _rewards_cfg():
  raise NotImplementedError("tracking + 안정(CAM/CMM) + (옵션) mpc_consistency")


def _terminations_cfg():
  raise NotImplementedError("base_fell + time_out (+ out_of_terrain_bounds)")


def locomotion_env_cfg(play: bool = False):
  """공통 ManagerBasedRlEnvCfg 조립 (velocity_env_cfg 구조 미러).

  순서: scene → observations → actions → commands → events → curriculum
        → rewards → terminations → sim/viewer/decimation/episode_length.
  config/go2 가 scene.entities·robot 종속 필드(geom/site/body)·rl_cfg override.
  """
  raise NotImplementedError(
    "ManagerBasedRlEnvCfg(scene=_scene_cfg, observations=_observations_cfg, "
    "actions=_actions_cfg, commands=_commands_cfg, events=_events_cfg, "
    "curriculum=_curriculum_cfg, rewards=_rewards_cfg, terminations=_terminations_cfg, "
    "sim=SimulationCfg(...), decimation=..., episode_length_s=...)"
  )
