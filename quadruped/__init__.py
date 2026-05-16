"""Go2 로봇 레이어 (mjlab asset_zoo 대응).

URDF(Pinocchio codegen용) + MJCF(mjlab scene용) + EntityCfg + 상태 변환.
mpc/ 와 locomotion/ 가 공용으로 import 하는 로봇 정의 — RL 보상/관측 없음.
"""

from quadruped.go2 import get_go2_robot_cfg as get_go2_robot_cfg
