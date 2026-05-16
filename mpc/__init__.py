"""SRBD-MPC 라이브러리 (배치/GPU 필수).

dynamics/ 의 강체항(.so)을 소비. locomotion/mdp 가 thin adapter 로 호출.
핵심 경계: srbd_mpc.solve_batch(state) -> plan  (순수 함수, side-effect 없음).
이 경계 덕에 V1(RL=WBC, obs로 plan) / V2(RL=planner, ActionTerm) 공존 가능.

srbd_model : SRBD A,B
srbd_mpc   : solve_batch (순수 경계)
batch_qp   : 배치 QP 백엔드
gait       : gait scheduler (접촉 스케줄)
reference  : com trajectory reference (twist → 추종 목표)
"""

from mpc.srbd_mpc import solve_batch as solve_batch
from mpc import gait as gait
from mpc import reference as reference
