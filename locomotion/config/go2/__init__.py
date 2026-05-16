"""Go2 task 등록 — import 시 register_mjlab_task 실행 (mjlab 발견 메커니즘).

scripts/train.py·play.py 가 `import locomotion.config.go2` 하면 등록됨.
mjlab 은 폴더명이 아니라 이 호출 + task_id 문자열로 task 를 찾음.
V2 추가 시 여기 register 한 줄 더 (Mythos-Go2-Planner).
"""

from mjlab.tasks.registry import register_mjlab_task

from locomotion.config.go2.env_cfgs import go2_wbc_env_cfg
from locomotion.config.go2.rl_cfg import go2_wbc_ppo_runner_cfg
from locomotion.rl import LocomotionOnPolicyRunner

# NOTE: cfg 들이 아직 NotImplementedError stub 이라 import-time 등록은
# 구현 완료 후 활성화. 스캐폴드 단계에서는 wiring 만 명시.
def register() -> None:
  register_mjlab_task(
    task_id="Mythos-Go2-WBC",
    env_cfg=go2_wbc_env_cfg(),
    play_env_cfg=go2_wbc_env_cfg(play=True),
    rl_cfg=go2_wbc_ppo_runner_cfg(),
    runner_cls=LocomotionOnPolicyRunner,
  )


# 구현 완료되면 아래 주석 해제 → import 만으로 자동 등록 (cartpole 패턴)
# register()
