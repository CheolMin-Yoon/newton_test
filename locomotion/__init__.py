"""mjlab RL task (mjlab 'velocity' task 관례 대응).

구조: locomotion_env_cfg.py(base) + mdp/(빌딩블록) + rl/runner.py + config/<robot>/(등록).
V1(RL=WBC): SRBD-MPC plan 을 mdp/observations 로 주입, action=관절토크(JointEffortAction).
"""
