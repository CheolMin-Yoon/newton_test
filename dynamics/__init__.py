"""Codegen 공장 산출물 레이어 (Mythos.md Part A).

cpin(Pinocchio symbolic) → CasADi → GPU codegen.
산출물: casadi_fns/*.casadi (symbolic, repo 커밋). .so 는 mjlab_env site-packages 에 설치(A8).
mpc/ 와 locomotion/ 가 공용으로 소비하는 하위 레이어.
"""
