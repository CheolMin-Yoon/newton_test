# batch

GPU 병렬 동역학 평가 + 로봇 시뮬레이션 학습 워크플로


## 구성

| 파일 | 내용 |
|---|---|
| [envs.md](envs.md) | **3-env 워크플로 전략** — codegen / runtime / debug 분리 |
| [batch.md](batch.md) | 환경 설치 절차 (CasADi cuda_codegen, robotpkg pinocchio, PyTorch, casadi-on-gpu) |
| [newton.md](newton.md) | Newton 시뮬레이터 활용 정리 — 단일/병렬 sim, diff-sim |
| [batch_dynamics.md](batch_dynamics.md) | Pinocchio 동역학 추출 + 학습/평가 시각화 레퍼런스 |


## 환경 구성 예정 

1. mj
python 3.12
무조코 + 피노키오 + 카사디 기반 베이스라인 디버깅


2. codegen
python 3.12
여기서는 코드 생성만


3. newton or isaaclab
미정

또는 Isaac Lab 3.0 기다림