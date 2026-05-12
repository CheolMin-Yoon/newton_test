# Newton 기반 GO2 시뮬레이션 — 진행 정리

## 배경: Genesis 한계 → Newton 전환

| 항목 | Genesis | Newton |
|---|---|---|
| rigid body diff sim end-to-end | 사실상 미작동 (backward hang) | **작동** (Featherstone solver) |
| ground contact 미분 | silent hang | 명시적 `enable_backward=False` 경고 |
| 진단 가능성 | 낮음 | 높음 |
| URDF/MJCF import | OK | OK + USD |
| MuJoCo Warp 통합 | X | ✅ 기본 backend |
| 학습 리소스 | Genesis 자체 | Warp 표준 + Disney/DeepMind/NVIDIA 합작 |

## 도달한 결과 (체크 완료)

### 1. 기본 인프라
- ✅ Newton repo 설치 + 예제 동작 확인
- ✅ conda env `batch` 안에 통합 (numpy 1.x + cuda_codegen casadi + robotpkg pinocchio 와 공존)

### 2. 파일별 진행

| 파일 | 내용 | 상태 |
|---|---|---|
| [go2_newton.py](go2_newton.py) | 단일 Go2 spawn + stand pose + sin motion | ✅ 동작 |
| [go2_newton_parallel.py](go2_newton_parallel.py) | `builder.replicate(robot, N, spacing=...)` 로 N개 병렬 spawn (16/64 환경) | ✅ 동작 |
| [go2_newton_parallel_dynamics.py](go2_newton_parallel_dynamics.py) | 병렬 sim + Newton state → (q, dq) → `casadi-on-gpu` 커널 (`go2_all_terms`) 매 step 호출 | ✅ M 대칭, 자체 일관 |
| [go2_newton_dynamics_crosscheck.py](go2_newton_dynamics_crosscheck.py) | Newton MuJoCo Warp 내부 데이터 vs casadi 출력 비교 | ✅ M trace, total mass 검증 (rotor 모델 1.068 kg 차이 확인) |
| [go2_newton_diffsim.py](go2_newton_diffsim.py) | SolverSemiImplicit + ground contact diff-sim | ⚠️ grad=0 또는 NaN (semi-implicit + contact의 한계) |
| [go2_newton_diffsim_airborne.py](go2_newton_diffsim_airborne.py) | **공중 부양 + Featherstone + joint pos target loss** | ✅ **학습 성공** (loss 매 iter 감소) |
| [go2_newton_diffsim_contact.py](go2_newton_diffsim_contact.py) | Featherstone + 지면 접촉 추가 | ✅ **학습 됨** (loss 1.08→0.62 in 3 iter), contact kernel은 `enable_backward=False` 경고 |

### 3. Diff-Sim 핵심 결과

**Newton + Featherstone**:
- forward ~57ms / backward ~220ms (horizon=5, single robot)
- 공중에서: loss 0.23 → 0.02 (10 iter)
- 지면 접촉: loss 1.08 → 0.62 (3 iter), grad nonzero
- **contact 자체의 미분은 부정확** (Newton이 명시적으로 경고), 다만 joint dynamics path는 정상 → 실용적 학습 가능

**미분 가능한 loss / 아닌 loss**:

| Loss 종류 | 신뢰성 |
|---|---|
| Joint position target | ✅ joint dynamics 지배 |
| Base CoM trajectory | ✅ inertia + joint coupling |
| Foot 위치 (fk-based) | 🟡 contact 무관하면 OK |
| Footstep timing / 접촉 패턴 | ❌ contact event 미분 부정확 |
| Gait phase reward | ❌ contact 기반 |

### 4. casadi-on-gpu 연결 검증

- `go2_all_terms.casadi` (M, g, C_dq, Ag, hg, I_W, p_com, v_com, Jcom, base_p, R_WB, R_WH, r_feet, R_feet, J_feet_W, Jd_dq_feet_W) 매 step GPU 평가
- Newton state (`state.joint_q`, `state.joint_qd`) → torch 텐서 변환 → 커널 호출 → 16 outputs
- M 대칭성, trace(M) = 49.7 (rotor 포함 Pinocchio 모델), g_base_z 자체 일관 확인

---

## 추후 가능한 것들

### A. Diff-MPC 데모 (가장 검증된 활용)

| 모델 | 태스크 | 난이도 |
|---|---|---|
| **Cartpole swing-up** | cart에 힘만 줘서 막대 위로 올리고 유지 | 표준 / 1순위 |
| **2-link arm reaching** | end-effector → target 위치 | 표준 / 빠른 수렴 |
| **Double pendulum / Acrobot** | 극한 비선형 swing-up | 중급 |

지면 접촉 없는 모델 — Newton + Featherstone diff sim이 완전히 신뢰 가능.
horizon 50~200 step, Adam optimizer + Newton step 가능.

### B. casadi-on-gpu 활용 — 모델 기반 RL (RAL2025 패턴)

- forward-only RL (PPO/SAC) + reward feature로 우리 동역학 양 (M, Ag, hg, J 등)
- **CAM-regularized reward**: `hg.angular` 의 L2를 reward로 → angular momentum regularization
- 학습 안정성 ↑, sim-to-real transfer ↑
- contact 미분 필요 없음 — 가장 실용적 path

### C. Soft Contact Diff-Sim 시도

- 자체 구현 penalty-based ground contact (예: `F = -k * max(0, -z) - d * vz`)
- ke/kd 직접 grad-supported kernel로 작성
- 발 접촉도 미분 가능하게 → locomotion learning with diff sim

### D. 학습 인프라 확장

- Adam optimizer (현재 plain SGD)
- TensorBoard / W&B 로깅
- Multi-environment 병렬 backward (memory 비용 ↑)
- Receding horizon control 루프 (diff-MPC 실시간)

### E. 추가 모델 export

- Manipulator URDF 추가 (`export_pinocchio_casadi.py` 일반화)
- Humanoid 모델 (Go2 노하우 그대로 적용)

---

## 한계 / 알려진 이슈

1. **Newton contact backward 미지원**: `_clear_active_kernel`, `_zero_count_and_contacts_kernel` 등이 `enable_backward=False`. Contact-aware MPC는 부정확.
2. **메모리 비용**: diff sim은 horizon × substeps × N states 만큼 미리 할당. Go2 horizon=10이면 ~40 state 시퀀스.
3. **backward 속도**: forward의 ~4배. horizon이나 N 늘리면 부담 증가.
4. **robotpkg pinocchio + numpy 2 충돌**: `.casadi` 재export 시 numpy<2 토글 필요.

---

## 빠른 실행 가이드

```bash
conda activate batch

# 단일 GO2
python go2_newton.py

# 16개 병렬
python go2_newton_parallel.py --world-count 16

# 병렬 + GPU 동역학 커널
python go2_newton_parallel_dynamics.py --world-count 16

# Diff sim (공중)
python go2_newton_diffsim_airborne.py --horizon 5 --no-viewer

# Diff sim (지면 접촉)
python go2_newton_diffsim_contact.py --horizon 5 --no-viewer
```

---

## 결론

**Newton의 진짜 활용 영역**:
- ✅ **지면 접촉 없는 시스템의 diff sim** (cartpole, manipulator, swing-up 등) — 완전 신뢰 가능
- ✅ **forward-only RL with analytical features** (우리 casadi-on-gpu와 조합)
- 🟡 **rigid robot with contact의 학습** — joint path는 신뢰, contact path는 부정확

Genesis의 silent hang 문제는 Newton에서 해결됐고, **rigid articulated diff sim이 진짜로 동작한다는 것**을 확인했습니다. 그 결정적 차이가 앞으로 모든 모델 기반 컨트롤 / MPC / 학습 실험을 가능하게 합니다.
