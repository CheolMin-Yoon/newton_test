# CLAUDE.md
# 새 환경에서 Claude Code 세션 시작 시 이 파일을 읽히면 기존 컨텍스트가 복원됩니다.

---

## 환경

- **conda 환경:** `mj`
- **주요 패키지:** pinocchio 4.0.0 / mujoco 3.6.0 / casadi 3.7.2
- **실행:** `conda run -n mj python3 ...` 또는 Jupyter 커널을 `mj` 환경으로 지정

---

## 프로젝트 구조

```
mj_opt/
├── models/              # MuJoCo XML, URDF, mesh (scene_torque.xml 등)
├── mj_sim/
│   ├── manipulator/     # Fixed-base (Panda, UR5e)
│   │   ├── core/
│   │   │   ├── fixed_base_robot_state.py   # 상태 컨테이너 (NQ=NV=6 기본, Panda는 9)
│   │   │   ├── mujoco_kernel.py            # 시뮬 I/O (step, ctrl read/write)
│   │   │   ├── pinocchio_wrapper.py        # 기구학·동역학 (Pinocchio 기반, 권장)
│   │   │   └── mujoco_wrapper.py           # 기구학·동역학 (MuJoCo API만, fallback)
│   │   ├── control/
│   │   │   ├── task_space_controller.py    # OSC impedance, DLS IK
│   │   │   └── trajectory_generator.py
│   │   └── utils/
│   │       ├── sim_scheduler.py            # ctrl_hz / render_hz 루프
│   │       ├── visualizer.py
│   │       ├── data_logger.py
│   │       └── plot_helpers.py
│   └── quadruped/       # Floating-base (Go2)
│       ├── core/
│       │   ├── floating_base_robot_state.py  # 상태 컨테이너 (NQ=19, NV=18)
│       │   ├── mujoco_kernel.py
│       │   └── pinocchio_wrapper.py
│       └── control/
│           ├── gait_scheduler.py
│           ├── leg_controller.py
│           └── srbd_model.py
├── test/                # 수치 검증 노트북
│   ├── test_core.ipynb              # core 모듈 전체 검증
│   ├── test_core_mj_vs_pin.ipynb    # MuJoCo vs Pinocchio 수치 비교
│   └── test_core_pin_vs_adam.ipynb  # Pinocchio vs Adam 비교 (미완)
└── newton_test/         # Newton 시뮬레이터 실험 (별도 repo, .gitignore)
```

---

## Core 아키텍처

### 설계 원칙

```
Mujoco_Kernel          ← 시뮬 I/O 전담 (mj_step, ctrl, qpos/qvel)
    │  model/data 참조
    ▼
Pinocchio_Wrapper      ← 기구학·동역학 계산 (권장)
Mujoco_Wrapper         ← 기구학·동역학 계산 (pinocchio 불가 시 fallback)
    │
    ▼
RobotState             ← q/dq 메모리 관리 (numpy view + copyto 패턴)
```

### Mujoco_Kernel

- `joint_names_pin_order` 필수 — Pinocchio joint 순서 기준으로 MuJoCo actuator 인덱스 매핑
- `_tau_perm`: arm actuator의 MuJoCo qvel 인덱스 배열 (그리퍼 등 비-arm 제외)
- `ctrl_tau` setter → `data.ctrl[_tau_perm]`에 직접 씀
- `q_keyframe`: MJCF `<keyframe>` 자세 (없으면 None)

### Pinocchio_Wrapper (권장)

- `update_model(q, dq)` 한 번으로 FK/M/C/g/nle/Jacobian 전부 갱신 (`computeAllTerms`)
- `compute_J_W(key)` / `compute_J_B(key)` / `compute_Jdot_dq_W(key)`
- `compute_ee_state_W(key)` → `(SE3, v_lin, v_ang)`
- `frames` 키는 URDF/MJCF link 이름 (`"ee"`, `"base"`)

### Mujoco_Wrapper (fallback)

- `kernel._tau_perm`으로 arm DOF 결정 (`self._col`, `self.nv = len(_tau_perm)`)
- Jacobian/M은 `model.nv` 크기로 계산 후 `[:, _col]` / `[ix_(_col,_col)]`로 arm 열만 반환
- **`mj_fwdPosition` 필수** — `mj_kinematics + mj_crb` 단독으로는 `qM`이 0 (관성 변환 미완료)
- `g`: `dq=0`에서 `mj_fwdVelocity` 후 `qfrc_bias[_col]`
- `wrapper` 생성 전 반드시 `kernel.reset_to_keyframe()` 먼저 호출 (q=0은 특이자세)

### RobotState

- **Fixed-base** (`FixedBaseRobotState`): NQ=NV=6 기본, Panda는 NQ=NV=9 서브클래스
- **Floating-base** (`FloatingBaseRobotState`): NQ=19 (pos3+quat4+joints12), NV=18
  - quat 컨벤션: Pinocchio `(x,y,z,w)` ↔ MuJoCo `(w,x,y,z)` — Kernel에서 변환
  - `compute_rpy_world()`: yaw 연속화(unwrap) 포함
- numpy view 패턴: `self.arm_pos = self._q[0:7]` → slice는 view, 값 갱신은 `np.copyto`

---

## 도메인 지식 — MuJoCo

### 액추에이터 타입별 B 행렬 gear 처리

LQR 등 제어기 설계 시 XML `<actuator>` 타입을 먼저 확인한다.

| 타입 | B 행렬 처리 | 이유 |
|------|------------|------|
| `motor` | `B = B_physical * gear` | `force = ctrl × gear` → ctrl 단위 맞추려면 gear 곱 필요 |
| `position` | gear 곱하지 않음 | ctrl이 목표 관절각, 내부 PD가 토크 계산 |
| `velocity` | gear 곱하지 않음 | ctrl이 목표 관절속도, 내부 PD가 토크 계산 |

대표 사례: 카트폴(`motor`, gear=50), UR5e/Go2(`position`)

---

## 코딩 작업 원칙

<!-- 출처: Anthropic Claude Code CLAUDE.md 가이드라인 기반 + 프로젝트 커스터마이징 -->

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it — don't delete it.
- Remove imports/variables/functions that YOUR changes made unused.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
```

---

## 명명·구조 컨벤션

<!-- 출처: mj_sim/작성원칙.md -->

핵심 한 줄: **단일문자(수식·내부)는 짧게, 단어(공개 API)는 명확하게, 좌표계는 `_W`/`_B`로 명시.**

### 메서드 prefix

| prefix | 의미 | 예시 |
|--------|------|------|
| `compute_*` | 호출 시점에 수치 연산 수행 (비용 있음) | `compute_J_W(key)`, `compute_ee_state_W(key)` |
| `update_*` | 매 step in-place 갱신, 반환 없음 | `update_model(q, dq)`, `update_robot_state(state)` |
| `solve_*` | QP/LP/MPC 솔버 호출 | `solve_dense_mpc()`, `solve_lqr()` |
| `get_*` | 캐시된 단순 조회 (인자로 키 받을 때만) | `get_hip_offset(leg)` |
| `@property` | 인자 없는 단일 값, 싸고 side-effect free | `wrapper.M`, `wrapper.nle`, `wrapper.p_com_W` |
| 동사형 | 부작용 있는 동작 | `step`, `reset_yaw_unwrap` |
| `plot_*` | matplotlib 함수 (`plot_helpers.py`) | `plot_ee_tracking(...)` |
| `draw_*` | MuJoCo viewer 오버레이 (`Visualizer`) | `draw_axes(...)` |

⚠️ `get_*`의 `set_*` 짝 금지 — setter는 `@property setter` 또는 직접 속성 할당.

### 변수 명명

- **단일문자** — 수식·내부 변수·클래스 멤버: `m`, `M`, `J`, `R`, `q`, `dq`, `tau`, `p`, `v`, `F`
- **단어** — 공개 API·로그 키: `target_pos`, `actual_pos`, `feet_jacobians`
- **suffix 규칙**:
  - desired: `_des` (`pos_des`, `tau_des`)
  - error: `_err` (`pose_err`, `pos_err`)
  - current/measured: **suffix 없음** (`pos`, `q`, `tau`)
  - reference trajectory (시계열): `_ref` (`x_ref`, `feet_pos_ref`)
  - terminal goal: `_target`
- **좌표계 suffix** (단일 대문자):
  - `_W` = world, `_B` = body, `_H` = hip/horizontal
  - 회전: `R_WB` (B→W), `R_BW` (W→B)
  - 예: `p_com_W`, `v_com_W`, `omega_base_B`, `r_feet`
- **물리량 prefix**: `p_` (pos), `v_` (vel), `a_` (acc), `J_`, `R_`, `M_`, `tau_`, `F_`
- **발 인덱싱**: 문자열 키 `"FL"/"FR"/"RL"/"RR"`, `LEG_KEYS` / `LEG_IDX` 상수로 관리
  ```python
  r_feet[:, LEG_IDX["FL"]]
  ```

### Underscore prefix 규칙

- `_xxx` — private 멤버/메서드 일관 적용
  - `self._q`, `self._dq` — Pinocchio API에 넘기는 raw 입력, `@property`로 외부 노출
  - `self._M_inv` — `update_model`에서 캐시, `@property M_inv`로 노출
- `_buf` suffix — pre-allocated numpy 버퍼, in-place 갱신 (`np.copyto` 또는 `[:] =`)
  - `self._q_buf`, `self._dq_buf`, `self._r_feet`
- `__xxx` — **사용 금지** (name mangling, 상속 깨짐). dunder 매직만 예외.

### 모듈 상수

- `UPPER_SNAKE_CASE`, import 직후 클래스 정의 위에 배치
- 예: `G1_FRAMES`, `LEG_KEYS`, `LEG_IDX`, `MANIPULATOR_FRAMES`

### 주석

- 카테고리 헤더는 한국어, 번호 없음, 위아래 빈 줄 1개
  ```python
  # 동역학

  @property
  def M(self): ...
  ```
- 인라인 주석은 "왜"만 — 코드가 이미 말하는 "무엇"은 쓰지 않음
- Docstring은 1줄 또는 생략 — 단위·shape·side effect가 모호한 경우만

### 클래스 내부 멤버 순서

1. `__init__`
2. 핵심 update/step — `update_model`, `update_robot_state`, `step`
3. `@property` (카테고리별 묶음)
4. 메서드 (`compute_*` → `solve_*` → `get_*` → 변환 → 유틸)
5. private 헬퍼 (`_build_*`, `_compute_*`)

### 클래스 vs 함수

- **상태가 있으면 클래스** (`Pinocchio_Wrapper`, `LQR(A, B, Q, R)`)
- **상태가 없으면 함수** (`rk4`, `zoh`, `c2d`) — 빈 `__init__` + 메서드는 함수의 위장
- MPC/SRBD는 책임별 분리: `SRBDDynamics`, `SRBDCost`, `SRBDConstraint`, `SRBDSolver`

### 한눈 요약

| 카테고리 | 컨벤션 | 예시 |
|---|---|---|
| 메서드 | `compute_* / update_* / solve_* / get_*` | `compute_J_W`, `update_model` |
| 단일문자 | 수식·멤버 | `m`, `M`, `J`, `R`, `q` |
| 단어 | 공개 API·로그 키 | `target_pos`, `actual_pos` |
| desired/error | `_des` / `_err` | `pos_des`, `pose_err` |
| current | 무접미사 | `pos`, `q` |
| 좌표계 | `_W` / `_B` / `_H` | `p_com_W`, `R_WB` |
| private | `_xxx` | `self._q`, `self._M_inv` |
| 버퍼 | `_xxx_buf` | `self._q_buf` |
| 모듈 상수 | `UPPER_SNAKE_CASE` | `LEG_KEYS`, `GO2_FRAMES` |

---

## 커밋 / 푸시 원칙

- 커밋·푸시 모두 명시적으로 요청할 때만 실행한다.
