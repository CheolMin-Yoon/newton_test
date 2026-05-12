# Batch Dynamics — 동역학 추출 & 시각화 레퍼런스

휴머노이드 학습/제어 작업에 활용할 동역학 항과 시각화 옵션 정리.
출처: `LearningHumanoidArmMotion-RAL2025-Code` 심층 분석 + 본 프로젝트에서 검증한 패턴.

---

## 1. 동역학 항 추출 (Pinocchio → CasADi → GPU)

### 1.1 추출 흐름

```
URDF
  ↓ pin.buildModelFromUrdf(..., JointModelFreeFlyer())
Pinocchio Model
  ↓ cpin.Model(model)
Symbolic CasADi Model (cpin)
  ↓ cpin.crba, cpin.computeCoriolisMatrix, cpin.computeCentroidalMap, ...
  ↓ ca.Function(...).expand()
  ↓ f.save(*.casadi)
.casadi 직렬화 파일
  ↓ generate_manifest_and_registry.py (cuda codegen)
.cu / .cuh 커널
  ↓ cmake build
casadi_on_gpu.*.so  ←  cog.launch(name, [in_ptrs], [out_ptrs], N)
```

### 1.2 추출 항목 (전체 목록)

휴머노이드 24-DOF (base 6 + leg 12 + arm 6) 기준. Go2처럼 18-DOF (base 6 + leg 12) 도 동일 구조, arm 부분만 제거.

| 함수 | 의미 | 입력 | 출력 shape |
|---|---|---|---|
| **M(q)** | 질량행렬 | q | (n_dof, n_dof) |
| **C(q,q̇)·q̇** | Coriolis 토크 | q, q̇ | (n_dof,) |
| **G(q)** | 중력 토크 | q | (n_dof,) |
| **nle(q,q̇)** | = C·q̇ + G | q, q̇ | (n_dof,) |
| **A(q) = CMM** | Centroidal Momentum Matrix | q | (6, n_dof) |
| **Ȧ(q,q̇) = dCMM** | dA/dt | q, q̇ | (6, n_dof) |
| **h = A·q̇** | Centroidal Momentum (linear + angular) | q, q̇ | (6,) |
| **ḣ = Ȧq̇ + Aq̈** | dCM | q, q̇, q̈ | (6,) |
| **CoM(q)** | Center of Mass position | q | (3,) |
| **v_CoM** | = A_linear · q̇ / m_total | q, q̇ | (3,) |
| **J_CoM** | CoM Jacobian | q | (3, n_dof) |
| **I_W = A_angular** | World-frame centroidal inertia (3×3 block from CMM) | q | (3, 3) |
| **fk_\<frame\>(q)** | Forward kinematics: position + rotation | q | (3,) + (3,3) |
| **J_\<frame\>(q)** | Frame Jacobian (LWA / LOCAL) | q | (6, n_dof) |
| **J̇_\<frame\>·q̇** | Frame Jacobian time variation (acceleration kinematic part) | q, q̇ | (6,) |
| **base_pos(q), base_rot(q)** | Base SE(3) | q | (3,), (3,3) |

### 1.3 단일 함수로 묶기 (권장 패턴)

RAL2025 는 9개 `.casadi` 파일로 분리. 본 프로젝트(Go2) 는 **하나의 함수로 묶음** — kernel launch 1회로 끝나서 batch sim에 유리.

본 프로젝트 사용 예시 ([export_pinocchio_casadi.py](mj_sim/quadruped/core/export_pinocchio_casadi.py)):

```python
f = Function(
    "go2_all_terms",
    [q, dq, theta],
    [M, g, C_dq, Ag, hg, Ig,
     p_com, v_com, Jcom,
     base_p, R_WB, R_WH,
     r_feet, R_feet,
     J_feet_W, Jd_dq_feet_W],
    ["q", "dq", "theta"],
    ["M", "g", "C_dq", "Ag", "hg", "I_W",
     "p_com", "v_com", "Jcom",
     "base_p", "R_WB", "R_WH",
     "r_feet", "R_feet", "J_feet_W", "Jd_dq_feet_W"],
)
f.expand()
f.save("go2_all_terms.casadi")
```

장점: launch 1회 × N개 환경 → GPU pointer-based 동기화 1회 → 단일 cudaStream 안에서 끝남.

### 1.4 핵심 인사이트 — CMM 분해 (base/leg/arm)

논문의 contribution 부분. CMM의 **열을 부분 group별로 잘라서** 각 그룹이 만드는 partial centroidal momentum을 분리:

```python
# CMM = A(q): shape (N, 6, n_dof)
CM       = (CMM @ qdot.unsqueeze(2)).squeeze(2)              # 전체 h
CM_base  = (CMM[:, :, 0:6]   @ qdot[:, 0:6].unsqueeze(2)).squeeze(2)
CM_leg   = (CMM[:, :, 6:18]  @ qdot[:, 6:18].unsqueeze(2)).squeeze(2)
CM_arm   = (CMM[:, :, 18:24] @ qdot[:, 18:24].unsqueeze(2)).squeeze(2)
CM_des   = (CMM @ qdot_desired.unsqueeze(2)).squeeze(2)       # 명령 기반 desired momentum
```

base-frame 변환:
```python
R_WB = quat_to_R(base_quat)
block = block_diag(R_WB.T, R_WB.T)   # (6,6)
CM_bf      = block @ CM
CM_leg_bf  = block @ CM_leg
CM_arm_bf  = block @ CM_arm
```

**왜 중요한가**:
- `CM_leg + CM_arm` → robot이 의도적으로 만드는 angular momentum
- arm이 leg의 yaw momentum을 상쇄하면 base가 yawing 없이 안정
- reward로 강제하면 정책이 "팔을 흔들어서 회전 보상" 학습

본 프로젝트 (Go2 — 팔 없음) 에서의 변형:
- `CM_leg` 만 분리해서 base CM 안정화에 활용
- 또는 `Ag` 의 회전 부분 (`Ag[:, 3:6, :]`) 을 직접 reward feature로

### 1.5 Reward 적용 예시 ([rewards.py:178-196](LearningHumanoidArmMotion-RAL2025-Code/extensions/humanoid/mdp/rewards.py#L178-L196))

```python
def CAM_xy_penalty(env, asset_cfg):
    # roll/pitch axis angular momentum 억제 (body frame)
    return -torch.sum(torch.square(env.CM_bf[:, 3:5]), dim=1)

def dCAM_xy_penalty(env, asset_cfg):
    # CM 의 양의 시간 변화만 penalize (build-up 방지)
    return -torch.clamp_min(torch.sum(env.CM_bf[:, 3:5] * env.dCM_bf[:, 3:5], dim=1), 0.0)

def tracking_CAM_reward(env, asset_cfg, command_name):
    # desired CAM (보통 yaw axis) tracking
    error = env.CM_des[:, 5] - env.CM[:, 5]
    error *= 1./(1. + torch.abs(env.CM_des[:, 5]))
    return torch.exp(-error**2 / (2*sigma**2))

def CAM_compensation_reward(env):
    # leg yaw momentum + arm yaw momentum = 0  (서로 상쇄)
    return torch.exp(-(env.CM_leg[:, -1] + env.CM_arm[:, -1])**2 / (2*sigma**2))
```

### 1.6 GRF (Ground Reaction Force) — toe / heel 분리

IsaacLab 기본 `ContactSensor` 를 custom 확장해서 **per-contact-point** 정보 노출:

```python
# 커스텀 sensor 출력
rf_contact.data.GRF_points_buffer  # (N, max_contacts, 3) world position
rf_contact.data.GRF_forces_buffer  # (N, max_contacts, 3) 3-axis force
rf_contact.data.GRF_count_buffer   # (N,) 유효 contact 개수
```

토우/힐 분류는 **기하학적 근접**으로 ([humanoid_vanilla.py:206-226](LearningHumanoidArmMotion-RAL2025-Code/extensions/humanoid/task/humanoid_vanilla.py#L206-L226)):
```python
TOL = 1e-2
toe_diff = (foot_to_toe_world.unsqueeze(1) - contact.GRF_points_buffer)
toe_mask = (toe_diff.norm(dim=-1) < TOL) * valid_mask
rf_toe_GRF = (GRF_forces * toe_mask.unsqueeze(2)).sum(dim=1)        # (N, 3)
```

Ground Reaction Moment (GRM):
```python
rf_GRM = cross(rfoot_to_toe_w, rf_toe_GRF) + cross(rfoot_to_heel_w, rf_heel_GRF)
```

본 프로젝트 (Newton + Go2) 에서는:
- MuJoCo Warp 기본 contact API 사용 — point 단위 force는 `mjw_data.contact` 에 있음
- 또는 `model.contacts()` 의 buffer 직접 활용

### 1.7 본 프로젝트와의 매핑

| 항목 | RAL2025 | 본 프로젝트 |
|---|---|---|
| Symbolic source | `cpin` (robotpkg pinocchio) | `cpin` (robotpkg pinocchio, numpy<2) |
| GPU backend | `cusadi` | `casadi-on-gpu` (cuda_codegen casadi 브랜치) |
| 함수 묶음 | 9개 분리 (`M_*.casadi`, `CMM_*.casadi`, ...) | 1개 통합 (`go2_all_terms.casadi`) |
| 호출 | `cusadi_CMM_fn.evaluate([q.cuda(), qdot.cuda()])` | `cog.launch("go2_all_terms", [q, dq, theta], outs, N)` |
| 검증 | IsaacLab MuJoCo PhysX mass matrix와 비교 | Newton MuJoCo Warp 내부 데이터와 비교 |

---

## 2. 시각화 / 평가 시각화

### 2.1 학습 중 (online)

| 도구 | 용도 | 비용 | 비고 |
|---|---|---|---|
| **TensorBoard** | scalar/curve/histogram, 실시간 브라우저 뷰 | 가벼움 | PyTorch `SummaryWriter` 표준 |
| **W&B (wandb)** | 클라우드 대시보드, 실험 비교, hyperparam sweep | 무료 + 인터넷 | RAL2025도 사용 (`requirements.txt`) |
| **matplotlib live plot** | 가장 가벼운 즉시 확인 | 가벼움 | jupyter inline 적합 |
| **rerun.io** | 3D viewer + scalar/log 통합 | 중간 | 최근 인기, Newton과 결합 사례 있음 |
| **viser** | 웹 기반 인터랙티브 3D + UI | 중간 | 학습 디버깅에 유용 |

학습 중 가장 자주 보는 것:
- **loss curve** (PPO surrogate, value loss, entropy)
- **reward components** (각 reward term별 평균)
- **episode length** (terminate 빈도)
- **action statistics** (mean, std)
- **mass matrix condition number** 같은 분석적 metric

#### TensorBoard 예시

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/go2_diff_mpc")

for it in range(num_iters):
    loss = ...                          # forward + backward
    writer.add_scalar("loss/total",       loss.item(), it)
    writer.add_scalar("loss/reward_cam",  cam_pen,     it)
    writer.add_scalar("metrics/base_z",   base_z[0],   it)
    writer.add_histogram("policy/action", action,      it)
    writer.add_histogram("dyn/M_trace",   M.diagonal(-2,-1).sum(-1), it)
```
→ `tensorboard --logdir runs` 한 줄.

#### W&B 예시 (RAL2025 스타일)

```python
import wandb
wandb.init(project="humanoid-cam", config=cfg)
wandb.log({
    "loss/total": loss.item(),
    "reward/cam_xy_penalty": cam_xy_pen,
    "reward/cam_compensation": cam_comp,
    "metrics/base_z_mean": base_z.mean().item(),
})
```

### 2.2 학습 후 / 평가 (offline)

RAL2025 의 [analysis_recorder.py](LearningHumanoidArmMotion-RAL2025-Code/extensions/humanoid/utils/analysis_recorder.py) 패턴 — **matplotlib + opencv 조합**.

#### Plot 종류와 산출물

| Plot | 산출물 | 내용 |
|---|---|---|
| **CoM velocity tracking** | `*_plot.mp4` (FuncAnimation) | CoM_vel x/y vs CoM_dvel x/y, 시간축 |
| **Step length/width** | 같은 mp4 (2-pane) | step length / step width vs desired |
| **4×4 Foot tracking error** | `mpc_foot_tracking.pdf` | (pos, vel) × (x, y, z, yaw) × (right, left) = 16 subplot |
| **Contact forces** | `mpc_contact_forces.pdf` | 3-pane: CoM_vel 명령 + RF toe/heel 3축 + LF toe/heel 3축 |
| **State error** | `mpc_state_error.pdf` | 12-dim: roll/pitch/yaw, x/y/z, ω_xyz, v_xyz 한 plot |
| **Solver comparison** | `_solver_comparison.pdf` (옵션) | ipopt vs proxqp vs customqp 결과 동시 plot |
| **Sim video** | `mpc.mp4` (cv2.VideoWriter h264) | IsaacSim viewport frames |

#### 핵심 패턴: 시계열 dict + 한 번에 plot

`defaultdict(list)` 에 매 step append → 마지막에 한 번에 그림:

```python
class AnalysisRecorder:
    def __init__(self, env):
        self.frames = []
        self.states_dict   = defaultdict(list)
        self.commands_dict = defaultdict(list)
        self.fps = int(1/env.step_dt)

    def log(self, image, states_dict, commands_dict):
        self.frames.append(image)
        for k, v in states_dict.items():
            if k == 'root_lin_vel_w':
                self.states_dict['COM_vel_x'].append(v[0].item())
                self.states_dict['COM_vel_y'].append(v[1].item())
            elif k == 'contact_forces':
                self.states_dict['F_right_toe_x'].append(v[0].item())
                # ... 12개 component 풀어서 저장
        # commands_dict 도 동일 패턴

    def save(self, folderpath):
        self.make_animation_vel_tracking(folderpath)
        self.make_plot_foot_tracking(folderpath)
        self.make_plot_contact_forces(folderpath)
        self.make_plot_state_error(folderpath)
        self.make_video(folderpath)
```

#### 시각화 스타일 컨벤션

- **시간축 ticks**: `np.arange(0, episode_length, 2*fps)` → 2초 간격, label `tick/fps` (초 단위)
- **색 컨벤션**:
  - 우측발 = 빨강 계열 (`r`, `m`)
  - 좌측발 = 파랑 계열 (`b`, `c`)
  - 명령(desired) = 검정 점선 (`k --`)
  - 실측(actual) = solid
- **저장 포맷**: PDF (plot, snapshot), MP4 with libx264 (video, animation)
- **resolution**: PDF 100 dpi, MP4 viewport 해상도 그대로

#### Animation 핵심 코드 (matplotlib FuncAnimation)

```python
from matplotlib.animation import FuncAnimation

def init():
    line_x.set_data([], [])
    line_dx.set_data([], [])
    return [line_x, line_dx]

def update(i):
    line_x.set_data(np.arange(i), data_x[:i])
    line_dx.set_data(np.arange(i), data_dx[:i])
    return [line_x, line_dx]

ani = FuncAnimation(fig, update, init_func=init,
                    frames=range(N), interval=50, blit=False)
ani.save("plot.mp4", fps=fps, extra_args=['-vcodec', 'libx264'])
```

#### Video 캡처 (cv2)

```python
import cv2
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("sim.mp4", fourcc, fps, (W, H))
for frame in self.frames:                  # frame: (H, W, 3) RGB uint8
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
out.release()
```

### 2.3 본 프로젝트 (Newton + Go2) 권장 시각화 셋업

학습/디버깅 단계별:

| 단계 | 도구 | 무엇을 볼까 |
|---|---|---|
| **초기 디버깅** | matplotlib live + Newton viewer | base trajectory, joint positions, contact patterns |
| **본격 학습** | TensorBoard | loss curve, reward components, episode length, action std |
| **실험 비교** | wandb | 여러 seed/cfg 동시 비교, hyperparam sweep |
| **최종 평가** | matplotlib FuncAnimation + cv2 video | 정책 trajectory animation, CAM/GRF plot, side-by-side mp4 |

#### 권장 metric 리스트 (학습 중 TB 로깅)

```python
writer.add_scalar("loss/policy",         ppo_policy_loss)
writer.add_scalar("loss/value",          ppo_value_loss)
writer.add_scalar("loss/entropy",        ppo_entropy)
writer.add_scalar("reward/total",        rew.mean())
writer.add_scalar("reward/cam_xy",       cam_xy_pen.mean())
writer.add_scalar("reward/cam_track",    cam_track_rew.mean())
writer.add_scalar("reward/tracking_vel", vel_track_rew.mean())
writer.add_scalar("metric/base_height",  base_z.mean())
writer.add_scalar("metric/episode_len",  episode_length.mean())
writer.add_scalar("metric/falls",        terminated.float().mean())
writer.add_histogram("action/leg",       action[..., :12])
writer.add_histogram("dyn/M_diag",       M.diagonal(-2,-1))      # 우리 GPU 커널 결과
writer.add_histogram("dyn/CAM_yaw",      h_angular[:, -1])
```

#### 권장 평가 plot 셋트 (matplotlib)

```
analysis/
  ├── traj_2d.pdf            # 평면 trajectory (x-y plot)
  ├── base_height.pdf        # z(t)
  ├── joint_tracking.pdf     # 12 joint × (pos, vel) grid
  ├── cam.pdf                # CM linear + angular vs desired
  ├── contact_forces.pdf     # GRF per foot, 3축
  ├── reward_components.pdf  # 각 reward term 시계열
  ├── eval.mp4               # 시뮬 영상
  └── animation.mp4          # 모든 plot 통합 애니메이션
```

### 2.4 Diff-Sim 학습용 시각화 (loss curve)

본 프로젝트 [go2_newton_diffsim_airborne.py](go2_newton_diffsim_airborne.py) 처럼 wp.Tape 기반 학습에서:

```python
# 매 iter
loss_val = float(self.loss.numpy()[0])
self.loss_history.append(loss_val)

# 끝나면
plt.plot(self.loss_history)
plt.xlabel("iteration"); plt.ylabel("loss"); plt.yscale("log")
plt.grid(); plt.savefig("loss.png")
```

또는 TensorBoard:
```python
writer.add_scalar("diffsim/loss", loss_val, iter)
writer.add_scalar("diffsim/grad_max", grad_max, iter)
writer.add_scalar("diffsim/lr", lr, iter)
```

---

## 부록 — 빠른 참조

### 추출 동역학 한 줄 요약
**Pinocchio의 모든 rigid body algorithms** (CRBA, RNEA, Centroidal, FK, J, Ġ) **를 SX 심볼릭으로 export → CasADi `.casadi` 파일 → GPU codegen 으로 N개 환경 한 번에 평가**.

### 분해의 핵심
**CMM의 열을 base/leg/arm 으로 partition** 하면 각 부분이 만드는 centroidal momentum 을 분리해서 reward 로 직접 활용 가능 — RAL2025의 핵심.

### 시각화의 두 트랙
1. **학습 중**: TensorBoard / W&B — scalar 시계열
2. **평가 후**: matplotlib FuncAnimation + cv2 video — 한 번에 종합 분석 mp4/pdf 산출

본 프로젝트의 [casadi-on-gpu Go2 파이프라인](go2_newton_parallel_dynamics.py) 위에 이 패턴을 그대로 얹을 수 있음.
