### A1. Conda env

```bash
conda create -n codegen_env python=3.12 -y
conda activate codegen_env
pip install "numpy<2"   # robotpkg pinocchio = numpy 1.x ABI. 영구 유지
```

### A2. robotpkg pinocchio

```bash
sudo apt install -qqy lsb-release curl
sudo mkdir -p /etc/apt/keyrings
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.asc \
    | sudo tee /etc/apt/keyrings/robotpkg.asc
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" \
    | sudo tee /etc/apt/sources.list.d/robotpkg.list
sudo apt update
sudo apt install -y robotpkg-py312-pinocchio
```

### A2b. CUDA 13.0 toolkit (nvcc 13)

`.so` 빌드(A5/A8)에 nvcc 13 필요. mjlab_env 의 torch CUDA 13.0 과 메이저 정렬.

```bash
conda install -n codegen_env -c nvidia cuda-toolkit=13.0 -y
conda activate codegen_env
nvcc --version | grep -i release   # release 13.x 확인
```

### A3. CasADi (cuda_codegen 브랜치) → `$CONDA_PREFIX` 에 install

```bash
sudo apt install -y swig build-essential cmake pkg-config

cd ~/Mythos
git clone -b cuda_codegen https://github.com/edxmorgan/casadi.git casadi
cd casadi && rm -rf build && mkdir build && cd build   # stale CMakeCache 방지 (env 재생성 시 필수)

PY_SITELIB=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")

cmake -DWITH_PYTHON=ON -DWITH_PYTHON3=ON \
      -DPython3_EXECUTABLE=$(which python) \
      -DPython3_ROOT_DIR=$CONDA_PREFIX \
      -DPython3_FIND_STRATEGY=LOCATION \
      -DPYTHON_PREFIX=$PY_SITELIB \
      -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make -j8 && make install
```

### A4. PyTorch (CUDA 13)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
python -c "import torch; print(torch.version.cuda)"   # 13.0 (mjlab_env 와 동일)
```

### A5. casadi-on-gpu (initial toy 검증)

```bash
cd ~/Mythos
git clone https://github.com/edxmorgan/casadi-on-gpu
cd casadi-on-gpu && mkdir -p examples/assets/mine

python - <<'EOF'
import casadi as ca
q  = ca.SX.sym("q", 4)
p1 = ca.SX.sym("p1", 6)
p2 = ca.SX.sym("p2", 6)
out = ca.vertcat(ca.sin(q[0]) + p1[0],
                 ca.cos(q[1]) + p2[0],
                 q[2]*q[3], p1[1]+p2[1],
                 q[0]+q[1], q[2]+q[3])
f = ca.Function("fkeval", [q, p1, p2], [out])
f.expand()
f.save("examples/assets/mine/fkeval.casadi")
EOF

./tools/generate_manifest_and_registry.py \
  --casadi examples/assets/mine/fkeval.casadi --batch-inputs 0

mkdir -p build && cd build
cmake -DBUILD_PYTHON=ON -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
cmake --build . -j && cmake --install .
ctest -V -R casadi_on_gpu_py_smoke
```

### A7. Go2 export → A8. 커널 빌드 → mjlab_env 직접 설치

```bash
# A7. Pinocchio cpin → .casadi   (codegen_env)
cd ~/Mythos
python dynamics/export_pinocchio_casadi.py
# → dynamics/casadi_fns/go2_all_terms.casadi

# A8. casadi-on-gpu codegen + build → casadi_on_gpu.*.so
#     install prefix = mjlab_env conda prefix → site-packages 에 바로 설치 (shared/ 없음)
cd ~/Mythos/casadi-on-gpu
./tools/generate_manifest_and_registry.py \
  --casadi ~/Mythos/dynamics/casadi_fns/go2_all_terms.casadi \
  --batch-inputs 0,1,2

MJLAB_PREFIX=$(conda run -n mjlab_env python -c "import sys; print(sys.prefix)")
rm -rf build && mkdir build && cd build
cmake -DBUILD_PYTHON=ON -DCMAKE_INSTALL_PREFIX="$MJLAB_PREFIX" ..
cmake --build . -j && cmake --install .
# → $MJLAB_PREFIX/lib/python3.12/site-packages/casadi_on_gpu.*.so (+ kernels_manifest.json)
# 빌드는 codegen_env(nvcc 13), 설치 타깃은 mjlab_env. 둘 다 cpython-312 ABI·CUDA 13.0 → 직접 import
```

> **A8b (여지, 미확정)**: 배치 SRBD-MPC QP 를 codegen 경로로 갈 경우
> `dynamics/export_srbd_mpc_casadi.py` → `dynamics/casadi_fns/srbd_mpc.casadi` → A8 와 동일 절차로
> mjlab_env 에 설치. torch/warp 배치 ADMM 으로 가면 이 단계 불필요 (`mpc/batch_qp.py` 결정 대기).

### 현재 repo 구조 (Part A~C 와 별개로, 실제 코드 레이아웃)

```
Mythos/
├── dynamics/          # codegen 산출물. export_pinocchio_casadi.py + casadi_fns/*.casadi(커밋)
│                       #   export_srbd_mpc_casadi.py 는 A8b 여지(빈 파일)
├── mpc/               # SRBD-MPC 라이브러리: srbd_model/srbd_mpc(solve_batch 순수경계)/
│                       #   batch_qp/gait/reference
├── quadruped/         # 로봇: go2.py(EntityCfg) + state_adapter + urdf/ xml/  (mjlab asset_zoo 에 Go2 없음)
├── locomotion/        # mjlab RL task (velocity 관례): locomotion_env_cfg + mdp/ + rl/ + config/go2/
│                       #   mdp/: commands events curriculums observations rewards terminations
│                       #         mpc(브리지) dynamics(.so→env 브리지)
├── scripts/           # train.py play.py (비패키지, sys.path+register 배선)
└── mj_opt/            # 포팅 소스(레거시, 참조용). gitignore
```

- 결합 구조: **V1 = RL(WBC) ← SRBD-MPC plan(obs)**. 의존 방향 `locomotion/mdp → mpc → dynamics(.so)`.
  V2(RL=planner) 는 `solve_batch` 순수경계 덕에 adapter+task_id 추가만으로 공존.
- Part A 의 codegen 산출물(`go2_all_terms.casadi`)·robot(URDF/MJCF)·MPC 알고리즘 본문은
  `mj_opt/mj_sim/quadruped`(수치·단일env)에서 **배치화 + 심볼릭 codegen + mjlab manager API 로 재작성**.

# Part B — `mjlab_env` (메인 학습 런타임)

### B1. `mjlab_env`

```bash
conda create -n mjlab_env python=3.12 -y
conda activate mjlab_env
pip install mjlab

# torch 가 CUDA 13.0 인지 확인 (codegen_env 의 .so 와 메이저 정렬)
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"  # cuda 13.0

# ★ warp-lang 핀 (필수). mjlab 1.3.0 sim.py 가 옛 공개 API
#   wp.context.runtime 을 호출 → warp-lang 1.13.0 이 이를 제거해 크래시.
#   mjlab 의 uv.lock 도 1.12.0 으로 박혀 있음. mjlab 1.3.0 은 그대로 두고
#   warp 만 다운그레이드하면 됨 (GPU 병렬 스폰 정상 확인).
pip install "warp-lang==1.12.0" --extra-index-url https://pypi.nvidia.com/
python -c "import warp as wp; assert hasattr(wp,'context'); print('warp', wp.config.version, 'OK')"

# casadi_on_gpu 는 A8 에서 이 env 의 site-packages 에 직접 설치됨 (shared/·PYTHONPATH 불필요)
python -c "import casadi_on_gpu as cog; print(cog.list_kernels())"
```

### 주의

- 두 env 모두 **Python 3.12 · CUDA 13.0**. `.so` 가 cpython-312 ABI 라 codegen_env→mjlab_env 직접 이식 가능.
- `codegen_env` 는 **numpy<2 영구 유지** (robotpkg pinocchio ABI). numpy 2 런타임은 mjlab_env 에서.
- `warp-lang==1.12.0` 고정. `pip install -U mjlab` 후엔 warp 가 딸려 올라갈 수 있으니 항상 재핀.
- 모델 변경 시 A7→A8 재실행 → mjlab_env site-packages 자동 갱신.

---

# Part C — 동역학 추출 & 시각화 레퍼런스

출처: `LearningHumanoidArmMotion-RAL2025-Code` 분석 + 본 프로젝트 검증 패턴.

## C1. 동역학 항 추출 (Pinocchio → CasADi → GPU)

```
URDF
  ↓ pin.buildModelFromUrdf(..., JointModelFreeFlyer())
Pinocchio Model  ↓ cpin.Model(model)
Symbolic CasADi (cpin)
  ↓ cpin.crba / computeCoriolisMatrix / computeCentroidalMap ...
  ↓ ca.Function(...).expand()  →  f.save(*.casadi)
.casadi  ↓ generate_manifest_and_registry.py (cuda codegen)
.cu/.cuh  ↓ cmake build  →  casadi_on_gpu.*.so
  →  cog.launch(name, [in_ptrs], [out_ptrs], N)
```

추출 항목 (휴머노이드 24-DOF = base6+leg12+arm6 기준. Go2 는 18-DOF, arm 제거):

| 함수 | 의미 | 입력 | 출력 shape |
|---|---|---|---|
| M(q) | 질량행렬 | q | (n,n) |
| C(q,q̇)·q̇ | Coriolis 토크 | q,q̇ | (n,) |
| G(q) | 중력 토크 | q | (n,) |
| nle | = C·q̇+G | q,q̇ | (n,) |
| A(q)=CMM | Centroidal Momentum Matrix | q | (6,n) |
| Ȧ=dCMM | dA/dt | q,q̇ | (6,n) |
| h=A·q̇ | Centroidal Momentum | q,q̇ | (6,) |
| ḣ | Ȧq̇+Aq̈ | q,q̇,q̈ | (6,) |
| CoM(q) | 무게중심 위치 | q | (3,) |
| v_CoM | A_lin·q̇/m | q,q̇ | (3,) |
| J_CoM | CoM Jacobian | q | (3,n) |
| I_W=A_ang | World centroidal inertia | q | (3,3) |
| fk_⟨frame⟩ | FK pos+rot | q | (3,)+(3,3) |
| J_⟨frame⟩ | Frame Jacobian (LWA/LOCAL) | q | (6,n) |
| J̇_⟨frame⟩·q̇ | Jacobian 시간변화 | q,q̇ | (6,) |
| base_pos/rot | Base SE(3) | q | (3,),(3,3) |

**단일 함수로 묶기 (권장)** — RAL2025 는 9개 `.casadi` 분리, 본 프로젝트(Go2)는 1개 통합 → kernel launch 1회로 batch sim 에 유리. (`export_pinocchio_casadi.py`)

```python
f = Function(
    "go2_all_terms",
    [q, dq, theta],
    [M, g, C_dq, Ag, hg, Ig, p_com, v_com, Jcom,
     base_p, R_WB, R_WH, r_feet, R_feet, J_feet_W, Jd_dq_feet_W],
    ["q", "dq", "theta"],
    ["M","g","C_dq","Ag","hg","I_W","p_com","v_com","Jcom",
     "base_p","R_WB","R_WH","r_feet","R_feet","J_feet_W","Jd_dq_feet_W"])
f.expand()
f.save("go2_all_terms.casadi")
```

**핵심 인사이트 — CMM 분해 (base/leg/arm)**: CMM 의 열을 그룹별로 잘라 partial centroidal momentum 분리. RAL2025 의 contribution.

```python
# CMM = A(q): (N,6,n_dof)
CM      = (CMM @ qdot.unsqueeze(2)).squeeze(2)                 # 전체 h
CM_base = (CMM[:,:,0:6]   @ qdot[:,0:6].unsqueeze(2)).squeeze(2)
CM_leg  = (CMM[:,:,6:18]  @ qdot[:,6:18].unsqueeze(2)).squeeze(2)
CM_arm  = (CMM[:,:,18:24] @ qdot[:,18:24].unsqueeze(2)).squeeze(2)
CM_des  = (CMM @ qdot_desired.unsqueeze(2)).squeeze(2)         # desired

R_WB  = quat_to_R(base_quat)
block = block_diag(R_WB.T, R_WB.T)   # (6,6) → base-frame 변환
CM_bf = block @ CM
```

왜 중요한가: `CM_leg + CM_arm` = robot 이 의도적으로 만드는 angular momentum. arm 이 leg yaw momentum 을 상쇄하면 base 가 yawing 없이 안정 → reward 로 강제하면 정책이 "팔 흔들어 회전 보상" 학습. **Go2(팔 없음)**: `CM_leg` 만 분리해 base CM 안정화, 또는 `Ag[:,3:6,:]` 회전부를 직접 reward feature 로.

Reward 예시 ([rewards.py:178-196](LearningHumanoidArmMotion-RAL2025-Code/extensions/humanoid/mdp/rewards.py#L178-L196)):

```python
def CAM_xy_penalty(env, asset_cfg):                 # roll/pitch CAM 억제 (body frame)
    return -torch.sum(torch.square(env.CM_bf[:, 3:5]), dim=1)

def dCAM_xy_penalty(env, asset_cfg):                # CM 양의 시간변화만 penalize
    return -torch.clamp_min(torch.sum(env.CM_bf[:,3:5]*env.dCM_bf[:,3:5], dim=1), 0.0)

def tracking_CAM_reward(env, asset_cfg, command_name):   # desired CAM (yaw) tracking
    error  = env.CM_des[:,5] - env.CM[:,5]
    error *= 1./(1. + torch.abs(env.CM_des[:,5]))
    return torch.exp(-error**2 / (2*sigma**2))

def CAM_compensation_reward(env):                   # leg+arm yaw momentum = 0
    return torch.exp(-(env.CM_leg[:,-1] + env.CM_arm[:,-1])**2 / (2*sigma**2))
```

> 위 reward/센서 코드는 **IsaacLab API** 기반. mjlab 에선 그대로 못 쓴다 — 개념(CMM 분해·CAM·toe/heel GRF)은 유지하되 mjlab 매니저 API(`ObservationTermCfg`/`RewardTermCfg`/`Entity.data.joint_pos`) + mujoco-warp contact 로 **재작성** 필요.

**GRF (toe/heel 분리)** — IsaacLab `ContactSensor` 를 custom 확장해 per-contact-point 노출:

```python
rf_contact.data.GRF_points_buffer  # (N, max_contacts, 3) world pos
rf_contact.data.GRF_forces_buffer  # (N, max_contacts, 3) 3-axis force
rf_contact.data.GRF_count_buffer   # (N,) 유효 contact 수
```

토우/힐 분류 = 기하 근접 ([humanoid_vanilla.py:206-226](LearningHumanoidArmMotion-RAL2025-Code/extensions/humanoid/task/humanoid_vanilla.py#L206-L226)):

```python
TOL = 1e-2
toe_diff   = foot_to_toe_world.unsqueeze(1) - contact.GRF_points_buffer
toe_mask   = (toe_diff.norm(dim=-1) < TOL) * valid_mask
rf_toe_GRF = (GRF_forces * toe_mask.unsqueeze(2)).sum(dim=1)   # (N,3)
rf_GRM     = cross(rfoot_to_toe_w, rf_toe_GRF) + cross(rfoot_to_heel_w, rf_heel_GRF)
```

본 프로젝트(mjlab+Go2): mjlab 의 mujoco-warp contact API (`mjw_data.contact`) 또는 `model.contacts()` buffer 직접 활용.

**RAL2025 ↔ 본 프로젝트 매핑:**

| 항목 | RAL2025 | 본 프로젝트 |
|---|---|---|
| Symbolic | cpin (robotpkg) | cpin (robotpkg, numpy<2, codegen_env) |
| GPU backend | cusadi | casadi-on-gpu (cuda_codegen, CUDA 13) |
| 함수 묶음 | 9개 분리 | 1개 통합 (go2_all_terms) |
| 호출 | `cusadi_*_fn.evaluate([...])` | `cog.launch("go2_all_terms", [...], outs, N)` |
| 검증 | IsaacLab PhysX mass matrix 비교 | mjlab(mujoco-warp) 내부 데이터 비교 |

## C2. 시각화

**학습 중 (online):**

| 도구 | 용도 | 비고 |
|---|---|---|
| TensorBoard | scalar/curve/histogram | PyTorch `SummaryWriter` 표준 |
| W&B | 클라우드 대시보드, sweep | RAL2025 사용 |
| matplotlib live | 가장 가벼운 즉시 확인 | jupyter inline |
| rerun.io | 3D viewer + scalar 통합 | Newton 결합 사례 |
| viser | 웹 인터랙티브 3D+UI | 학습 디버깅 |

가장 자주 보는 것: loss curve (PPO surrogate/value/entropy), reward components, episode length, action stats, mass matrix condition number.

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/go2_diff_mpc")
writer.add_scalar("loss/policy",        ppo_policy_loss, it)
writer.add_scalar("loss/value",         ppo_value_loss,  it)
writer.add_scalar("loss/entropy",       ppo_entropy,     it)
writer.add_scalar("reward/total",       rew.mean(),      it)
writer.add_scalar("reward/cam_xy",      cam_xy_pen.mean(), it)
writer.add_scalar("reward/cam_track",   cam_track_rew.mean(), it)
writer.add_scalar("reward/tracking_vel",vel_track_rew.mean(), it)
writer.add_scalar("metric/base_height", base_z.mean(),   it)
writer.add_scalar("metric/episode_len", episode_length.mean(), it)
writer.add_scalar("metric/falls",       terminated.float().mean(), it)
writer.add_histogram("action/leg",      action[..., :12], it)
writer.add_histogram("dyn/M_diag",      M.diagonal(-2,-1), it)   # GPU 커널 결과
writer.add_histogram("dyn/CAM_yaw",     h_angular[:, -1], it)
# tensorboard --logdir runs
```

W&B (RAL2025 스타일): `wandb.init(project=..., config=cfg)` → `wandb.log({...})`.

**학습 후 (offline)** — RAL2025 `analysis_recorder.py` = matplotlib + opencv. `defaultdict(list)` 에 매 step append → 마지막에 한 번에 그림.

| Plot | 산출물 | 내용 |
|---|---|---|
| CoM vel tracking | mp4 (FuncAnimation) | CoM_vel xy vs dvel xy |
| Step length/width | mp4 2-pane | vs desired |
| Foot tracking error | pdf | (pos,vel)×(x,y,z,yaw)×(R,L)=16 |
| Contact forces | pdf 3-pane | CoM_vel 명령 + RF/LF toe·heel 3축 |
| State error | pdf | 12-dim rpy/xyz/ω/v |
| Sim video | mp4 (h264) | viewport frames |

스타일 컨벤션: 시간축 ticks `np.arange(0, ep_len, 2*fps)` (2초 간격, label = tick/fps). 우측발=빨강(`r,m`), 좌측발=파랑(`b,c`), desired=검정 점선(`k--`), actual=solid. 저장 PDF 100dpi / MP4 libx264.

```python
from matplotlib.animation import FuncAnimation
def init():   line_x.set_data([],[]); line_dx.set_data([],[]); return [line_x,line_dx]
def update(i):line_x.set_data(np.arange(i),data_x[:i]); line_dx.set_data(np.arange(i),data_dx[:i]); return [line_x,line_dx]
ani = FuncAnimation(fig, update, init_func=init, frames=range(N), interval=50, blit=False)
ani.save("plot.mp4", fps=fps, extra_args=['-vcodec','libx264'])

import cv2
out = cv2.VideoWriter("sim.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (W,H))
for frame in frames: out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))   # frame: (H,W,3) RGB uint8
out.release()
```
