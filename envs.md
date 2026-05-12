# 3-Env 워크플로 전략

본 프로젝트는 **MuJoCo, casadi-on-gpu, Newton** (또는 IsaacLab) 을 분업으로 사용한다.
각 도구가 자기 자리에 있을 때 가장 효율적 — 한 env에 다 욱여넣으려 하지 말 것.

---

## 핵심 철학

| 도구 | 본질 | 역할 |
|---|---|---|
| **MuJoCo** | Ground truth dynamics, Python REPL | 단일환경 디버깅, 알고리즘 설계 |
| **casadi-on-gpu** | Sim-agnostic 분석적 동역학 | `.casadi` 가 portable artifact — 어디서든 호출 |
| **Newton / IsaacLab** | 병렬 sim | RL 학습 / diff-sim MPC 의 양산 라인 |

`.casadi` + `.so` 두 artifact 가 세 env 를 잇는 **portable bridge**.

---

## 환경 1: `batch_gen` — Codegen 공장

`.casadi` 생성 + `.so` 빌드. 모델이 바뀔 때만 가끔 돌림.

### 구성
- python 3.12
- **numpy < 2** (robotpkg pinocchio가 numpy 1.x ABI로 컴파일됨)
- robotpkg pinocchio (`/opt/openrobots`, cpin 포함)
- cuda_codegen casadi (`edxmorgan/casadi:cuda_codegen` 브랜치)
- swig, cmake, gcc, nvcc, pybind11

### 설치 (요약)
```bash
conda create -n batch_gen python=3.12 -y
conda activate batch_gen
pip install "numpy<2"

# robotpkg pinocchio (apt)
# 자세한 절차는 batch.md §2 참조
sudo apt install -y robotpkg-py312-pinocchio swig

# cuda_codegen casadi 빌드 → $CONDA_PREFIX 에 install
# 자세한 절차는 batch.md §3 참조
```

### 작업 흐름
```bash
conda activate batch_gen

# 1) Pinocchio cpin 으로 .casadi export
python export_pinocchio_casadi.py
# → casadi_fns/go2_all_terms.casadi 생성

# 2) casadi-on-gpu codegen → .cu/.cuh + 매니페스트
cd ~/batch/casadi-on-gpu
./tools/generate_manifest_and_registry.py \
  --casadi ~/batch/shared/casadi_fns/go2_all_terms.casadi \
  --batch-inputs 0,1,2

# 3) cmake build → casadi_on_gpu.cpython-312-*.so
rm -rf build && mkdir build && cd build
cmake -DBUILD_PYTHON=ON -DCMAKE_INSTALL_PREFIX=$HOME/batch/shared ..
cmake --build . -j
cmake --install .
```

### 산출물
```
~/batch/shared/
├── casadi_fns/
│   └── go2_all_terms.casadi
└── lib/python3.12/site-packages/
    ├── casadi_on_gpu.cpython-312-x86_64-linux-gnu.so
    └── kernels_manifest.json
```

### 빈도
- URDF 변경 시
- 동역학 함수 추가/제거 시 (예: arm 추가, randomization 파라미터 도입)
- 평소엔 **건들지 않음**

---

## 환경 2: `batch` — 메인 학습/시뮬 런타임

매일 작업하는 env. Newton (또는 IsaacLab) 으로 병렬 sim + 학습.

### 구성
- python 3.12 (`batch_gen` 과 minor 버전 일치 필수)
- **numpy >= 2**
- torch + torchvision (CUDA 12.8 wheel)
- newton-physics (또는 IsaacLab)
- (Mujoco도 옵션으로 깔아둬도 무방)

### 설치
```bash
conda create -n batch python=3.12 -y
conda activate batch
pip install "numpy>=2"
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install newton-physics  # 또는 IsaacLab 절차
```

### `batch_gen` 산출물 가져오기
```bash
# activate.d 에 PYTHONPATH 추가 — env 활성화 시 자동
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/shared_libs.sh <<'EOF'
export PYTHONPATH="$HOME/batch/shared/lib/python3.12/site-packages:$PYTHONPATH"
EOF
conda deactivate && conda activate batch
python -c "import casadi_on_gpu as cog; print(cog.list_kernels())"
```

### 작업 흐름
```bash
conda activate batch
python go2_newton_parallel_dynamics.py --world-count 64
python go2_newton_diffsim_airborne.py --horizon 5
# RL 학습 스크립트 ...
```

### 빈도
- **매일** (학습, 평가, 디버깅)

---

## 환경 3: `mj_mpc` — MuJoCo 단일환경 MPC/디버깅

알고리즘 설계, controller prototyping, MPC 알고리즘 검증.
Python REPL / Jupyter 친화적, 빠른 iteration.

### 구성
- python 3.12 (또는 무관)
- mujoco (pip)
- pinocchio (pip 일반판, no cuda_codegen — 단순 dynamics 확인용)
- casadi (pip 일반판)
- matplotlib, scipy
- 본인의 `mj_sim/quadruped/core/Pinocchio_Wrapper`

### 설치
```bash
conda create -n mj_mpc python=3.12 -y
conda activate mj_mpc
pip install numpy mujoco pin casadi matplotlib scipy
# `mj_sim/` 코드는 별도 install 없이 PYTHONPATH 만 추가
```

### `batch_gen` 산출물 가져오기 (선택)
`mj_mpc` 에서 GPU 커널 쓰고 싶으면 동일하게 PYTHONPATH 추가.
다만 보통은 단일 env 디버깅이라 numpy/torch 만으로 충분.

### 작업 흐름
- Jupyter 노트북 / Python REPL 에서:
  - Pinocchio_Wrapper 로 단일 step dynamics 확인
  - MPC 알고리즘 (직접 구현 OSQP, proxsuite, casadi 의 NLP solver 등) prototyping
  - 매 step state inspection, breakpoint
- 단일 환경이라 디버그/print 자유

### 빈도
- 새 알고리즘 / 컨트롤러 시작할 때
- `batch` 학습이 안 풀릴 때 single-env 로 돌아와서 원인 분리
- 실로봇 transfer 직전 sanity check

---

## env 간 자산 흐름

```
         (URDF / 모델 정의 변경)
                 │
                 ▼
        ┌──────────────────┐
        │   batch_gen      │  ← 가끔 돌림 (모델 바뀔 때만)
        │  numpy<2, cpin   │
        │ + cuda_codegen   │
        └────────┬─────────┘
                 │ produces
       ┌─────────┴──────────┐
       │                    │
       ▼                    ▼
  .casadi 파일       casadi_on_gpu.so
       │                    │
       └──────┬─────────────┘
              │ (shared via ~/batch/shared/)
              │
     ┌────────┴─────────┐
     ▼                  ▼
 ┌──────┐         ┌──────────┐
 │batch │         │ mj_mpc   │
 │(매일)│         │(설계 시) │
 └──────┘         └──────────┘
   학습             MPC 검증
   eval
   sim
```

---

## 주의할 점

1. **Python ABI 일치**: 세 env 모두 **Python 3.12.x** (같은 minor 버전). `.so` 가 `cpython-312` 라 3.11/3.13 이면 import 안 됨.
2. **CUDA driver / toolkit**: `batch_gen` 에서 빌드한 CUDA 버전과 호환되는 driver 가 `batch` 머신에도 있어야 함.
3. **`.casadi` 직렬화 호환**: cuda_codegen casadi 브랜치는 vanilla casadi 와 직렬화 포맷이 다를 수 있음. **`batch_gen` 에서 만든 `.casadi` 는 `batch` 의 casadi 로 load 가능한지** 1회 검증 필요 (`Function.load(...)`).
4. **모델 변경 시 워크플로**:
   - URDF 수정 → `batch_gen` → `.casadi` 재생성 → `.so` rebuild → `batch` / `mj_mpc` 에서 자동 반영
   - **CI 스크립트** 로 한 번에 돌리는 게 편함:
     ```bash
     # rebuild_all.sh
     conda activate batch_gen
     python export_pinocchio_casadi.py
     cd ~/batch/casadi-on-gpu
     ./tools/generate_manifest_and_registry.py --casadi ... --batch-inputs ...
     rm -rf build && mkdir build && cd build
     cmake -DBUILD_PYTHON=ON -DCMAKE_INSTALL_PREFIX=$HOME/batch/shared ..
     cmake --build . -j && cmake --install .
     ```
5. **버전 lock**: casadi 같은 핵심 의존성은 git commit hash 까지 박아둬야 재현 가능.

---

## 한 줄 요약

**`batch_gen` 은 공장 (가끔 가동), `batch` 와 `mj_mpc` 는 그 공장 생산물 (`.casadi` + `.so`) 을 쓰는 라인 (매일 가동).**

각 env 가 numpy 버전과 의존성 충돌 없이 자기 일에 집중하게 분리하는 게 가장 안정적인 운영 방식.
