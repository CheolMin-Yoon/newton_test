## 1. Conda env

```bash
conda create -n batch python=3.12 -y
conda activate batch
pip install "numpy<2"
```

## 2. robotpkg pinocchio

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

## 3. CasADi (cuda_codegen)

```bash
sudo apt install -y swig build-essential cmake pkg-config

cd ~/batch
git clone -b cuda_codegen https://github.com/edxmorgan/casadi.git casadi
cd casadi
mkdir -p build && cd build

PY_SITELIB=$(python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")

cmake -DWITH_PYTHON=ON -DWITH_PYTHON3=ON \
      -DPython3_EXECUTABLE=$(which python) \
      -DPython3_ROOT_DIR=$CONDA_PREFIX \
      -DPython3_FIND_STRATEGY=LOCATION \
      -DPYTHON_PREFIX=$PY_SITELIB \
      -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make -j8
make install
```

## 4. PyTorch

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

## 5. casadi-on-gpu (initial toy)

```bash
cd ~/batch
git clone https://github.com/edxmorgan/casadi-on-gpu
cd casadi-on-gpu
mkdir -p examples/assets/mine

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
  --casadi examples/assets/mine/fkeval.casadi \
  --batch-inputs 0

mkdir -p build && cd build
cmake -DBUILD_PYTHON=ON -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
cmake --build . -j
cmake --install .
ctest -V -R casadi_on_gpu_py_smoke
```

## 6. cpin ↔ cuda_codegen casadi 호환성 확인

```bash
python - <<'EOF'
import sys
sys.path.insert(0, '/opt/openrobots/lib/python3.12/site-packages')
from pinocchio import casadi as cpin
import pinocchio as pin
import casadi as ca
import numpy as np

model = pin.buildSampleModelManipulator()
cmodel = cpin.Model(model)
cdata = cmodel.createData()

q  = ca.SX.sym("q",  model.nq)
dq = ca.SX.sym("dq", model.nv)
M = cpin.crba(cmodel, cdata, q)
g = cpin.computeGeneralizedGravity(cmodel, cdata, q)

f = ca.Function("toy_dyn", [q, dq], [M, g])
f.expand()
f.save("/tmp/toy_dyn.casadi")

f2 = ca.Function.load("/tmp/toy_dyn.casadi")
print("loaded:", f2)
print("eval:", f2(np.zeros(model.nq), np.zeros(model.nv)))
EOF
```

## 7. Go2 export

```bash
cd ~/batch/mj_sim
python quadruped/core/export_pinocchio_casadi.py
```

## 8. Go2 kernel build

```bash
cd ~/batch/casadi-on-gpu
./tools/generate_manifest_and_registry.py \
  --casadi ~/batch/mj_sim/quadruped/core/casadi_fns/go2_all_terms.casadi \
  --batch-inputs 0,1,2

rm -rf build && mkdir build && cd build
cmake -DBUILD_PYTHON=ON -DCMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
cmake --build . -j && cmake --install .
```

## 9. GPU 배치 검증

```bash
python - <<'EOF'
import torch, casadi_on_gpu as cog
print("kernels:", cog.list_kernels())

N = 4096
q     = torch.randn((N, 19), device="cuda", dtype=torch.float32)
dq    = torch.randn((N, 18), device="cuda", dtype=torch.float32)
theta = torch.zeros((N, 1),  device="cuda", dtype=torch.float32)

sizes = [324, 18, 18, 108, 6, 9, 3, 3, 54, 3, 9, 9, 12, 36, 432, 24]
names = ["M","g","C_dq","Ag","hg","I_W","p_com","v_com","Jcom",
         "base_p","R_WB","R_WH","r_feet","R_feet","J_feet_W","Jd_dq_feet_W"]
outs = [torch.empty((N, s), device="cuda", dtype=torch.float32) for s in sizes]

stream = torch.cuda.current_stream().cuda_stream
cog.launch("go2_all_terms",
           [q.data_ptr(), dq.data_ptr(), theta.data_ptr()],
           [o.data_ptr() for o in outs],
           N,
           stream_ptr=stream,
           sync=True)

for n, s, o in zip(names, sizes, outs):
    print(f"  {n:>14}  flat={s:4d}  nan={torch.isnan(o).any().item()}  inf={torch.isinf(o).any().item()}")

M = outs[0].reshape(N, 18, 18).transpose(-1, -2)
print("M[0] symmetric:", torch.allclose(M[0], M[0].T, atol=1e-3))
EOF
```

## 10. CPU/GPU 수치 일치 확인

```bash
python - <<'EOF'
import torch, numpy as np, casadi as ca, casadi_on_gpu as cog

f_cpu = ca.Function.load("/home/frlab/batch/mj_sim/quadruped/core/casadi_fns/go2_all_terms.casadi")

np.random.seed(0)
q_np  = np.random.randn(19).astype(np.float32)
dq_np = np.random.randn(18).astype(np.float32)
th_np = np.zeros(1, dtype=np.float32)
M_cpu = np.array(f_cpu(q_np, dq_np, th_np)[0])

N = 1
q   = torch.tensor(q_np,  device="cuda").unsqueeze(0)
dq  = torch.tensor(dq_np, device="cuda").unsqueeze(0)
th  = torch.tensor(th_np, device="cuda").unsqueeze(0)

sizes = [324, 18, 18, 108, 6, 9, 3, 3, 54, 3, 9, 9, 12, 36, 432, 24]
outs = [torch.empty((N, s), device="cuda", dtype=torch.float32) for s in sizes]

stream = torch.cuda.current_stream().cuda_stream
cog.launch("go2_all_terms",
           [q.data_ptr(), dq.data_ptr(), th.data_ptr()],
           [o.data_ptr() for o in outs],
           N, stream_ptr=stream, sync=True)

M_gpu = outs[0].reshape(18,18).cpu().numpy().T
diff = np.abs(M_gpu - M_cpu).max()
print("max |M_gpu - M_cpu|:", diff)
print("OK" if diff < 1e-3 else "MISMATCH")
EOF
```

## 11. numpy 2 복귀 (Genesis 호환)

```bash
pip install -U "numpy>=2"
```
