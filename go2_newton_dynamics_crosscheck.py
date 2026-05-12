# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""
Newton ↔ casadi-on-gpu 동역학 비교.

비교 항목:
1. URDF total mass: XML 파싱 vs Pinocchio (rotor 포함 여부)
2. g(q): MuJoCo Warp qfrc_gravcomp  vs  casadi-on-gpu 출력
3. trace(M), M 대칭성

Run:
    python go2_newton_dynamics_crosscheck.py
"""

import os
import numpy as np
import torch
import warp as wp

import newton
import newton.examples
import casadi_on_gpu as cog

_URDF_CANDIDATES = [
    "/home/frlab/anaconda3/envs/batch/lib/python3.12/site-packages/genesis/assets/urdf/go2/urdf/go2.urdf",
    "/home/frlab/batch/models/go2/urdf/go2_description.urdf",
    "/home/frlab/mj_opt/models/go2/urdf/go2_description.urdf",
]
URDF_PATH = next((p for p in _URDF_CANDIDATES if os.path.isfile(p)), None)
print(f"[xcheck] URDF: {URDF_PATH}")

PIN_JOINT_ORDER = [
    "FL_hip_joint",  "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint",  "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint",  "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint",  "RR_thigh_joint", "RR_calf_joint",
]
STAND_PIN_ORDER = np.array(
    [0.0, 0.8, -1.5,  0.0, 0.8, -1.5,  0.0, 1.0, -1.5,  0.0, 1.0, -1.5],
    dtype=np.float32,
)


def urdf_total_mass(path):
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    return sum(float(m.get("value")) for m in root.findall(".//inertial/mass"))


def build_joint_reorder(joint_label, target_names):
    qi = []
    for name in target_names:
        idx = next(
            (i for i, lbl in enumerate(joint_label) if lbl.endswith(f"/{name}")),
            None,
        )
        if idx is None:
            raise ValueError(f"joint not found: {name}")
        qi.append(idx + 6)
    return qi


# --- Build single-robot model (N=1) ---
robot = newton.ModelBuilder()
newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
robot.add_urdf(URDF_PATH, xform=wp.transform(wp.vec3(0,0,0.42), wp.quat_identity()),
               floating=True, enable_self_collisions=False,
               collapse_fixed_joints=True, ignore_inertial_definitions=False)
pin_q_idx = build_joint_reorder(robot.joint_label, PIN_JOINT_ORDER)
for qi, v in zip(pin_q_idx, STAND_PIN_ORDER):
    robot.joint_q[qi] = float(v)

model = robot.finalize()
solver = newton.solvers.SolverMuJoCo(model)
state = model.state()
control = model.control()
contacts = model.contacts()
newton.eval_fk(model, model.joint_q, model.joint_qd, state)

# 한 step 돌려서 MuJoCo 내부 데이터 정상화
state_1 = model.state()
solver.step(state, state_1, control, contacts, 1e-3)
state, state_1 = state_1, state

# --- 1. URDF total mass ---
m_urdf = urdf_total_mass(URDF_PATH)
print(f"\n[1] URDF total mass: {m_urdf:.4f} kg")

# --- 2. Newton MuJoCo Warp qfrc_gravcomp ---
# qfrc_gravcomp: m * g 보상 토크. 음수 부호로 적용하면 중력 그대로.
# free joint 6dof + 12 joints = 18 dof.
qfrc_gravcomp = solver.mjw_data.qfrc_gravcomp.numpy()[0]   # (nv,) for env 0
print(f"\n[2] MuJoCo Warp qfrc_gravcomp shape: {qfrc_gravcomp.shape}")
print(f"   gravcomp[base lin (0:3)]: {qfrc_gravcomp[0:3]}")
print(f"   gravcomp[base ang (3:6)]: {qfrc_gravcomp[3:6]}")
print(f"   gravcomp[joints 6:18]   : {qfrc_gravcomp[6:18]}")

# --- 3. Our casadi kernel ---
jq  = wp.to_torch(state.joint_q).view(1, 19).clone()
jqd = wp.to_torch(state.joint_qd).view(1, 18).clone()

# Newton q layout: [pos(3), quat_wxyz(4), joints(12 in URDF order)]
# Convert to Pinocchio format
pin_q_idx_t = torch.tensor(pin_q_idx, device="cuda", dtype=torch.long)
local_joint_idx = pin_q_idx_t - 7

base_pos = jq[:, 0:3]
quat_wxyz = jq[:, 3:7]
quat_xyzw = torch.stack([quat_wxyz[:,1], quat_wxyz[:,2], quat_wxyz[:,3], quat_wxyz[:,0]], dim=-1)
joints_q = jq[:, 7:19][:, local_joint_idx]

v_w = jqd[:, 0:3]
w_w = jqd[:, 3:6]
joints_dq = jqd[:, 6:18][:, local_joint_idx]

# quat → R_BW
x, y, z, w = quat_xyzw.unbind(-1)
xx,yy,zz = x*x,y*y,z*z; xy,xz,yz = x*y,x*z,y*z; wx,wy,wz = w*x,w*y,w*z
R_WB = torch.stack([
    torch.stack([1-2*(yy+zz),   2*(xy-wz),   2*(xz+wy)], -1),
    torch.stack([  2*(xy+wz), 1-2*(xx+zz),   2*(yz-wx)], -1),
    torch.stack([  2*(xz-wy),   2*(yz+wx), 1-2*(xx+yy)], -1),
], -2)
R_BW = R_WB.transpose(-1, -2)
v_body = torch.einsum("bij,bj->bi", R_BW, v_w)
w_body = torch.einsum("bij,bj->bi", R_BW, w_w)

q_t  = torch.zeros((1, 19), device="cuda", dtype=torch.float32)
dq_t = torch.zeros((1, 18), device="cuda", dtype=torch.float32)
theta = torch.zeros((1, 1), device="cuda", dtype=torch.float32)
q_t[:,  :3] = base_pos
q_t[:, 3:7] = quat_xyzw
q_t[:, 7: ] = joints_q
dq_t[:,  :3] = v_body
dq_t[:, 3:6] = w_body
dq_t[:, 6: ] = joints_dq

sizes = [324, 18, 18, 108, 6, 9, 3, 3, 54, 3, 9, 9, 12, 36, 432, 24]
outs = [torch.empty((1, s), device="cuda", dtype=torch.float32) for s in sizes]
cog.launch("go2_all_terms",
           [q_t.data_ptr(), dq_t.data_ptr(), theta.data_ptr()],
           [o.data_ptr() for o in outs],
           1, sync=True)

M_pin = outs[0].reshape(18, 18).T.cpu().numpy()
g_pin = outs[1].squeeze(0).cpu().numpy()

print(f"\n[3] casadi-on-gpu (Pinocchio) g(q) — body frame for free joint")
print(f"   g[base lin (0:3)]: {g_pin[0:3]}")
print(f"   g[base ang (3:6)]: {g_pin[3:6]}")
print(f"   g[joints (6:18)] : {g_pin[6:18]}")

print(f"\n[4] M (Pinocchio) symmetric: {np.allclose(M_pin, M_pin.T, atol=1e-3)}")
print(f"   trace(M_pin) = {np.trace(M_pin):.4f}")
print(f"   M_pin[0:3,0:3] diag (linear inertia ≈ m): {np.diag(M_pin[:3,:3])}")
print(f"   → 추정 total mass = {np.trace(M_pin[:3,:3])/3:.4f} kg")

# --- 5. Joint-only block 비교 (free joint frame 컨벤션 차이 회피) ---
g_pin_joint  = g_pin[6:18]
g_mjwarp_joint = qfrc_gravcomp[6:18]
# joint order: Pinocchio 출력은 PIN_JOINT_ORDER, MuJoCo Warp은 URDF 순서
# URDF 순서 → Pinocchio 순서로 reorder
g_mjwarp_joint_pin = g_mjwarp_joint[np.array(local_joint_idx.cpu())]

# qfrc_gravcomp는 중력 보상 *반대 부호* (실제 적용 시 negate해서 사용한다고 가정 — convention 차이 가능)
diff_a = np.abs(g_mjwarp_joint_pin - g_pin_joint)
diff_b = np.abs(g_mjwarp_joint_pin + g_pin_joint)
print(f"\n[5] g[joints] 비교 (PIN 순서)")
print(f"   casadi g:  {g_pin_joint}")
print(f"   mjw  g:   {g_mjwarp_joint_pin}")
print(f"   |Δ| (같은 부호): max={diff_a.max():.4f}")
print(f"   |Δ| (반대 부호): max={diff_b.max():.4f}")
print(f"   → 부호 일치 여부로 컨벤션 확인")
