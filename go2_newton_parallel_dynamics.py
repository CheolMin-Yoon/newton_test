# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""
Newton port of go2_parallel_dynamics.py — N개 Go2 병렬 sim + casadi-on-gpu 동역학 커널.

매 step: state.joint_q / joint_qd → (q[N,19], dq[N,18]) torch 텐서 → cog.launch("go2_all_terms", ...)
→ 출력 (M, g, C_dq, Ag, hg, I_W, p_com, v_com, Jcom, base_p, R_WB, R_WH, r_feet, R_feet, J_feet_W, Jd_dq_feet_W)

주의:
- Newton/MuJoCo Warp free joint: qvel은 **world frame** (Pinocchio LOCAL과 다름) → R_BW로 회전
- Newton joint_q quat: wxyz (Pinocchio xyzw) → reorder
- Joint order: URDF 파싱 순서에 의존 → joint_label에서 동적 추출

Run:
    python go2_newton_parallel_dynamics.py --world-count 16
"""

import math
import os
import numpy as np
import torch
import warp as wp

import newton
import newton.examples
import newton.utils
import casadi_on_gpu as cog

_URDF_CANDIDATES = [
    "/home/frlab/anaconda3/envs/batch/lib/python3.12/site-packages/genesis/assets/urdf/go2/urdf/go2.urdf",
    "/home/frlab/batch/models/go2/urdf/go2_description.urdf",
    "/home/frlab/mj_opt/models/go2/urdf/go2_description.urdf",
]
URDF_PATH = next((p for p in _URDF_CANDIDATES if os.path.isfile(p)), None)
if URDF_PATH is None:
    raise FileNotFoundError(f"URDF not found in {_URDF_CANDIDATES}")
print(f"[dyn] URDF: {URDF_PATH}")

# Pinocchio 측 조인트 순서 (export_pinocchio_casadi.py가 만든 함수의 순서)
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
NQ_PER = 19   # 3 pos + 4 quat + 12 joints
NV_PER = 18   # 6 free + 12 joints


def build_joint_reorder(joint_label, target_names):
    """
    Newton joint_label에서 'something/<joint_name>' 형태 라벨을 검색해서
    target_names 순서로 Newton joint q 인덱스 (label_idx + 6 → q idx) 반환.
    free joint 빼고 12개에 대한 q 인덱스 리스트.
    """
    qi = []
    for name in target_names:
        idx = next(
            (i for i, lbl in enumerate(joint_label) if lbl.endswith(f"/{name}")),
            None,
        )
        if idx is None:
            raise ValueError(f"joint not found: {name}")
        qi.append(idx + 6)   # +6 보정 (anymal example과 동일)
    return qi


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.world_count = args.world_count

        # --- Build single Go2 ---
        robot = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
        robot.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06, limit_ke=1.0e3, limit_kd=1.0e1,
        )
        robot.default_shape_cfg.ke = 5.0e4
        robot.default_shape_cfg.kd = 5.0e2
        robot.default_shape_cfg.kf = 1.0e3
        robot.default_shape_cfg.mu = 0.75
        robot.add_urdf(
            URDF_PATH,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.42), wp.quat_identity()),
            floating=True, enable_self_collisions=False,
            collapse_fixed_joints=True, ignore_inertial_definitions=False,
        )

        # Newton URDF joint 순서 발견 → Pinocchio 순서로 매핑
        self.pin_q_idx = build_joint_reorder(robot.joint_label, PIN_JOINT_ORDER)
        print(f"[dyn] Newton q indices (Pinocchio order): {self.pin_q_idx}")

        # Stand pose (Pinocchio 순서로 STAND 값을 Newton joint_q 위치에 기록)
        for qi, value in zip(self.pin_q_idx, STAND_PIN_ORDER):
            robot.joint_q[qi] = float(value)

        for i in range(len(robot.joint_target_ke)):
            robot.joint_target_ke[i] = 150
            robot.joint_target_kd[i] = 5

        # --- Master builder + replicate ---
        builder = newton.ModelBuilder()
        builder.replicate(robot, self.world_count, spacing=(2.0, 2.0, 0.0))
        builder.add_ground_plane()

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverMuJoCo(self.model)

        # --- Sim timing ---
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.viewer.set_model(self.model)

        # --- target buffer (free 7 zeros + stand 12 in Pinocchio order at pin_q_idx) per env ---
        self.target_buf = wp.zeros(self.model.joint_q.shape[0], dtype=wp.float32)
        # 초기 1회 baseline 값 채워 두기
        baseline_one = np.zeros(NQ_PER, dtype=np.float32)
        for qi, v in zip(self.pin_q_idx, STAND_PIN_ORDER):
            baseline_one[qi] = v       # qi는 single robot의 절대 q idx (free 7 이후)
        self.baseline_tiled = np.tile(baseline_one, self.world_count)
        self.target_buf.assign(self.baseline_tiled)
        self.control.joint_target_pos = self.target_buf

        # Per-env phase for sin motion
        self.phase_per_env = np.linspace(0.0, 2.0 * math.pi, self.world_count, dtype=np.float32)
        self.AMPLITUDE = 0.3
        self.FREQ = 2.0

        # --- casadi-on-gpu 출력 버퍼 ---
        sizes = [324, 18, 18, 108, 6, 9, 3, 3, 54, 3, 9, 9, 12, 36, 432, 24]
        self.out_names = ["M","g","C_dq","Ag","hg","I_W","p_com","v_com","Jcom",
                          "base_p","R_WB","R_WH","r_feet","R_feet","J_feet_W","Jd_dq_feet_W"]
        self.outs = [torch.empty((self.world_count, s), device="cuda", dtype=torch.float32)
                     for s in sizes]
        self.q_buf  = torch.empty((self.world_count, NQ_PER), device="cuda", dtype=torch.float32)
        self.dq_buf = torch.empty((self.world_count, NV_PER), device="cuda", dtype=torch.float32)
        self.theta  = torch.zeros((self.world_count, 1),      device="cuda", dtype=torch.float32)

        # Pinocchio 조인트 인덱스를 torch 텐서로 (한 env 안에서)
        self.pin_q_idx_t = torch.tensor(self.pin_q_idx, device="cuda", dtype=torch.long)

        # 첫 frame에 한 번 검증
        self.verified = False

    # ----------------------- state assembly -----------------------
    def _quat_wxyz_to_xyzw(self, q_wxyz):
        return torch.stack([q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3], q_wxyz[:, 0]], dim=-1)

    def _quat_to_R_BW(self, q_xyzw):
        x, y, z, w = q_xyzw.unbind(-1)
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        R = torch.stack([
            torch.stack([1-2*(yy+zz),   2*(xy-wz),   2*(xz+wy)], -1),
            torch.stack([  2*(xy+wz), 1-2*(xx+zz),   2*(yz-wx)], -1),
            torch.stack([  2*(xz-wy),   2*(yz+wx), 1-2*(xx+yy)], -1),
        ], -2)
        return R.transpose(-1, -2)

    def assemble(self):
        # wp arrays → torch
        jq_flat  = wp.to_torch(self.state_0.joint_q).view(self.world_count, NQ_PER)
        jqd_flat = wp.to_torch(self.state_0.joint_qd).view(self.world_count, NV_PER)

        base_pos       = jq_flat[:, 0:3]
        base_quat_wxyz = jq_flat[:, 3:7]
        joints_q       = jq_flat[:, 7:19]   # Newton URDF order (= replicate된 robot 순서)

        # Pinocchio joint 순서로 reorder
        # pin_q_idx_t는 single-robot 절대 q index (7..18 범위) — 7 빼서 0..11로 변환
        local_joint_idx = self.pin_q_idx_t - 7
        joints_q_pin = joints_q[:, local_joint_idx]

        # qvel layout: Newton/MuJoCo free joint qvel = [lin_w(3), ang_w(3)] world frame
        v_world = jqd_flat[:, 0:3]
        w_world = jqd_flat[:, 3:6]
        joints_dq = jqd_flat[:, 6:18]
        joints_dq_pin = joints_dq[:, local_joint_idx]

        # quat wxyz → xyzw
        q_xyzw = self._quat_wxyz_to_xyzw(base_quat_wxyz)
        R_BW = self._quat_to_R_BW(q_xyzw)
        v_body = torch.einsum("bij,bj->bi", R_BW, v_world)
        w_body = torch.einsum("bij,bj->bi", R_BW, w_world)

        # q[N,19], dq[N,18] 채우기
        self.q_buf[:,  :3] = base_pos
        self.q_buf[:, 3:7] = q_xyzw
        self.q_buf[:, 7: ] = joints_q_pin
        self.dq_buf[:,  :3] = v_body
        self.dq_buf[:, 3:6] = w_body
        self.dq_buf[:, 6: ] = joints_dq_pin

    def launch_kernel(self):
        stream = torch.cuda.current_stream().cuda_stream
        cog.launch(
            "go2_all_terms",
            [self.q_buf.data_ptr(), self.dq_buf.data_ptr(), self.theta.data_ptr()],
            [o.data_ptr() for o in self.outs],
            self.world_count,
            stream_ptr=stream,
            sync=False,
        )

    # ----------------------- sim loop -----------------------
    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if self.contacts is not None:
                self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # sin offset 적용
        t = self.sim_time
        offset = self.AMPLITUDE * np.sin(2.0 * math.pi * self.FREQ * t + self.phase_per_env)
        ref = self.baseline_tiled.copy()
        for e in range(self.world_count):
            base = e * NQ_PER
            # FR_thigh / FL_thigh / RR_thigh / RL_thigh slots in PIN_JOINT_ORDER
            # PIN order: FL(0,1,2) FR(3,4,5) RL(6,7,8) RR(9,10,11)
            # thigh slots = 1, 4, 7, 10
            for slot, sign in [(1, +1), (4, +1), (7, +1), (10, +1)]:
                ref[base + self.pin_q_idx[slot]] += sign * offset[e]
        self.target_buf.assign(ref)

        self.simulate()

        # 우리 동역학 커널
        self.assemble()
        self.launch_kernel()

        # 첫 frame 검증
        if not self.verified:
            torch.cuda.synchronize()
            M_view = self.outs[0].reshape(self.world_count, 18, 18).transpose(-1, -2)
            sym_ok = torch.allclose(M_view[0], M_view[0].T, atol=1e-3)
            any_nan = any(torch.isnan(o).any().item() or torch.isinf(o).any().item() for o in self.outs)
            print(f"[verify] M[0] symmetric: {sym_ok}   any nan/inf: {any_nan}")
            self.verified = True

        # 매 100 step마다 env 0 진단
        n_step = int(self.sim_time / self.frame_dt)
        if n_step % 100 == 0 and n_step > 0:
            torch.cuda.synchronize()
            g0 = self.outs[1][0]
            M0 = self.outs[0][0].reshape(18, 18).T
            print(f"[step {n_step:4d}] g_base_z={g0[2].item():+.3f}  trace(M)={M0.diag().sum().item():.3f}")

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model, self.state_0,
            "all bases above ground",
            lambda q, qd: q[2] > 0.05,
        )

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        parser.set_defaults(world_count=16)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
