# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""
Go2 Diff-Sim — '공중 부양' 시나리오 (ground 없음).

목적: rigid body + joint torque → state 의 미분 가능성을 발 접촉 문제를 회피해서 검증.
ground plane 제거 + Go2를 z=2.0에 부양. drone 예제처럼 contact-free → wp.Tape backward 통과 기대.

Run:
    python go2_newton_diffsim_airborne.py --horizon 10 --no-viewer
"""

import os
import numpy as np
import warp as wp

import newton
import newton.examples

_URDF_CANDIDATES = [
    "/home/frlab/anaconda3/envs/batch/lib/python3.12/site-packages/genesis/assets/urdf/go2/urdf/go2.urdf",
    "/home/frlab/batch/models/go2/urdf/go2_description.urdf",
]
URDF_PATH = next((p for p in _URDF_CANDIDATES if os.path.isfile(p)), None)
print(f"[diff-air] URDF: {URDF_PATH}")

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


@wp.kernel
def loss_kernel(joint_q: wp.array[float], loss: wp.array[float], qz_index: int):
    loss[0] = -joint_q[qz_index]


@wp.kernel
def joint_target_loss_kernel(joint_q: wp.array[float],
                             target_q: wp.array[float],
                             dof_idx: wp.array[int],
                             loss: wp.array[float]):
    j = wp.tid()
    qi = dof_idx[j]
    # joint_q index에서 free joint 7 entries 이후가 joint pos
    diff = joint_q[qi] - target_q[j]
    wp.atomic_add(loss, 0, diff * diff)


@wp.kernel
def step_kernel(x: wp.array[float], grad: wp.array[float], alpha: float):
    tid = wp.tid()
    x[tid] = x[tid] - grad[tid] * alpha


@wp.kernel
def _scatter_kernel(tau: wp.array2d[float], t: int,
                    dst_idx: wp.array[int], dst: wp.array[float]):
    j = wp.tid()
    dst[dst_idx[j]] = tau[t, j]


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer

        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.06, limit_ke=1.0e3, limit_kd=1.0e1,
        )
        builder.add_urdf(
            URDF_PATH,
            xform=wp.transform(wp.vec3(0,0,2.0), wp.quat_identity()),  # 공중 부양 z=2.0
            floating=True, enable_self_collisions=False,
            collapse_fixed_joints=True, ignore_inertial_definitions=False,
        )
        # ★ add_ground_plane() 호출 X — 지면 없음

        # stand pose
        for name, value in zip(PIN_JOINT_ORDER, STAND_PIN_ORDER):
            idx = next((i for i, lbl in enumerate(builder.joint_label) if lbl.endswith(f"/{name}")), None)
            builder.joint_q[idx + 6] = float(value)

        self.model = builder.finalize(requires_grad=True)
        # Featherstone: articulated rigid body 전용 solver, joint_f input + requires_grad 명시적 지원
        self.solver = newton.solvers.SolverFeatherstone(self.model)

        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_steps = args.horizon
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        n_states = self.sim_steps * self.sim_substeps + 1
        self.states = [self.model.state() for _ in range(n_states)]
        self.control = self.model.control()
        self.contacts = None   # 공중이라 contact 불필요
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])

        if self.viewer is not None:
            self.viewer.set_model(self.model)

        self.n_act = 12
        self.tau = wp.zeros((self.sim_steps, self.n_act),
                            dtype=wp.float32, requires_grad=True)

        act_dof_idx = []
        for name in PIN_JOINT_ORDER:
            idx = next(i for i, lbl in enumerate(builder.joint_label) if lbl.endswith(f"/{name}"))
            act_dof_idx.append(idx + 5)
        self.act_dof_idx_wp = wp.array(act_dof_idx, dtype=int)
        print(f"[diff-air] act_dof_idx: {act_dof_idx}")

        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)
        self.qz_index = 2

        # joint position target — stand pose에서 +0.5 rad 이동시키는 목표
        # 우리 joint dof idx는 act_dof_idx (PIN 순서, free 6 이후 6..17 범위에 매핑)
        # joint_q에서 해당 위치 = act_dof_idx + 1 (joint_q는 free 7개 후 joint 12개)
        self.target_dof_idx = wp.array([i + 1 for i in act_dof_idx], dtype=int)  # joint_q idx
        target_vals = np.array(STAND_PIN_ORDER, dtype=np.float32) + 0.5  # 목표 q
        self.target_q = wp.array(target_vals, dtype=wp.float32)

        self.train_iter = 0
        self.train_rate = 50.0     # ↑↑ grad 작아서 큰 lr
        self.loss_history = []

    def _apply_torque(self, t):
        self.control.joint_f.zero_()
        wp.launch(_scatter_kernel, dim=self.n_act,
                  inputs=[self.tau, t, self.act_dof_idx_wp, self.control.joint_f])

    def simulate(self):
        for sim_step in range(self.sim_steps):
            self._apply_torque(sim_step)
            for sub in range(self.sim_substeps):
                t = sim_step * self.sim_substeps + sub
                self.states[t].clear_forces()
                self.solver.step(self.states[t], self.states[t+1], self.control, self.contacts, self.sim_dt)
        # ★ joint target loss: |joint_q(final) - target|^2
        self.loss.zero_()
        wp.launch(joint_target_loss_kernel, dim=self.n_act,
                  inputs=[self.states[-1].joint_q, self.target_q, self.target_dof_idx, self.loss])
        return self.loss

    def forward_backward(self):
        self.tape = wp.Tape()
        with self.tape:
            self.simulate()
        self.tape.backward(self.loss)

    def step(self):
        self.forward_backward()

        loss_val = float(self.loss.numpy()[0])
        if self.tau.grad is not None:
            g_np = self.tau.grad.numpy()
            grad_max = float(np.nanmax(np.abs(g_np))) if np.isfinite(g_np).any() else float("nan")
            grad_has_nan = bool(np.isnan(g_np).any())
        else:
            grad_max, grad_has_nan = 0.0, False

        print(f"[iter {self.train_iter:3d}] loss={loss_val:+.4f}  |grad|max={grad_max:.4e}  nan={grad_has_nan}")
        self.loss_history.append(loss_val)

        valid = (self.tau.grad is not None) and (not grad_has_nan) and (grad_max > 1e-12)
        if valid:
            CLIP = 1e2
            g_clipped = np.clip(g_np, -CLIP, CLIP)
            self.tau.grad.assign(g_clipped)
            wp.launch(step_kernel, dim=self.tau.size,
                      inputs=[self.tau.flatten(), self.tau.grad.flatten(), self.train_rate])

        self.tape.zero()
        self.train_iter += 1

    def render(self):
        if self.viewer is None:
            return
        self.viewer.begin_frame(0.0)
        self.viewer.log_state(self.states[-1])
        self.viewer.end_frame()

    def test_final(self):
        pass

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--horizon", type=int, default=10)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
