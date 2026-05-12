# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""
Go2 Differentiable Sim 테스트 (Newton + Warp Tape).

Genesis rigid에서 backward가 hang났던 같은 시나리오를 Newton의 wp.Tape 으로 시도.
이 패턴은 newton/examples/diffsim/example_diffsim_ball.py 와 동일하게 검증된 길.

검증:
  - per-step joint torque (또는 control target) 에 grad를 붙이고
  - HORIZON step 굴린 후 final base height에 대한 loss를 계산
  - tape.backward() 가 통과하고 grad가 0이 아니어야 OK

Run:
    python go2_newton_diffsim.py
"""

import os
import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils

_URDF_CANDIDATES = [
    "/home/frlab/anaconda3/envs/batch/lib/python3.12/site-packages/genesis/assets/urdf/go2/urdf/go2.urdf",
    "/home/frlab/batch/models/go2/urdf/go2_description.urdf",
]
URDF_PATH = next((p for p in _URDF_CANDIDATES if os.path.isfile(p)), None)
print(f"[diff] URDF: {URDF_PATH}")

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
    # base z 위치 = joint_q[qz_index]  (env 0의 free joint pos z)
    loss[0] = -joint_q[qz_index]   # base를 위로 올리는 게 목표 → loss는 음의 z


@wp.kernel
def step_kernel(x: wp.array[float], grad: wp.array[float], alpha: float):
    tid = wp.tid()
    x[tid] = x[tid] - grad[tid] * alpha


def build_robot():
    robot = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(robot)
    robot.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        armature=0.06, limit_ke=1.0e3, limit_kd=1.0e1,
    )
    robot.add_urdf(
        URDF_PATH,
        xform=wp.transform(wp.vec3(0,0,0.42), wp.quat_identity()),
        floating=True, enable_self_collisions=False,
        collapse_fixed_joints=True, ignore_inertial_definitions=False,
    )
    # stand pose
    for name, value in zip(PIN_JOINT_ORDER, STAND_PIN_ORDER):
        idx = next((i for i, lbl in enumerate(robot.joint_label) if lbl.endswith(f"/{name}")), None)
        robot.joint_q[idx + 6] = float(value)
    return robot


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer

        builder = build_robot()
        # diff sim 활성화 — 핵심
        self.model = builder.finalize(requires_grad=True)

        # MuJoCo solver는 grad 지원 — example_diffsim_ball.py 도 SolverSemiImplicit 사용
        # rigid + grad가 검증된 솔버 사용
        self.solver = newton.solvers.SolverSemiImplicit(
            self.model,
            joint_attach_ke=1600.0,
            joint_attach_kd=20.0,
        )

        # --- Sim timing ---
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_steps = args.horizon
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps

        # state 시퀀스 (ball 예제처럼 매 substep마다 state 보관, backward용)
        n_states = self.sim_steps * self.sim_substeps + 1
        self.states = [self.model.state() for _ in range(n_states)]
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])

        if self.viewer is not None:
            self.viewer.set_model(self.model)

        # --- 미분 대상: per-step joint torque (12 joints × HORIZON steps) ---
        self.n_act = 12
        self.tau = wp.zeros((self.sim_steps, self.n_act),
                            dtype=wp.float32, requires_grad=True)

        # actuated joint dof index map: PIN order → control.joint_f index
        # joint_f shape == nv. free joint는 0..5, joints는 6..17.
        # PIN order에 맞춰 12개 슬롯 추출
        self.act_dof_idx = []
        for name in PIN_JOINT_ORDER:
            idx = next(i for i, lbl in enumerate(builder.joint_label) if lbl.endswith(f"/{name}"))
            self.act_dof_idx.append(idx + 5)   # +5: free joint occupies 6 dof; label_idx 1..12 → dof 6..17
        self.act_dof_idx_wp = wp.array(self.act_dof_idx, dtype=int)
        print(f"[diff] act_dof_idx: {self.act_dof_idx}")

        # loss buffer
        self.loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

        # base z의 q index = 2 (env 0 free joint position z)
        self.qz_index = 2

        self.train_iter = 0
        self.train_rate = 1e-3
        self.loss_history = []

    def _apply_torque(self, t):
        """torque tau[t] (12,) 를 control.joint_f 의 actuated slots에 적재."""
        # control.joint_f 는 (nv,) 전체. 우리는 actuated 12개 위치만 채움.
        joint_f = self.control.joint_f
        joint_f.zero_()
        # 단순 indexing으로 채우기 — wp.copy로 actuated slots에
        wp.launch(
            kernel=_scatter_kernel,
            dim=self.n_act,
            inputs=[self.tau, t, self.act_dof_idx_wp, joint_f],
        )

    def simulate(self):
        for sim_step in range(self.sim_steps):
            self._apply_torque(sim_step)
            for sub in range(self.sim_substeps):
                t = sim_step * self.sim_substeps + sub
                self.states[t].clear_forces()
                self.solver.step(self.states[t], self.states[t+1], self.control, self.contacts, self.sim_dt)

        # final loss
        wp.launch(loss_kernel, dim=1,
                  inputs=[self.states[-1].joint_q, self.loss, self.qz_index])
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

        print(f"[iter {self.train_iter:3d}] loss={loss_val:+.4f}  |grad|max={grad_max:.4e}  nan_in_grad={grad_has_nan}")
        self.loss_history.append(loss_val)

        # NaN/inf 가 끼면 optimizer 건너뜀 + tau는 직전 값 유지 (NaN propagation 차단)
        valid = (self.tau.grad is not None) and (not grad_has_nan) and (grad_max > 1e-12) and np.isfinite(grad_max)

        # 추가 안전 장치: grad clip (per-element)
        CLIP = 1e2
        if valid:
            g_clipped = np.clip(g_np, -CLIP, CLIP)
            self.tau.grad.assign(g_clipped)
            wp.launch(step_kernel, dim=self.tau.size,
                      inputs=[self.tau.flatten(), self.tau.grad.flatten(), self.train_rate])
        elif grad_has_nan:
            # tau에 NaN이 끼었을 수 있어 한 번 reset
            tau_np = self.tau.numpy()
            if np.isnan(tau_np).any() or np.isinf(tau_np).any():
                print("  [reset] tau had NaN/Inf — reset to zero")
                self.tau.zero_()

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
        parser.add_argument("--horizon", type=int, default=10,
                            help="number of sim steps per forward pass (smaller=faster backward)")
        return parser


@wp.kernel
def _scatter_kernel(tau: wp.array2d[float], t: int,
                    dst_idx: wp.array[int], dst: wp.array[float]):
    j = wp.tid()
    dst[dst_idx[j]] = tau[t, j]


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
