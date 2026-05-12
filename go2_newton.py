# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""
Newton port of go2.py — single GO2 URDF spawn + stand pose + sin motion.

Run:
    python go2_newton.py                 # GUI viewer
    python go2_newton.py --no-viewer     # headless
"""

import math
import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils

import os

# Try a few common locations
_URDF_CANDIDATES = [
    "/home/frlab/anaconda3/envs/batch/lib/python3.12/site-packages/genesis/assets/urdf/go2/urdf/go2.urdf",
    "/home/frlab/batch/models/go2/urdf/go2_description.urdf",
    "/home/frlab/mj_opt/models/go2/urdf/go2_description.urdf",
]
URDF_PATH = next((p for p in _URDF_CANDIDATES if os.path.isfile(p)), None)
if URDF_PATH is None:
    raise FileNotFoundError(f"None of these URDFs exist: {_URDF_CANDIDATES}")
print(f"[go2_newton] using URDF: {URDF_PATH}")

JOINT_NAMES = [
    "FR_hip_joint",  "FR_thigh_joint", "FR_calf_joint",
    "FL_hip_joint",  "FL_thigh_joint", "FL_calf_joint",
    "RR_hip_joint",  "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint",  "RL_thigh_joint", "RL_calf_joint",
]
STAND_Q = np.array(
    [0.0, 0.8, -1.5,  0.0, 0.8, -1.5,  0.0, 1.0, -1.5,  0.0, 1.0, -1.5],
    dtype=np.float32,
)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer

        # --- Build model ---
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
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )
        robot.add_ground_plane()

        # --- Stand pose: write q values into builder (skip 7 free-floating) ---
        for joint_name, value in zip(JOINT_NAMES, STAND_Q):
            idx = next(
                (i for i, lbl in enumerate(robot.joint_label) if lbl.endswith(f"/{joint_name}")),
                None,
            )
            if idx is None:
                raise ValueError(f"Joint '{joint_name}' not found in builder.joint_label")
            robot.joint_q[idx + 6] = float(value)

        # PD targets (used by SolverMuJoCo via control.joint_target_pos)
        for i in range(len(robot.joint_target_ke)):
            robot.joint_target_ke[i] = 150
            robot.joint_target_kd[i] = 5

        self.model = robot.finalize()
        self.solver = newton.solvers.SolverMuJoCo(self.model)

        # --- Sim timing ---
        self.fps = 50
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.sim_step = 0

        # --- States/control/contacts ---
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        self.viewer.set_model(self.model)

        # Cache joint target buffer (6 zeros for free joint + 12 joints)
        # SolverMuJoCo expects joint_target_pos shaped (n_qs,) — we'll write to it each frame
        self.target_buf = wp.zeros(self.model.joint_q.shape, dtype=wp.float32)
        self.control.joint_target_pos = self.target_buf

        self.AMPLITUDE = 0.3
        self.FREQ = 2.0
        self.stand_q_with_free = np.concatenate([np.zeros(6, dtype=np.float32), STAND_Q])

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if self.contacts is not None:
                self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        # sin offset on FR/FL thigh and RR/RL thigh
        t = self.sim_time
        offset = self.AMPLITUDE * math.sin(2.0 * math.pi * self.FREQ * t)
        ref = self.stand_q_with_free.copy()
        # JOINT_NAMES order: FR_hip,FR_thigh,FR_calf,FL_hip,FL_thigh,FL_calf,...
        # indices in stand_q_with_free are 6 + (joint slot in JOINT_NAMES)
        ref[6 + 1]  += offset   # FR_thigh
        ref[6 + 4]  += offset   # FL_thigh
        ref[6 + 7]  -= offset   # RR_thigh
        ref[6 + 10] -= offset   # RL_thigh
        self.target_buf.assign(ref)

        self.simulate()
        self.sim_time += self.frame_dt
        self.sim_step += 1

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        # 단순 검증: base가 지면 위에 있어야 함
        newton.examples.test_body_state(
            self.model, self.state_0,
            "base above ground",
            lambda q, qd: q[2] > 0.05,
            indices=[0],
        )

    @staticmethod
    def create_parser():
        return newton.examples.create_parser()


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
