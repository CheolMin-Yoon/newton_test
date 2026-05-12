# SPDX-FileCopyrightText: Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0
"""
Newton port of go2_parallel.py — N개 Go2를 replicate으로 병렬 spawn.

Run:
    python go2_newton_parallel.py --world-count 16
    python go2_newton_parallel.py --world-count 64 --no-viewer
"""

import math
import os
import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils

_URDF_CANDIDATES = [
    "/home/frlab/anaconda3/envs/batch/lib/python3.12/site-packages/genesis/assets/urdf/go2/urdf/go2.urdf",
    "/home/frlab/batch/models/go2/urdf/go2_description.urdf",
    "/home/frlab/mj_opt/models/go2/urdf/go2_description.urdf",
]
URDF_PATH = next((p for p in _URDF_CANDIDATES if os.path.isfile(p)), None)
if URDF_PATH is None:
    raise FileNotFoundError(f"None of these URDFs exist: {_URDF_CANDIDATES}")
print(f"[go2_newton_parallel] URDF: {URDF_PATH}")

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
# Per-robot q size: 3(pos) + 4(quat) + 12(joints) = 19
NQ_PER = 19


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
            floating=True,
            enable_self_collisions=False,
            collapse_fixed_joints=True,
            ignore_inertial_definitions=False,
        )

        # Stand pose in single robot builder (replicate will copy to all N)
        for joint_name, value in zip(JOINT_NAMES, STAND_Q):
            idx = next(
                (i for i, lbl in enumerate(robot.joint_label) if lbl.endswith(f"/{joint_name}")),
                None,
            )
            if idx is None:
                raise ValueError(f"Joint '{joint_name}' not found")
            robot.joint_q[idx + 6] = float(value)

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

        # --- States/control/contacts ---
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.viewer.set_model(self.model)

        # Per-step joint target buffer (shape = (world_count * 19,))
        n_q_total = self.model.joint_q.shape[0]
        assert n_q_total == self.world_count * NQ_PER, \
            f"unexpected joint_q size {n_q_total}, expected {self.world_count}*{NQ_PER}"
        self.target_buf = wp.zeros(n_q_total, dtype=wp.float32)
        self.control.joint_target_pos = self.target_buf

        # Per-robot joint_q = 3(pos) + 4(quat) + 12(joints) = 19; free joint occupies 7 entries
        baseline_one = np.concatenate([np.zeros(7, dtype=np.float32), STAND_Q])  # 19
        self.baseline_tiled = np.tile(baseline_one, self.world_count)            # (N*19,)

        # Per-env phase for sin motion
        self.phase_per_env = np.linspace(0.0, 2.0 * math.pi, self.world_count, dtype=np.float32)

        self.AMPLITUDE = 0.3
        self.FREQ = 2.0

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if self.contacts is not None:
                self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        t = self.sim_time
        # offset per env  (N,)
        offset = self.AMPLITUDE * np.sin(2.0 * math.pi * self.FREQ * t + self.phase_per_env)

        # Build full target buffer (N * 19,)
        ref = self.baseline_tiled.copy()
        # 12-joint block within each robot starts at idx 6 (skip 6 free)
        # joint slots in JOINT_NAMES: FR_thigh=1, FL_thigh=4, RR_thigh=7, RL_thigh=10
        for e in range(self.world_count):
            base = e * NQ_PER + 7   # skip 7 (3 pos + 4 quat) free joint q's
            ref[base + 1]  += offset[e]   # FR_thigh
            ref[base + 4]  += offset[e]   # FL_thigh
            ref[base + 7]  -= offset[e]   # RR_thigh
            ref[base + 10] -= offset[e]   # RL_thigh
        self.target_buf.assign(ref)

        self.simulate()
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
