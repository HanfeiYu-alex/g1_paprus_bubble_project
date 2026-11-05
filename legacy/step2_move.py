"""Step 2 - command a joint and watch it move.

The G1's 29 actuators are torque motors: data.ctrl is newton-metres, not an
angle. To hold an angle you write the loop yourself,

    tau = kp * (q_target - q) - kd * qdot

which is exactly what the real robot's low-level controller runs, and what
the walking policy sits on top of.

The base is pinned in the air so nothing falls over while you learn the arm.
Balance is a separate problem and it is solved later by the policy, not here.

    # hold every joint at zero, pinned, nothing moving
    python step2_move.py

    # sweep one joint and watch which way it goes
    python step2_move.py --joint left_shoulder_pitch_joint --amp 0.6

    # list the joints you can name
    python step2_move.py --list

    # let it fall on its feet instead of hanging (it will collapse)
    python step2_move.py --no-pin
"""

import argparse
import math
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

DEFAULT_MODEL = "~/ws_g1/third_party/unitree_mujoco/unitree_robots/g1/scene.xml"

# Torque motors need gains. Heavy joints that carry the body get more, the
# wrists get very little because their torque limit is only 5 Nm.
GAINS = [
    ("hip", 150.0, 5.0),
    ("knee", 200.0, 8.0),
    ("ankle", 60.0, 3.0),
    ("waist", 150.0, 5.0),
    ("shoulder", 80.0, 3.0),
    ("elbow", 60.0, 2.0),
    ("wrist", 15.0, 0.8),
]


def gains_for(joint_name: str) -> tuple[float, float]:
    for key, kp, kd in GAINS:
        if key in joint_name:
            return kp, kd
    return 60.0, 2.0


class Robot:
    """Thin wrapper that maps actuator index -> joint state addresses."""

    def __init__(self, model):
        self.model = model
        self.nu = model.nu
        self.joint_names = []
        self.qpos_adr = np.zeros(model.nu, dtype=int)
        self.qvel_adr = np.zeros(model.nu, dtype=int)
        self.kp = np.zeros(model.nu)
        self.kd = np.zeros(model.nu)
        self.tau_limit = np.zeros(model.nu)

        for a in range(model.nu):
            jnt = model.actuator_trnid[a, 0]
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt)
            self.joint_names.append(name)
            # A joint's angle does not live at index a. It lives wherever the
            # model put it, and with a floating base that is offset by 7.
            # Asking the model is the only safe way.
            self.qpos_adr[a] = model.jnt_qposadr[jnt]
            self.qvel_adr[a] = model.jnt_dofadr[jnt]
            kp, kd = gains_for(name)
            self.kp[a] = kp
            self.kd[a] = kd
            self.tau_limit[a] = model.actuator_ctrlrange[a, 1]

    def index_of(self, joint_name: str) -> int:
        if joint_name not in self.joint_names:
            raise SystemExit(
                f"no actuated joint named {joint_name!r}.\n"
                "run with --list to see the names."
            )
        return self.joint_names.index(joint_name)

    def q(self, data) -> np.ndarray:
        return data.qpos[self.qpos_adr]

    def qdot(self, data) -> np.ndarray:
        return data.qvel[self.qvel_adr]

    def pd(self, data, target: np.ndarray) -> np.ndarray:
        """The whole controller. Three lines, and the real robot's is too."""
        tau = self.kp * (target - self.q(data)) - self.kd * self.qdot(data)
        return np.clip(tau, -self.tau_limit, self.tau_limit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--joint", default=None, help="joint to sweep")
    ap.add_argument("--amp", type=float, default=0.5, help="sweep amplitude [rad]")
    ap.add_argument("--freq", type=float, default=0.25, help="sweep frequency [Hz]")
    ap.add_argument("--no-pin", action="store_true", help="let the base fall")
    ap.add_argument("--list", action="store_true", help="print joint names and exit")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(os.path.expanduser(args.model))
    data = mujoco.MjData(model)
    robot = Robot(model)

    if args.list:
        print(f"{robot.nu} actuated joints:\n")
        for a, name in enumerate(robot.joint_names):
            kp, kd = robot.kp[a], robot.kd[a]
            print(f"  [{a:2d}] {name:28s} qpos[{robot.qpos_adr[a]:2d}] "
                  f"qvel[{robot.qvel_adr[a]:2d}]  "
                  f"kp {kp:5.0f} kd {kd:4.1f}  |tau| <= {robot.tau_limit[a]:5.1f}")
        return

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    # Pinning: remember the starting base pose, and put it back after every
    # step. This is a teaching scaffold, not physics -- it quietly injects
    # whatever force is needed to hold the base still. It is here so that arm
    # motion can be studied without balance in the way.
    base_qpos = data.qpos[:7].copy()
    if not args.no_pin:
        base_qpos[2] += 0.35  # lift it clear of the floor

    target = np.zeros(robot.nu)
    swept = robot.index_of(args.joint) if args.joint else None

    print(f"physics {1 / model.opt.timestep:.0f} Hz, {robot.nu} actuators")
    if swept is None:
        print("holding every joint at 0. pass --joint NAME to sweep one.")
    else:
        print(f"sweeping [{swept}] {args.joint}: "
              f"{args.amp:+.2f} rad at {args.freq} Hz")
        print("watch which way it moves -- that tells you the sign convention.")
    print(f"base pinned: {not args.no_pin}")
    print()

    steps_per_frame = max(1, int((1 / 60) / model.opt.timestep))
    last_print = 0.0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        wall0 = time.time()
        while viewer.is_running():
            for _ in range(steps_per_frame):
                if swept is not None:
                    target[swept] = args.amp * math.sin(
                        2 * math.pi * args.freq * data.time
                    )
                data.ctrl[:] = robot.pd(data, target)
                mujoco.mj_step(model, data)

                if not args.no_pin:
                    data.qpos[:7] = base_qpos
                    data.qvel[:6] = 0.0

            viewer.sync()

            if swept is not None and data.time - last_print > 0.25:
                last_print = data.time
                q = robot.q(data)[swept]
                tau = data.ctrl[swept]
                err = target[swept] - q
                print(f"  t={data.time:6.2f}s  target={target[swept]:+.3f} rad"
                      f"  actual={q:+.3f}  error={err:+.3f}"
                      f"  tau={tau:+7.2f} / {robot.tau_limit[swept]:.0f} Nm")

            # keep the viewer near real time
            lag = (data.time) - (time.time() - wall0)
            if lag > 0:
                time.sleep(lag)


if __name__ == "__main__":
    main()
