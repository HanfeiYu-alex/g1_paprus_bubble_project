"""Step 12 - find out what each arm joint actually does, then set a pose by eye.

Every pose in this project has been chosen by a cost function built on an
assumption that was never checked: that `elbow` bends the arm, that `roll`
swings it out, that `yaw` twists it. Only `shoulder_pitch` was ever
verified, and that was verified by watching the screen.

If `elbow` turns about an axis that rolls the forearm instead of folding
it, then commanding 1.46 rad produces a straight arm that has simply
rotated about its own length -- which is exactly what the screen keeps
showing. No amount of searching fixes a wrong assumption about what the
search variable means.

So:

    python step12_pose_tool.py --atlas

sweeps each joint on its own and prints where the elbow and wrist end up,
which says plainly what that joint does and in which direction.

    python step12_pose_tool.py --pitch -1.0 --roll 0.2 --yaw 0.0 --elbow 1.6

holds that pose with the box welded and opens the viewer, so a pose can be
chosen by looking at it. Add --headless to get the numbers without a window.
"""

import argparse
import os

import mujoco
import mujoco.viewer
import numpy as np

DEFAULT_SCENE = (
    "~/ws_g1/third_party/unitree_mujoco/unitree_robots/g1/scene_bubble.xml"
)
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
PATCHES = ["chest_upper", "chest_lower",
           "left_upper_arm", "left_lower_arm", "left_forearm",
           "right_upper_arm", "right_lower_arm", "right_forearm"]
GAINS = {"shoulder": (110., 4.), "elbow": (85., 3.), "wrist": (20., 1.),
         "hip": (150., 5.), "knee": (200., 8.), "ankle": (60., 3.),
         "waist": (150., 5.)}


def gains_for(n):
    for k, v in GAINS.items():
        if k in n:
            return v
    return 60., 2.


class Rig:
    def __init__(self, model):
        self.model = model
        self.names, self.qadr, self.vadr = [], [], []
        self.kp, self.kd, self.tmax = [], [], []
        for a in range(model.nu):
            j = model.actuator_trnid[a, 0]
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            self.names.append(n)
            self.qadr.append(model.jnt_qposadr[j])
            self.vadr.append(model.jnt_dofadr[j])
            p, d = gains_for(n)
            self.kp.append(p); self.kd.append(d)
            self.tmax.append(model.actuator_ctrlrange[a, 1])
        for nm in ("qadr", "vadr", "kp", "kd", "tmax"):
            setattr(self, nm, np.array(getattr(self, nm)))
        self.nu = model.nu

    def pose(self, pitch, roll, yaw, elbow, m_yaw=1.):
        q = np.zeros(self.nu)
        for side, s in (("left", 1.), ("right", -1.)):
            for joint, v in zip(ARM_JOINTS,
                                (pitch, s * roll, s * m_yaw * yaw, elbow)):
                jn = f"{side}_{joint}_joint"
                if jn in self.names:
                    q[self.names.index(jn)] = v
        return q

    def pd(self, data, target):
        tau = (self.kp * (target - data.qpos[self.qadr])
               - self.kd * data.qvel[self.vadr])
        return np.clip(tau, -self.tmax, self.tmax)


def xyz(model, data, body):
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    return np.array(data.xpos[b]) if b >= 0 else np.full(3, np.nan)


def atlas(model, data, rig):
    """Move one joint at a time and report what moves, and which way."""
    print("=" * 76)
    print("WHAT EACH ARM JOINT ACTUALLY DOES")
    print("=" * 76)
    print("Left arm only. Every other joint held at zero. Positions are the")
    print("elbow and wrist in world coordinates, so the direction of motion")
    print("is visible directly.\n")

    for joint in ARM_JOINTS:
        jn = f"left_{joint}_joint"
        if jn not in rig.names:
            continue
        idx = rig.names.index(jn)
        j = model.actuator_trnid[idx, 0]
        axis = np.array(model.jnt_axis[j])
        lo, hi = model.jnt_range[j]
        print(f"{jn}   local axis ({axis[0]:+.2f},{axis[1]:+.2f},"
              f"{axis[2]:+.2f})   range [{lo:+.2f}, {hi:+.2f}]")
        print(f"  {'angle':>8s} {'elbow x':>9s} {'elbow y':>9s} {'elbow z':>9s}"
              f" {'wrist x':>9s} {'wrist y':>9s} {'wrist z':>9s}"
              f" {'arm bend':>9s}")
        for v in np.linspace(max(lo, -1.6), min(hi, 2.0), 7):
            q = np.zeros(rig.nu)
            q[idx] = v
            data.qpos[:] = 0.
            data.qpos[2] = .793
            data.qpos[3] = 1.
            data.qpos[rig.qadr] = q
            mujoco.mj_forward(model, data)
            sh = xyz(model, data, "left_shoulder_roll_link")
            el = xyz(model, data, "left_elbow_link")
            wr = xyz(model, data, "left_wrist_roll_link")
            # Angle at the elbow between upper arm and forearm. 180 deg is
            # a straight arm; this is the number that says whether a joint
            # actually folds the arm.
            u = el - sh
            f = wr - el
            nu_, nf = np.linalg.norm(u), np.linalg.norm(f)
            if nu_ > 1e-6 and nf > 1e-6:
                bend = np.degrees(np.arccos(
                    np.clip(np.dot(u, f) / (nu_ * nf), -1, 1)))
                bend = 180. - bend
            else:
                bend = float("nan")
            print(f"  {v:+8.2f} {el[0]:+9.3f} {el[1]:+9.3f} {el[2]:+9.3f}"
                  f" {wr[0]:+9.3f} {wr[1]:+9.3f} {wr[2]:+9.3f}"
                  f" {bend:8.1f}d")
        print()

    print("Read the 'arm bend' column: 180 deg is a straight arm. The joint")
    print("whose bend column changes is the one that folds the arm. If that")
    print("is not `elbow`, every pose searched so far was wrong about it.")


def show(model, data, rig, args):
    target = rig.pose(args.pitch, args.roll, args.yaw, args.elbow,
                      m_yaw=args.yaw_mirror)
    weld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "box_hold")
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    base = data.qpos[:7].copy()

    adr = []
    for p in PATCHES:
        s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"touch_{p}")
        adr.append(int(model.sensor_adr[s]) if s >= 0 else -1)
    adr = np.array(adr)

    print(f"holding  pitch {args.pitch:+.2f}  roll {args.roll:+.2f}  "
          f"yaw {args.yaw:+.2f}  elbow {args.elbow:+.2f}  "
          f"(yaw mirror {args.yaw_mirror:+.0f})")
    if weld < 0:
        print("  no weld in this scene: the box will fall on its own")

    viewer = None
    if not args.headless:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.__enter__()
    spf = max(1, int((1 / 60) / model.opt.timestep))
    last = -1.
    try:
        while data.time < args.duration:
            if viewer is not None and not viewer.is_running():
                break
            for _ in range(spf):
                data.ctrl[:] = rig.pd(data, target)
                if weld >= 0:
                    data.eq_active[weld] = 1 if not args.release else 0
                mujoco.mj_step(model, data)
                data.qpos[:7] = base
                data.qvel[:6] = 0.
            if viewer is not None:
                viewer.sync()
            if data.time - last >= 1.0:
                last = data.time
                t = data.sensordata[adr]
                sh = xyz(model, data, "left_shoulder_roll_link")
                el = xyz(model, data, "left_elbow_link")
                wr = xyz(model, data, "left_wrist_roll_link")
                u, f = el - sh, wr - el
                bend = 180. - np.degrees(np.arccos(np.clip(
                    np.dot(u, f) / (np.linalg.norm(u) * np.linalg.norm(f)),
                    -1, 1)))
                print(f"  t={data.time:4.1f}s  arm bend {bend:5.1f} deg   "
                      f"touch {t.sum():7.2f} N over "
                      f"{int((t > .5).sum())}/8 patches")
                for p, v in zip(PATCHES, t):
                    if v > .5:
                        print(f"      {p:16s} {v:7.2f} N")
    finally:
        if viewer is not None:
            viewer.__exit__(None, None, None)

    print(f"\n  elbow   ({el[0]:+.3f}, {el[1]:+.3f}, {el[2]:+.3f})")
    print(f"  wrist   ({wr[0]:+.3f}, {wr[1]:+.3f}, {wr[2]:+.3f})")
    print(f"  arm bend {bend:.1f} deg   (180 is straight)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--atlas", action="store_true",
                    help="sweep each joint alone and report what it does")
    ap.add_argument("--pitch", type=float, default=-1.0)
    ap.add_argument("--roll", type=float, default=0.15)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--elbow", type=float, default=1.6)
    ap.add_argument("--yaw-mirror", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--release", action="store_true",
                    help="turn the weld off so the box must be held")
    ap.add_argument("--headless", action="store_true")
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(os.path.expanduser(args.scene))
    data = mujoco.MjData(model)
    rig = Rig(model)

    if args.atlas:
        atlas(model, data, rig)
    else:
        show(model, data, rig, args)


if __name__ == "__main__":
    main()
