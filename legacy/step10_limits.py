"""Step 10 - what the arm joints are actually allowed to do, and whether the
commanded pose is reached.

Every failure so far has been diagnosed from the solver's own forward
kinematics, which answers "where would the patch be at these angles". It
never answers "can the robot hold these angles". Those differ for two
reasons and both have been ignored until now:

  1. Joint limits. The search explored shoulder yaw over +/-1.6 rad and the
     elbow over 0.4..2.3 rad without ever reading model.jnt_range. If the
     real joint stops short, the physics clamps it and the arm ends up
     somewhere the solver never considered.

  2. Torque limits and gravity. A pose can be inside the limits and still
     not be held, if the PD gain cannot supply the torque the pose needs.

The symptom in run6 was exactly this shape: the solver reported the patches
sitting on the box surface (sdf -0.001), and the simulation reported zero
contact for the whole settle phase. Those two statements cannot both
describe the same arm.

    python step10_limits.py                        # limits + tracking test
    python step10_limits.py --pose -0.775 -0.155 -0.605 1.830
"""

import argparse
import os

import mujoco
import numpy as np

DEFAULT_SCENE = (
    "~/ws_g1/third_party/unitree_mujoco/unitree_robots/g1/scene_bubble.xml"
)
ARM_JOINTS = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow")
PATCHES = ["chest_upper", "chest_lower",
           "left_upper_arm", "left_lower_arm", "left_forearm",
           "right_upper_arm", "right_lower_arm", "right_forearm"]
GAINS = {"shoulder": (110., 4.), "elbow": (85., 3.), "wrist": (15., .8),
         "hip": (150., 5.), "knee": (200., 8.), "ankle": (60., 3.),
         "waist": (150., 5.)}


def gains_for(n):
    for k, v in GAINS.items():
        if k in n:
            return v
    return 60., 2.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--pose", type=float, nargs=4,
                    default=[-0.775, -0.155, -0.605, 1.830],
                    metavar=("PITCH", "ROLL", "YAW", "ELBOW"))
    ap.add_argument("--yaw-mirror", type=float, default=-1.0)
    ap.add_argument("--settle", type=float, default=3.0)
    args = ap.parse_args()

    model = mujoco.MjModel.from_xml_path(os.path.expanduser(args.scene))
    data = mujoco.MjData(model)

    names, qadr, vadr, kp, kd, tmax = [], [], [], [], [], []
    for a in range(model.nu):
        j = model.actuator_trnid[a, 0]
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        names.append(n)
        qadr.append(model.jnt_qposadr[j])
        vadr.append(model.jnt_dofadr[j])
        p, d = gains_for(n)
        kp.append(p); kd.append(d)
        tmax.append(model.actuator_ctrlrange[a, 1])
    qadr = np.array(qadr); vadr = np.array(vadr)
    kp = np.array(kp); kd = np.array(kd); tmax = np.array(tmax)

    print("=" * 72)
    print("ARM JOINT LIMITS  (model.jnt_range, in radians and degrees)")
    print("=" * 72)
    print(f"  {'joint':28s} {'min':>8s} {'max':>8s}  {'min':>7s} {'max':>7s}"
          f"  {'tau max':>8s}")
    limits = {}
    for a, n in enumerate(names):
        if not any(k in n for k in ARM_JOINTS):
            continue
        j = model.actuator_trnid[a, 0]
        lo, hi = model.jnt_range[j]
        limited = bool(model.jnt_limited[j])
        limits[n] = (lo, hi, limited)
        flag = "" if limited else "   (unlimited)"
        print(f"  {n:28s} {lo:+8.3f} {hi:+8.3f}  {np.degrees(lo):+7.1f} "
              f"{np.degrees(hi):+7.1f}  {tmax[a]:8.1f}{flag}")

    print()
    print("=" * 72)
    print("IS THE REQUESTED POSE EVEN LEGAL?")
    print("=" * 72)
    pitch, roll, yaw, elbow = args.pose
    m = args.yaw_mirror
    bad = 0
    for side, s in (("left", 1.), ("right", -1.)):
        for joint, v in zip(ARM_JOINTS, (pitch, s * roll, s * m * yaw, elbow)):
            jn = f"{side}_{joint}_joint"
            if jn not in limits:
                continue
            lo, hi, limited = limits[jn]
            if limited and not (lo <= v <= hi):
                clamp = min(max(v, lo), hi)
                print(f"  OUT OF RANGE  {jn:28s} asked {v:+.3f}, "
                      f"limit [{lo:+.3f}, {hi:+.3f}] -> clamps to {clamp:+.3f}"
                      f"   ({np.degrees(v - clamp):+.1f} deg lost)")
                bad += 1
            else:
                print(f"  ok            {jn:28s} asked {v:+.3f}")
    if bad == 0:
        print("\n  Every commanded angle is inside its limit.")
    else:
        print(f"\n  {bad} commanded angles are outside the joint limits.")
        print("  The solver was searching poses the robot cannot adopt, so")
        print("  its patch positions describe an arm that never existed.")

    print()
    print("=" * 72)
    print("DOES THE PD CONTROLLER ACTUALLY GET THERE?")
    print("=" * 72)
    target = np.zeros(model.nu)
    for side, s in (("left", 1.), ("right", -1.)):
        for joint, v in zip(ARM_JOINTS, (pitch, s * roll, s * m * yaw, elbow)):
            jn = f"{side}_{joint}_joint"
            if jn in names:
                target[names.index(jn)] = v

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    base = data.qpos[:7].copy()
    weld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "box_hold")

    steps = int(args.settle / model.opt.timestep)
    for _ in range(steps):
        tau = kp * (target - data.qpos[qadr]) - kd * data.qvel[vadr]
        data.ctrl[:] = np.clip(tau, -tmax, tmax)
        if weld >= 0:
            data.eq_active[weld] = 1
        mujoco.mj_step(model, data)
        data.qpos[:7] = base
        data.qvel[:6] = 0.

    print(f"  after {args.settle:.1f} s of settling with the box welded\n")
    print(f"  {'joint':28s} {'asked':>8s} {'got':>8s} {'error':>8s} "
          f"{'tau':>8s} {'limit':>8s}")
    worst = 0.
    for a, n in enumerate(names):
        if not any(k in n for k in ARM_JOINTS):
            continue
        got = float(data.qpos[qadr[a]])
        err = target[a] - got
        worst = max(worst, abs(err))
        mark = "  <-- saturated" if abs(data.ctrl[a]) > 0.97 * tmax[a] else ""
        print(f"  {n:28s} {target[a]:+8.3f} {got:+8.3f} {err:+8.3f} "
              f"{data.ctrl[a]:+8.2f} {tmax[a]:8.1f}{mark}")
    print(f"\n  worst tracking error {worst:+.3f} rad "
          f"({np.degrees(worst):.1f} deg)")
    if worst > 0.15:
        print("  The arm is NOT where it was told to go. Any plan built on")
        print("  the commanded angles is describing a different robot.")

    print()
    print("=" * 72)
    print("WHERE THE PATCHES REALLY ENDED UP")
    print("=" * 72)
    print(f"  {'patch':16s} {'x':>8s} {'y':>8s} {'z':>8s}   touch")
    adr = {}
    for p in PATCHES:
        s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"touch_{p}")
        adr[p] = int(model.sensor_adr[s]) if s >= 0 else -1
    for p in PATCHES:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"bubble_{p}")
        v = data.site_xpos[sid]
        f = data.sensordata[adr[p]] if adr[p] >= 0 else float("nan")
        print(f"  {p:16s} {v[0]:+8.3f} {v[1]:+8.3f} {v[2]:+8.3f}  "
              f"{f:7.2f} N")

    box = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    bg = model.body_geomadr[box]
    c = np.array(data.xpos[box]); h = np.array(model.geom_size[bg])
    print(f"\n  box spans x {c[0]-h[0]:+.3f}..{c[0]+h[0]:+.3f}   "
          f"y {-h[1]:+.3f}..{h[1]:+.3f}   z {c[2]-h[2]:+.3f}..{c[2]+h[2]:+.3f}")

    print(f"\n  {data.ncon} contacts in the scene")
    for cidx in range(data.ncon):
        g1 = int(data.contact[cidx].geom1); g2 = int(data.contact[cidx].geom2)
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                               int(model.geom_bodyid[g1]))
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                               int(model.geom_bodyid[g2]))
        if "box" in (n1, n2):
            print(f"    {n1} <-> {n2}   depth {data.contact[cidx].dist:+.4f}")


if __name__ == "__main__":
    main()
