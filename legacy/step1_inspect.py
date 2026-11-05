"""Step 1 - look at what is actually inside the G1 model.

Loads the MJCF and prints the parts we need before writing any control
code: how the state vector is laid out, which joints exist, what the
actuators drive, whether a standing pose is stored in the file, and which
bodies we will later attach touch sensors to.

    python step1_inspect.py
    python step1_inspect.py --model ~/ws_g1/unitree_mujoco/unitree_robots/g1/scene_29dof.xml
"""

import argparse
import os

import mujoco

JOINT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}

DEFAULT_MODEL = os.path.expanduser(
    "~/ws_g1/unitree_mujoco/unitree_robots/g1/scene.xml"
)

# The links the mentor's wrap-around carry touches, from g1_bubble.yaml.
WRAP_LINKS = [
    "torso_link",
    "waist_support_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--all-bodies", action="store_true", help="list every body")
    args = ap.parse_args()

    path = os.path.expanduser(args.model)
    print(f"loading {path}\n")
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print("=" * 62)
    print("SIZES")
    print("=" * 62)
    print(f"  nq   {model.nq:4d}   position state (qpos)")
    print(f"  nv   {model.nv:4d}   velocity state (qvel)")
    print(f"  nu   {model.nu:4d}   actuators you can command (ctrl)")
    print(f"  njnt {model.njnt:4d}   joints")
    print(f"  nbody{model.nbody:4d}   bodies")
    print(f"  nsensor {model.nsensor:4d}   sensors already defined")
    print(f"  timestep {model.opt.timestep} s"
          f"   ->  {1.0 / model.opt.timestep:.0f} Hz physics")
    print()

    # nq > nv means there is a free joint: a quaternion uses 4 numbers of
    # position but only 3 of velocity.
    has_free = any(model.jnt_type[j] == 0 for j in range(model.njnt))
    print(f"  floating base (free joint): {has_free}")
    if has_free:
        print("    -> qpos[0:3] base position, qpos[3:7] base quaternion,")
        print("       qpos[7:] joint angles. qvel[0:6] is the base twist.")
    print()

    print("=" * 62)
    print("ACTUATORS   (what you write into data.ctrl)")
    print("=" * 62)
    kinds = {}
    for a in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"act{a}"
        jnt = model.actuator_trnid[a, 0]
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt) or "?"
        lo, hi = model.actuator_ctrlrange[a]
        gain = model.actuator_gainprm[a, 0]
        bias = model.actuator_biasprm[a, 1]
        kind = "position-like (has bias)" if bias != 0 else "torque (motor)"
        kinds[kind] = kinds.get(kind, 0) + 1
        if a < 6 or a >= model.nu - 3:
            print(f"  [{a:2d}] {name:28s} -> {jname:28s} "
                  f"ctrl [{lo:7.1f},{hi:7.1f}] gain {gain:7.1f}")
        elif a == 6:
            print(f"   ... {model.nu - 9} more ...")
    print()
    for k, v in kinds.items():
        print(f"  {v} actuators are {k}")
    if "torque (motor)" in kinds:
        print("  -> torque actuators mean YOU write the PD loop. That is what")
        print("     step 2 does, and it is what the real robot does too.")
    print()

    print("=" * 62)
    print("KEYFRAMES   (stored poses you can snap to)")
    print("=" * 62)
    if model.nkey == 0:
        print("  none. Step 2 will build a standing pose by hand.")
    else:
        for k in range(model.nkey):
            kname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_KEY, k) or f"key{k}"
            print(f"  [{k}] {kname}")
        print("  -> mujoco.mj_resetDataKeyframe(model, data, 0) snaps to the first.")
    print()

    print("=" * 62)
    print("SENSORS   (what already reports data)")
    print("=" * 62)
    if model.nsensor == 0:
        print("  none. This is why we have to add touch sensors ourselves.")
    else:
        for s in range(model.nsensor):
            sname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, s) or f"s{s}"
            stype = model.sensor_type[s]
            dim = model.sensor_dim[s]
            adr = model.sensor_adr[s]
            print(f"  [{s:2d}] {sname:28s} type {stype:3d}  dim {dim}  "
                  f"-> data.sensordata[{adr}:{adr + dim}]")
    print()

    print("=" * 62)
    print("BODIES WE NEED FOR THE WRAP-AROUND CARRY")
    print("=" * 62)
    print("  (touch sensors go on these, following g1_bubble.yaml)")
    missing = []
    for link in WRAP_LINKS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
        if bid < 0:
            missing.append(link)
            print(f"  MISSING  {link}")
        else:
            pos = data.xpos[bid]
            mass = model.body_mass[bid]
            print(f"  ok [{bid:3d}] {link:28s} world xyz "
                  f"({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})  "
                  f"mass {mass:.2f} kg")
    if missing:
        print()
        print(f"  {len(missing)} link(s) missing from this model file.")
        print("  Run with --all-bodies to see what the real names are.")
    print()

    if args.all_bodies:
        print("=" * 62)
        print("ALL BODIES")
        print("=" * 62)
        for b in range(model.nbody):
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "?"
            print(f"  [{b:3d}] {bname}")
        print()

    print("=" * 62)
    print("STARTING STATE")
    print("=" * 62)
    print(f"  base height qpos[2] = {data.qpos[2]:.3f} m")
    print(f"  total model mass    = {model.body_mass.sum():.2f} kg")
    print()
    print("  If the base height is near zero the robot starts on the floor")
    print("  and collapses, which is what you saw last night. Step 2 fixes")
    print("  that by holding a pose with a PD controller.")


if __name__ == "__main__":
    main()
