"""Step 14 - inspect the OpenHomie scene before integrating anything into it.

The hug work was built on unitree_mujoco's g1_29dof.xml with the floating
base pinned. The walking policy is an ONNX file whose input width is fixed
at 6 x (10 + 2*n + 12); with n = 27 that is 456, so the policy only accepts
the 27-joint OpenHomie model. The integration therefore has to move onto
THAT model, which means the bubble pads, the posture search and the contact
tests all have to be re-derived there.

None of that can be written from guesswork. This dumps everything the
integration depends on:

  * joint names, order, qpos/dof addresses and limits
  * actuator names and torque limits
  * whether the arm joints follow the same naming as our code assumes
  * the body names the sensor placement needs
  * the existing box and the `grab` equality
  * the torso's collision envelope, measured from its geometry
  * the arm segment vectors
  * the ONNX policy's real input and output shapes

    python step14_inspect_homie.py
    python step14_inspect_homie.py --xml /path/to/my_scene.xml
"""

import argparse
import os

import mujoco
import numpy as np

ROOT = os.path.expanduser("~/g1/OpenHomie")
XML = os.path.join(
    ROOT, "HomieRL/legged_gym/resources/robots/g1_description/my_scene.xml")
POLICY = os.path.join(ROOT, "HomieDeploy/deploy.onnx")

# What our hug code assumes exists. Anything missing here has to be renamed
# or remapped before the two halves can be joined.
WANT_BODIES = ["torso_link",
               "left_shoulder_pitch_link", "left_shoulder_roll_link",
               "left_shoulder_yaw_link", "left_elbow_link",
               "left_wrist_roll_link",
               "right_shoulder_pitch_link", "right_shoulder_roll_link",
               "right_shoulder_yaw_link", "right_elbow_link",
               "right_wrist_roll_link",
               "box"]
WANT_JOINTS = [f"{s}_{j}_joint"
               for s in ("left", "right")
               for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                         "elbow")]
ARM_SEGMENTS = [("left_shoulder_yaw_link", "left_elbow_link"),
                ("left_elbow_link", "left_wrist_roll_link"),
                ("right_shoulder_yaw_link", "right_elbow_link"),
                ("right_elbow_link", "right_wrist_roll_link")]


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def geom_corners(model, g):
    t = int(model.geom_type[g])
    s = np.array(model.geom_size[g], dtype=float)
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(model.geom_dataid[g])
        if mid < 0:
            return None
        a = int(model.mesh_vertadr[mid]); n = int(model.mesh_vertnum[mid])
        if n <= 0:
            return None
        v = np.array(model.mesh_vert[a:a + n], dtype=float).reshape(-1, 3)
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        v = np.array([[sx * s[0], sy * s[1], sz * s[2]]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        v = np.array([[sx * s[0], sy * s[0], sz * s[0]]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        r, hl = s[0], s[1]
        hzz = hl + (r if t == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.)
        v = np.array([[sx * r, sy * r, sz * hzz]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        v = np.array([[sx * s[0], sy * s[1], sz * s[2]]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    else:
        return None
    R = quat_to_mat(np.array(model.geom_quat[g], dtype=float))
    return (R @ v.T).T + np.array(model.geom_pos[g], dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=XML)
    ap.add_argument("--policy", default=POLICY)
    args = ap.parse_args()

    xml = os.path.expanduser(args.xml)
    if not os.path.isfile(xml):
        raise SystemExit(f"{xml} not found. Pass --xml with the real path.")
    print(f"scene: {xml}\n")
    print("files beside it:")
    for f in sorted(os.listdir(os.path.dirname(xml))):
        print(f"    {f}")

    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    print()
    print("=" * 74)
    print("SIZES")
    print("=" * 74)
    print(f"  nq {model.nq}   nv {model.nv}   nu {model.nu}   "
          f"njnt {model.njnt}   nbody {model.nbody}   neq {model.neq}")
    free = [j for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
    print(f"  free joints: "
          + ", ".join(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
                      or f"j{j}" for j in free))
    print(f"  walk_g1.py computes n = nq - 14 = {model.nq - 14}; the policy "
          f"expects 6 x (10 + 2n + 12)")
    print(f"  = {6 * (10 + 2 * (model.nq - 14) + 12)} inputs")

    print()
    print("=" * 74)
    print("ACTUATORS, in ctrl order")
    print("=" * 74)
    print(f"  {'i':>3s} {'actuator':28s} {'joint':28s} {'qadr':>5s} "
          f"{'dadr':>5s} {'lo':>7s} {'hi':>7s} {'tau':>7s}")
    jname_of_act = {}
    for a in range(model.nu):
        j = int(model.actuator_trnid[a, 0])
        an = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or ""
        jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        jname_of_act[jn] = a
        lo, hi = model.jnt_range[j]
        lim = model.jnt_limited[j]
        print(f"  {a:3d} {an:28s} {jn:28s} {model.jnt_qposadr[j]:5d} "
              f"{model.jnt_dofadr[j]:5d} "
              + (f"{lo:+7.3f} {hi:+7.3f}" if lim else f"{'free':>7s} "
                 f"{'free':>7s}")
              + f" {model.actuator_ctrlrange[a, 1]:7.1f}")

    print()
    print("=" * 74)
    print("DOES OUR HUG CODE FIND WHAT IT NEEDS?")
    print("=" * 74)
    missing = []
    for b in WANT_BODIES:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, b)
        print(f"  body   {b:30s} {'ok' if i >= 0 else 'MISSING'}")
        if i < 0:
            missing.append(b)
    for jn in WANT_JOINTS:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        extra = ""
        if i >= 0:
            extra = (f"  ctrl index {jname_of_act[jn]}"
                     if jn in jname_of_act else "  NOT ACTUATED")
        print(f"  joint  {jn:30s} {'ok' if i >= 0 else 'MISSING'}{extra}")
        if i < 0:
            missing.append(jn)
    print(f"\n  {len(missing)} names missing"
          + ("" if not missing else "  -> these need remapping"))

    print()
    print("=" * 74)
    print("EQUALITY CONSTRAINTS")
    print("=" * 74)
    for e in range(model.neq):
        en = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_EQUALITY, e) or ""
        t = int(model.eq_type[e])
        b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                               int(model.eq_obj1id[e])) or "?"
        b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                               int(model.eq_obj2id[e])) or "world"
        print(f"  {en:20s} type {t}  {b1} <-> {b2}  active0 "
              f"{int(model.eq_active0[e])}")

    print()
    print("=" * 74)
    print("THE BOX ALREADY IN THE SCENE")
    print("=" * 74)
    bb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    if bb < 0:
        print("  no body named 'box'")
    else:
        print(f"  world pos {np.array(data.xpos[bb])}")
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) != bb:
                continue
            gn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}"
            s = np.array(model.geom_size[g])
            print(f"  geom {gn:20s} type {int(model.geom_type[g])}  "
                  f"half-size {s}  full {2*s}  mass {model.body_mass[bb]:.3f}")

    print()
    print("=" * 74)
    print("TORSO COLLISION ENVELOPE  (where the chest pad must go)")
    print("=" * 74)
    tb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    if tb < 0:
        print("  no torso_link")
    else:
        lo = np.full(3, np.inf); hi = np.full(3, -np.inf); used = 0
        for g in range(model.ngeom):
            if int(model.geom_bodyid[g]) != tb:
                continue
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            gn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}"
            c = geom_corners(model, g)
            how = "exact"
            if c is None or len(c) == 0:
                r = float(model.geom_rbound[g])
                p = np.array(model.geom_pos[g], dtype=float)
                c = np.array([p - r, p + r]); how = "rbound"
            lo = np.minimum(lo, c.min(axis=0))
            hi = np.maximum(hi, c.max(axis=0))
            used += 1
            print(f"  geom {gn:24s} type {int(model.geom_type[g]):2d}  {how}")
        if used:
            Rt = np.array(data.xmat[tb]).reshape(3, 3)
            org = np.array(data.xpos[tb])
            print(f"  {used} collidable geoms")
            print(f"  torso frame:  x {lo[0]:+.3f}..{hi[0]:+.3f}   "
                  f"y {lo[1]:+.3f}..{hi[1]:+.3f}   z {lo[2]:+.3f}..{hi[2]:+.3f}")
            fw = org + Rt @ np.array([hi[0], 0., 0.])
            print(f"  world at rest: chest surface x {fw[0]:+.3f}, "
                  f"torso origin z {org[2]:+.3f}")
        else:
            print("  torso_link has NO collidable geometry")

    print()
    print("=" * 74)
    print("ARM SEGMENTS  (the capsule pads follow these)")
    print("=" * 74)
    for host, child in ARM_SEGMENTS:
        hb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, host)
        cb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, child)
        if hb < 0 or cb < 0:
            print(f"  {host:28s} SKIP (missing body)")
            continue
        seg = np.array(model.body_pos[cb], dtype=float)
        print(f"  {host:28s} -> {child:24s} len {np.linalg.norm(seg):.4f} m"
              f"   world {np.array(data.xpos[hb])}")
    for s in ("left", "right"):
        sh = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                               f"{s}_shoulder_pitch_link")
        el = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                               f"{s}_elbow_link")
        wr = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                               f"{s}_wrist_roll_link")
        if min(sh, el, wr) < 0:
            continue
        a, b, c = (np.array(data.xpos[i]) for i in (sh, el, wr))
        print(f"  {s}: shoulder {a}  elbow {b}  wrist {c}")
        print(f"      upper arm {np.linalg.norm(b-a):.4f} m, "
              f"forearm {np.linalg.norm(c-b):.4f} m, "
              f"total reach {np.linalg.norm(b-a)+np.linalg.norm(c-b):.4f} m")

    print()
    print("=" * 74)
    print("EXISTING TOUCH SENSORS")
    print("=" * 74)
    n_touch = 0
    for s in range(model.nsensor):
        if model.sensor_type[s] != mujoco.mjtSensor.mjSENS_TOUCH:
            continue
        n_touch += 1
        print(f"  {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, s)}")
    print(f"  {n_touch} touch sensors (we need 6)")

    print()
    print("=" * 74)
    print("POLICY")
    print("=" * 74)
    pol = os.path.expanduser(args.policy)
    if not os.path.isfile(pol):
        print(f"  {pol} NOT FOUND")
    else:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(pol)
            for i in sess.get_inputs():
                print(f"  input  {i.name:12s} {i.shape}  {i.type}")
            for o in sess.get_outputs():
                print(f"  output {o.name:12s} {o.shape}  {o.type}")
        except ImportError:
            print("  onnxruntime not installed:  pip install onnxruntime")
        except Exception as ex:
            print(f"  could not open the policy: {ex}")


if __name__ == "__main__":
    main()
