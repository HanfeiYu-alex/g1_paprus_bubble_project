"""Step 5b - why did the arms pass through the box?

The arms visually penetrated the box and every touch channel stayed at
0.00 N. Those two facts together point at one cause: no collision is being
generated between the arm links and the box. A touch sensor reports the
normal forces of contacts inside its site, so with no contact there is
nothing to report, and with no contact the solver has no reason to stop the
arm either.

This script checks, in order:

  1. does each wrap link own any geom at all
  2. are those geoms collidable, or visual-only
  3. does the collision mask let them meet the box
  4. is the pair excluded explicitly
  5. with the arms driven into the box, how many contacts actually appear

MuJoCo collides two geoms only when

    (contype1 & conaffinity2) or (contype2 & conaffinity1)

so a geom with contype = 0 and conaffinity = 0 is decoration: it is drawn
and never touched.

    python step5b_diagnose.py
"""

import argparse
import os

import mujoco
import numpy as np

DEFAULT_SCENE = (
    "~/ws_g1/third_party/unitree_mujoco/unitree_robots/g1/scene_bubble.xml"
)

WRAP_LINKS = [
    "torso_link",
    "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link",
    "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link",
]

GEOM_TYPES = {
    0: "plane", 1: "hfield", 2: "sphere", 3: "capsule", 4: "ellipsoid",
    5: "cylinder", 6: "box", 7: "mesh", 8: "sdf",
}


def can_collide(model, g1, g2):
    return bool(
        (model.geom_contype[g1] & model.geom_conaffinity[g2])
        or (model.geom_contype[g2] & model.geom_conaffinity[g1])
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    args = ap.parse_args()

    path = os.path.expanduser(args.scene)
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    box_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    box_geoms = list(range(model.body_geomadr[box_bid],
                           model.body_geomadr[box_bid] + model.body_geomnum[box_bid]))

    print("=" * 74)
    print("1-2. GEOMS ON EACH WRAP LINK")
    print("=" * 74)
    print(f"{'link':28s} {'geom':22s} {'type':8s} {'ctype':>5s} "
          f"{'caff':>5s} {'grp':>4s}  collidable")
    collidable_arm_geoms = []
    for link in WRAP_LINKS:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
        if bid < 0:
            print(f"{link:28s} (no such body)")
            continue
        start = model.body_geomadr[bid]
        n = model.body_geomnum[bid]
        if n == 0:
            print(f"{link:28s} {'-- no geoms at all --':22s}")
            continue
        for g in range(start, start + n):
            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}"
            ct = int(model.geom_contype[g])
            ca = int(model.geom_conaffinity[g])
            grp = int(model.geom_group[g])
            live = ct != 0 or ca != 0
            if live:
                collidable_arm_geoms.append((link, g, gname))
            print(f"{link:28s} {gname[:22]:22s} "
                  f"{GEOM_TYPES.get(int(model.geom_type[g]), '?'):8s} "
                  f"{ct:5d} {ca:5d} {grp:4d}  {'yes' if live else 'NO (visual only)'}")

    print()
    print("=" * 74)
    print("3. CAN THOSE GEOMS MEET THE BOX?")
    print("=" * 74)
    for bg in box_geoms:
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, bg) or f"g{bg}"
        print(f"  box geom {bname}  contype {int(model.geom_contype[bg])} "
              f"conaffinity {int(model.geom_conaffinity[bg])}")
    if not collidable_arm_geoms:
        print()
        print("  No collidable geom on any wrap link, so the answer is no.")
        print("  The arms are visual meshes. They will pass through anything.")
    else:
        ok = 0
        for link, g, gname in collidable_arm_geoms:
            for bg in box_geoms:
                if can_collide(model, g, bg):
                    ok += 1
                    break
        print()
        print(f"  {ok} of {len(collidable_arm_geoms)} collidable link geoms "
              f"pass the mask test against the box")

    print()
    print("=" * 74)
    print("4. EXPLICIT EXCLUSIONS")
    print("=" * 74)
    print(f"  {model.nexclude} <contact><exclude> pairs in the model")

    print()
    print("=" * 74)
    print("5. DRIVE THE ARMS INTO THE BOX AND COUNT CONTACTS")
    print("=" * 74)

    # Put the arms where the hug pose would put them and let it settle.
    qadr, names = [], []
    for a in range(model.nu):
        jnt = model.actuator_trnid[a, 0]
        names.append(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jnt))
        qadr.append(model.jnt_qposadr[jnt])
    qadr = np.array(qadr)

    target = np.zeros(model.nu)
    for side, mirror in (("left", 1.0), ("right", -1.0)):
        for joint, value in (("shoulder_pitch", -1.25),
                             ("shoulder_roll", mirror * 0.10),
                             ("elbow", 1.25)):
            jn = f"{side}_{joint}_joint"
            if jn in names:
                target[names.index(jn)] = value

    base = data.qpos[:7].copy()
    kp, kd = 90.0, 3.5
    for _ in range(3000):
        tau = kp * (target - data.qpos[qadr]) - kd * data.qvel[
            [model.jnt_dofadr[model.actuator_trnid[a, 0]] for a in range(model.nu)]
        ]
        data.ctrl[:] = np.clip(tau, -model.actuator_ctrlrange[:, 1],
                               model.actuator_ctrlrange[:, 1])
        mujoco.mj_step(model, data)
        data.qpos[:7] = base
        data.qvel[:6] = 0.0

    print(f"  after settling: {data.ncon} contacts in the whole scene")
    arm_box = 0
    seen = {}
    for c in range(data.ncon):
        g1 = int(data.contact[c].geom1)
        g2 = int(data.contact[c].geom2)
        b1 = int(model.geom_bodyid[g1])
        b2 = int(model.geom_bodyid[g2])
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b1)
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b2)
        key = tuple(sorted((n1, n2)))
        seen[key] = seen.get(key, 0) + 1
        if "box" in (n1, n2):
            arm_box += 1
    for (a, b), n in sorted(seen.items()):
        mark = "  <-- box" if "box" in (a, b) else ""
        print(f"    {a} <-> {b}   x{n}{mark}")
    print()
    print(f"  contacts involving the box: {arm_box}")

    # Where did the arms end up, and where is the box?
    print()
    for link in ("left_elbow_link", "right_elbow_link"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link)
        p = data.xpos[bid]
        print(f"  {link:24s} ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})")
    p = data.xpos[box_bid]
    hs = model.geom_size[box_geoms[0]]
    print(f"  {'box centre':24s} ({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f})"
          f"   half-size ({hs[0]:.2f}, {hs[1]:.2f}, {hs[2]:.2f})")
    print(f"  {'box spans x':24s} {p[0]-hs[0]:+.3f} .. {p[0]+hs[0]:+.3f}")
    print(f"  {'box spans y':24s} {p[1]-hs[1]:+.3f} .. {p[1]+hs[1]:+.3f}")
    print(f"  {'box spans z':24s} {p[2]-hs[2]:+.3f} .. {p[2]+hs[2]:+.3f}")

    print()
    print("=" * 74)
    print("VERDICT")
    print("=" * 74)
    if not collidable_arm_geoms:
        print("  The wrap links carry no collidable geometry.")
        print("  Fix: add collision geoms to those links. Step 6 does that.")
    elif arm_box == 0:
        print("  The links CAN collide, but nothing reached the box.")
        print("  The arm pose is wrong, not the model. Compare the elbow")
        print("  positions above against the box spans and adjust the pose.")
    else:
        print(f"  {arm_box} contacts with the box exist. If the touch")
        print("  channels still read zero, the sites are not where the")
        print("  contacts are: enlarge --radius or move the site positions.")


if __name__ == "__main__":
    main()
