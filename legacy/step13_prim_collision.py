"""Step 13 - replace the G1's mesh collision geometry with primitives.

Taken from the collision model in ResMimic (Amazon FAR),
assets/g1/g1_custom_collision_29dof.urdf, which swaps the torso's STL for a
single cylinder:

    <collision><origin xyz="0 0 0.1"/>
      <geometry><cylinder length="0.35" radius="0.08"/></geometry>
    </collision>
    <!-- <mesh filename="torso_link_rev_1_0.STL"/>  commented out -->

and the two shoulder joints for small cylinders. Their arm links stay as
meshes; this script extends the same idea to the upper arm and the forearm,
because for this project that is where it pays.

Why bother:

  Contacts become analytic. A convex mesh pair is resolved by a general
  hull algorithm that returns a varying number of contact points in
  varying places. Two primitives are resolved in closed form. The chest
  reported 378 N for about 2 mm of overlap, and the per-frame contact
  count wandered between 3 and 8; that is the mesh solver, not the hug.

  The sensor zones stop lying. The bubble pads are capsules laid along the
  bone. If the ARM is also a capsule along the same bone, the collision
  surface and the sensor zone are the same shape about the same axis, and
  "the pad is on the arm" is exact rather than approximate.

  The search gets faster, which matters when it evaluates 47000 poses.

The arm capsules are not hand-written numbers. They run from the link's own
origin to the next joint, which model.body_pos gives directly, so they
follow whatever the model says the segment is.

    python step13_prim_collision.py
    python step13_prim_collision.py --arm-radius 0.030 --keep-hands

Then point step9 at the result:

    python step9_place_sites.py --source g1_29dof_prim.xml
    python carry_box.py
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

DEFAULT_DIR = "~/ws_g1/third_party/unitree_mujoco/unitree_robots/g1"

# ResMimic's own numbers, converted from URDF to MJCF conventions.
# MJCF cylinders take size="radius halflength" with the axis along local z,
# so a URDF length of 0.35 becomes a halflength of 0.175.
TORSO = dict(type="cylinder", size="0.08 0.175", pos="0 0 0.1")

# rpy="0 1.5708 0" is a +90 deg rotation about y, which lays the cylinder
# along x. As a quaternion (w x y z) that is (cos45, 0, sin45, 0).
_Q_Y90 = f"{math.cos(math.pi/4):.5f} 0 {math.sin(math.pi/4):.5f} 0"
SHOULDER = {
    "left_shoulder_pitch_link":
        dict(type="cylinder", size="0.03 0.025", pos="0 0.04 -0.01",
             quat=_Q_Y90),
    "right_shoulder_pitch_link":
        dict(type="cylinder", size="0.03 0.025", pos="0 -0.04 -0.01",
             quat=_Q_Y90),
    "left_shoulder_roll_link":
        dict(type="cylinder", size="0.03 0.015", pos="-0.004 0.006 -0.053"),
    "right_shoulder_roll_link":
        dict(type="cylinder", size="0.03 0.015", pos="-0.004 -0.006 -0.053"),
}

# Our extension: one capsule per arm segment, from this link's origin to the
# next joint. (host link, link that ends the segment, radius scale)
ARM_SEGMENTS = [
    ("left_shoulder_yaw_link",  "left_elbow_link",       1.00),
    ("left_elbow_link",         "left_wrist_roll_link",  0.95),
    ("right_shoulder_yaw_link", "right_elbow_link",      1.00),
    ("right_elbow_link",        "right_wrist_roll_link", 0.95),
]
HAND_LINKS = ["left_wrist_roll_link", "left_wrist_pitch_link",
              "left_wrist_yaw_link", "left_rubber_hand",
              "right_wrist_roll_link", "right_wrist_pitch_link",
              "right_wrist_yaw_link", "right_rubber_hand"]
TORSO_LINK = "torso_link"


def find_body(root, name):
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    return None


def mute(body):
    """Turn every geom on this body into a visual-only geom.

    Rather than deleting elements -- which would also delete whatever the
    renderer draws, and would break any <default> class the file relies on
    -- each geom gets contype and conaffinity written explicitly. An
    attribute on the element always beats the class it inherits from, so
    this is the one edit that is guaranteed to take effect regardless of how
    the file is organised.
    """
    n = 0
    for g in body.findall("geom"):
        g.set("contype", "0")
        g.set("conaffinity", "0")
        n += 1
    return n


def add(body, name, spec, rgba="0.9 0.3 0.2 0.35", group="4"):
    a = {"name": name, "contype": "1", "conaffinity": "1",
         "group": group, "rgba": rgba, "density": "0"}
    a.update({k: v for k, v in spec.items()})
    ET.SubElement(body, "geom", a)


def collidable_box(model, bid):
    """(lo, hi) of the collidable geometry on a body, in its own frame."""
    lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
    used = 0
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != bid:
            continue
        if int(model.geom_contype[g]) == 0 and \
           int(model.geom_conaffinity[g]) == 0:
            continue
        r = float(model.geom_rbound[g])
        p = np.array(model.geom_pos[g], dtype=float)
        lo = np.minimum(lo, p - r)
        hi = np.maximum(hi, p + r)
        used += 1
    return ((lo, hi), used) if used else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--source", default="g1_29dof.xml")
    ap.add_argument("--out", default="g1_29dof_prim.xml")
    ap.add_argument("--arm-radius", type=float, default=0.040,
                    help="radius of the upper-arm capsule [m]; the forearm "
                         "capsule is slightly thinner")
    ap.add_argument("--extend", type=float, default=0.050,
                    help="how far each capsule runs past each end of its "
                         "segment [m]. This is NOT cosmetic. The segment "
                         "shoulder_yaw_link -> elbow_link is only 0.082 m "
                         "long, but the anatomical upper arm is 0.185 m: "
                         "the yaw link's origin already sits halfway down "
                         "it, and everything above that belongs to "
                         "shoulder_roll_link, which ResMimic gives a 30 mm "
                         "stub. Without the extension the upper arm has "
                         "almost no collision surface, the box cannot touch "
                         "it, and the upper-arm pads read 0.00 N.")
    ap.add_argument("--keep-hands", action="store_true",
                    help="leave the wrist and hand meshes collidable "
                         "instead of switching them off")
    ap.add_argument("--keep-torso", action="store_true",
                    help="leave the torso meshes alone")
    args = ap.parse_args()

    d = os.path.expanduser(args.dir)
    src = os.path.join(d, args.source)
    if not os.path.isfile(src):
        raise SystemExit(f"{src} not found")

    # Measure on the untouched model first, so the change can be reported.
    probe = os.path.join(d, "_prim_measure.xml")
    with open(probe, "w") as fh:
        fh.write(f'<mujoco model="m">\n  <include file="{args.source}"/>\n'
                 '  <worldbody><geom name="floor" size="0 0 .05" type="plane"/>'
                 "</worldbody>\n</mujoco>\n")
    model = mujoco.MjModel.from_xml_path(probe)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    os.remove(probe)

    def bid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)

    before_ncol = sum(1 for g in range(model.ngeom)
                      if model.geom_contype[g] or model.geom_conaffinity[g])
    before_mesh = sum(1 for g in range(model.ngeom)
                      if (model.geom_contype[g] or model.geom_conaffinity[g])
                      and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)

    print("=" * 72)
    print("BEFORE")
    print("=" * 72)
    print(f"  {before_ncol} collidable geoms, {before_mesh} of them meshes")
    tb = bid(TORSO_LINK)
    box, n = collidable_box(model, tb)
    if box is not None:
        print(f"  {TORSO_LINK}: {n} collidable geoms")

    # Measure the arm segments while the clean model is still loaded.
    segs = []
    for host, child, scale in ARM_SEGMENTS:
        hb, cb = bid(host), bid(child)
        if hb < 0 or cb < 0:
            print(f"  SKIP {host}: {host if hb < 0 else child} not in model")
            continue
        v = np.array(model.body_pos[cb], dtype=float)
        L = float(np.linalg.norm(v))
        if L < 1e-6:
            print(f"  SKIP {host}: zero-length segment")
            continue
        u = v / L
        a = -u * args.extend
        b = v + u * args.extend
        segs.append((host, child, a, b, L, args.arm_radius * scale))

    print()
    print("=" * 72)
    print("REPLACING")
    print("=" * 72)

    tree = ET.parse(src)
    root = tree.getroot()
    muted = 0

    if not args.keep_torso:
        body = find_body(root, TORSO_LINK)
        if body is None:
            print(f"  {TORSO_LINK} not found in the XML")
        else:
            k = mute(body)
            add(body, "col_torso", TORSO)
            muted += k
            print(f"  {TORSO_LINK:28s} {k} mesh geoms off -> cylinder "
                  f"r 0.080, length 0.350 at z 0.100   [ResMimic]")

    for name, spec in SHOULDER.items():
        body = find_body(root, name)
        if body is None:
            continue
        k = mute(body)
        add(body, f"col_{name}", spec)
        muted += k
        print(f"  {name:28s} {k} geoms off -> cylinder "
              f"r {spec['size'].split()[0]}   [ResMimic]")

    for host, child, a, b, L, r in segs:
        body = find_body(root, host)
        if body is None:
            continue
        k = mute(body)
        add(body, f"col_{host}", dict(
            type="capsule", size=f"{r:.4f}",
            fromto=" ".join(f"{v:.5f}" for v in list(a) + list(b))),
            rgba="0.2 0.6 0.95 0.35")
        muted += k
        print(f"  {host:28s} {k} geoms off -> capsule r {r:.3f}, "
              f"L {L:.3f} to {child}")

    if not args.keep_hands:
        for name in HAND_LINKS:
            body = find_body(root, name)
            if body is None:
                continue
            k = mute(body)
            muted += k
            if k:
                print(f"  {name:28s} {k} geoms off   (no replacement: the "
                      f"hand is not a contact surface here)")

    # The extended upper-arm capsule reaches back far enough to overlap the
    # shoulder-pitch cylinder. MuJoCo excludes parent-child pairs on its
    # own, but shoulder_yaw's parent is shoulder_roll, so yaw and pitch are
    # grandchild and grandparent and are NOT excluded: they showed up as a
    # permanent 11 mm self-collision at the rest pose. They are one shoulder
    # complex and should never generate contact, so say so explicitly.
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    have = {(e.get("body1"), e.get("body2"))
            for e in contact.findall("exclude")}
    for side in ("left", "right"):
        pair = (f"{side}_shoulder_pitch_link", f"{side}_shoulder_yaw_link")
        if pair in have or pair[::-1] in have:
            continue
        if find_body(root, pair[0]) is None or find_body(root, pair[1]) is None:
            continue
        ET.SubElement(contact, "exclude",
                      {"name": f"excl_{side}_shoulder",
                       "body1": pair[0], "body2": pair[1]})
        print(f"  exclude contact {pair[0]} <-> {pair[1]}")

    out = os.path.join(d, args.out)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=False)
    print(f"\n  wrote {out}   ({muted} geoms made visual-only)")

    # ---- verify ---------------------------------------------------------
    probe2 = os.path.join(d, "_prim_verify.xml")
    with open(probe2, "w") as fh:
        fh.write(f'<mujoco model="v">\n  <include file="{args.out}"/>\n'
                 '  <worldbody><geom name="floor" size="0 0 .05" type="plane"/>'
                 "</worldbody>\n</mujoco>\n")
    m2 = mujoco.MjModel.from_xml_path(probe2)
    d2 = mujoco.MjData(m2)
    mujoco.mj_resetData(m2, d2)
    mujoco.mj_forward(m2, d2)

    after_ncol = sum(1 for g in range(m2.ngeom)
                     if m2.geom_contype[g] or m2.geom_conaffinity[g])
    after_mesh = sum(1 for g in range(m2.ngeom)
                     if (m2.geom_contype[g] or m2.geom_conaffinity[g])
                     and m2.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
    print()
    print("=" * 72)
    print("AFTER")
    print("=" * 72)
    print(f"  {after_ncol} collidable geoms, {after_mesh} of them meshes "
          f"(was {before_ncol} / {before_mesh})")

    tb2 = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, TORSO_LINK)
    names = {"chest surface": None}
    for g in range(m2.ngeom):
        if int(m2.geom_bodyid[g]) == tb2 and (m2.geom_contype[g]
                                              or m2.geom_conaffinity[g]):
            gn = mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}"
            t = int(m2.geom_type[g])
            print(f"  {TORSO_LINK}: {gn}, type {t}, size "
                  f"{np.array(m2.geom_size[g])}")

    print("\n  arm capsules in world at the rest pose")
    print(f"  {'link':28s} {'radius':>7s}  {'start':>24s} {'end':>24s}")
    for host, child, a, b, L, r in segs:
        hb = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_BODY, host)
        R = np.array(d2.xmat[hb]).reshape(3, 3)
        o = np.array(d2.xpos[hb])
        wa, wb = o + R @ a, o + R @ b
        f = lambda v: f"({v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f})"
        print(f"  {host:28s} {r:7.3f}  {f(wa):>24s} {f(wb):>24s}")

    # Does the robot still stand without exploding? A primitive swap can
    # introduce self-collisions the meshes did not have.
    bad = 0
    for c in range(d2.ncon):
        g1 = int(d2.contact[c].geom1); g2 = int(d2.contact[c].geom2)
        b1 = mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_BODY,
                               int(m2.geom_bodyid[g1]))
        b2 = mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_BODY,
                               int(m2.geom_bodyid[g2]))
        if "world" in (str(b1), str(b2)):
            continue
        if float(d2.contact[c].dist) < -0.002:
            bad += 1
            print(f"  SELF-COLLISION {b1} <-> {b2}  depth "
                  f"{d2.contact[c].dist:+.4f}")
    if bad == 0:
        print("\n  no self-collisions at the rest pose")
    else:
        print(f"\n  {bad} self-collisions. Thin the capsules with "
              f"--arm-radius 0.030.")
    os.remove(probe2)

    print("\nnext:")
    print(f"  python step9_place_sites.py --source {args.out}")
    print("  python carry_box.py")


if __name__ == "__main__":
    main()
