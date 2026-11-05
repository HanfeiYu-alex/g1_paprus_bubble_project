"""Step 9 (v2) - six bubble pads, matching the real sensor hardware.

The eight-patch layout was a software convenience, not the hardware. On the
real robot the chest is ONE large bubble, the back is ONE large bubble, and
each arm segment carries one pad. Six channels total:

    chest             one large pad over the whole front of the torso
    back              one large pad over the whole back of the torso
    left_upper_arm    one pad wrapping the upper arm  (shoulder -> elbow)
    left_lower_arm    one pad wrapping the forearm    (elbow -> wrist)
    right_upper_arm
    right_lower_arm

Two things changed besides the count.

  The torso pads are no longer hand-placed numbers. The previous version put
  the chest sites at local x 0.12 and 0.13 with a 0.075 radius, and they read
  0.00 N in every single run, because a site is only a marker: the surface
  that generates contact is the torso's collision geometry, and that sits at
  x 0.070. The sites were 5 cm in front of the surface they were meant to be
  mounted on. This version MEASURES the torso's collision geometry -- every
  collidable geom on torso_link, meshes included, reduced to a box in the
  torso's own frame -- and lays each pad straddling the real surface, so a
  contact anywhere on the chest lands inside the pad.

  The arm pads are capsules along the bone, not spheres beside it. A bubble
  wraps the limb, so the sensor zone should too. The capsule axis comes from
  model.body_pos of the child link, which IS the segment vector, so the pad
  stays on the arm whatever pose the arm is in.

    python step9_place_sites.py
    python step9_place_sites.py --pad-thick 0.045 --arm-radius 0.055

Writes g1_29dof_bubble.xml, then run carry_box.py.
"""

import argparse
import math
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

DEFAULT_DIR = "~/ws_g1/third_party/unitree_mujoco/unitree_robots/g1"

# The channel order. Everything downstream indexes by this, so it is fixed.
PATCHES = ["chest", "back",
           "left_upper_arm", "left_lower_arm",
           "right_upper_arm", "right_lower_arm"]

# name, host link, link whose origin ends the segment
ARM_SEGMENTS = [
    ("left_upper_arm",  "left_shoulder_yaw_link",  "left_elbow_link"),
    ("left_lower_arm",  "left_elbow_link",         "left_wrist_roll_link"),
    ("right_upper_arm", "right_shoulder_yaw_link", "right_elbow_link"),
    ("right_lower_arm", "right_elbow_link",        "right_wrist_roll_link"),
]
TORSO_LINK = "torso_link"


def quat_z_to(u):
    """Quaternion (w x y z) rotating local +z onto u."""
    u = np.asarray(u, dtype=float)
    u = u / np.linalg.norm(u)
    d = float(u[2])
    if d > 1.0 - 1e-9:
        return np.array([1., 0., 0., 0.])
    if d < -1.0 + 1e-9:
        return np.array([0., 1., 0., 0.])
    ax = np.cross([0., 0., 1.], u)
    ax /= np.linalg.norm(ax)
    ang = math.acos(max(-1., min(1., d)))
    s = math.sin(ang / 2.)
    return np.array([math.cos(ang / 2.), ax[0] * s, ax[1] * s, ax[2] * s])


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def geom_local_corners(model, g):
    """Corner points of geom g, expressed in its BODY's frame.

    Covers the geom types the G1 actually uses. Meshes are reduced through
    their stored vertices, which is exact; primitives through their size.
    """
    t = int(model.geom_type[g])
    s = np.array(model.geom_size[g], dtype=float)
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(model.geom_dataid[g])
        if mid < 0:
            return None
        a = int(model.mesh_vertadr[mid]); n = int(model.mesh_vertnum[mid])
        if n <= 0:
            return None
        # model.mesh_vert has shape (nmeshvert, 3) in the Python bindings,
        # NOT a flat array. Slicing it as 3*a .. 3*(a+n) indexes ROWS, runs
        # off the end, and hands back an empty array -- which is how every
        # torso mesh silently fell through to its bounding sphere and
        # inflated the chest surface from x 0.070 to x 0.216.
        v = np.array(model.mesh_vert[a:a + n], dtype=float).reshape(-1, 3)
        if len(v) != n:
            return None
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
        h = s[:3]
        v = np.array([[sx * h[0], sy * h[1], sz * h[2]]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
        h = np.array([s[0], s[0], s[0]])
        v = np.array([[sx * h[0], sy * h[1], sz * h[2]]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        r, hl = s[0], s[1]
        hz = hl + (r if t == mujoco.mjtGeom.mjGEOM_CAPSULE else 0.)
        v = np.array([[sx * r, sy * r, sz * hz]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    elif t == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        v = np.array([[sx * s[0], sy * s[1], sz * s[2]]
                      for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    else:
        return None
    R = quat_to_mat(np.array(model.geom_quat[g], dtype=float))
    return (R @ v.T).T + np.array(model.geom_pos[g], dtype=float)


def arm_radius(model, bid, axis, default=0.035):
    """Measured radius of a limb's collision geometry about its own bone.

    The bubble on the real arm is a hollow sleeve 3-5 mm thick wrapped
    round the limb, so the sensor zone is the limb's own radius plus that
    thickness -- not a number picked to look right. This measures the limb
    by taking the perpendicular distance of every collision vertex from the
    segment axis and using the 90th percentile, which ignores the few
    vertices that stick out at the joint housings.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / max(np.linalg.norm(axis), 1e-9)
    radii = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != bid:
            continue
        if int(model.geom_contype[g]) == 0 and \
           int(model.geom_conaffinity[g]) == 0:
            continue
        c = geom_local_corners(model, g)
        if c is None or len(c) == 0:
            continue
        perp = c - np.outer(c @ axis, axis)
        radii.append(np.linalg.norm(perp, axis=1))
    if not radii:
        return default, False
    return float(np.percentile(np.concatenate(radii), 90)), True


def torso_box(model, bid, verbose=True):
    """The torso's collision envelope as (lo, hi) in the torso's own frame.

    Geoms whose exact extent cannot be read -- an empty mesh, a type this
    code does not decompose -- fall back to their bounding sphere, which is
    always present as model.geom_rbound. The first version of this just
    crashed on an empty mesh, which is worse than an approximate answer.
    """
    lo = np.full(3, np.inf); hi = np.full(3, -np.inf)
    exact, approx = 0, 0
    rows = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != bid:
            continue
        if int(model.geom_contype[g]) == 0 and \
           int(model.geom_conaffinity[g]) == 0:
            continue  # visual only; it never generates a contact
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}"
        c = geom_local_corners(model, g)
        how = "exact"
        if c is None or len(c) == 0:
            r = float(model.geom_rbound[g])
            if r <= 0.:
                rows.append((name, int(model.geom_type[g]), "skipped"))
                continue
            p = np.array(model.geom_pos[g], dtype=float)
            c = np.array([p - r, p + r])
            how = "rbound"
            approx += 1
        else:
            exact += 1
        lo = np.minimum(lo, c.min(axis=0))
        hi = np.maximum(hi, c.max(axis=0))
        rows.append((name, int(model.geom_type[g]), how))
    if verbose:
        for name, t, how in rows:
            print(f"    geom {name:28s} type {t:2d}  {how}")
    if exact + approx == 0:
        return None, 0
    return (lo, hi), exact + approx


def find_body(root, name):
    for b in root.iter("body"):
        if b.get("name") == name:
            return b
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DEFAULT_DIR)
    ap.add_argument("--source", default="g1_29dof.xml")
    ap.add_argument("--out", default="g1_29dof_bubble.xml")
    ap.add_argument("--pad-thick", type=float, default=0.020,
                    help="half-thickness of the torso pads, straddling the "
                         "measured collision surface [m]. The real chest "
                         "bubble is one 20 mm sheet over the whole front.")
    ap.add_argument("--pad-shrink", type=float, default=0.85,
                    help="fraction of the torso's width and height the pads "
                         "cover; below 1 so the pads do not pick up the arms "
                         "resting against the body")
    ap.add_argument("--arm-radius", type=float, default=None,
                    help="force the arm pad radius [m]. Left unset, it is "
                         "MEASURED from the limb and the sleeve added.")
    ap.add_argument("--keep-hands", action="store_true",
                    help="leave the wrist and hand meshes in place. By "
                         "default they are switched off, for collision and "
                         "for display: OpenHomie's g1.xml carries a "
                         "fully articulated hand with splayed fingers that "
                         "the robot in the lab does not have, and it both "
                         "renders as a distraction and generates contacts "
                         "nothing on the real arm could generate.")
    ap.add_argument("--arm-sleeve", type=float, default=0.004,
                    help="wall thickness of the arm bubble sleeve [m]; the "
                         "hardware is 3-5 mm, so 4 mm is the middle")
    ap.add_argument("--arm-extend", type=float, default=0.030,
                    help="how far the arm pad runs past each end of its "
                         "segment [m]")
    ap.add_argument("--upper-extend", type=float, default=0.100,
                    help="extra length added to the UPPER-arm pads at the "
                         "shoulder end [m]. The segment this pad is hosted "
                         "on, shoulder_yaw_link -> elbow_link, is only "
                         "0.082 m, but the anatomical upper arm is 0.193 m: "
                         "the top half belongs to shoulder_roll_link and "
                         "was not covered at all. The real bubble is a "
                         "sleeve round the whole upper arm, so the pad "
                         "should be too -- otherwise contact that creeps "
                         "toward the shoulder leaves the sensor zone and "
                         "reads 0.00 N while the arm is plainly touching.")
    args = ap.parse_args()

    d = os.path.expanduser(args.dir)
    src = os.path.join(d, args.source)
    if not os.path.isfile(src):
        raise SystemExit(f"{src} not found")

    probe = os.path.join(d, "_measure.xml")
    with open(probe, "w") as fh:
        # No floor geom here. Measuring needs only mj_forward, and
        # OpenHomie's g1.xml is a full scene that already defines one --
        # adding a second is a duplicate-name error, not a warning.
        fh.write(f'<mujoco model="m">\n  <include file="{args.source}"/>\n'
                 "</mujoco>\n")
    model = mujoco.MjModel.from_xml_path(probe)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    os.remove(probe)

    def bid(n):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)

    # ---- the torso pads, measured ---------------------------------------
    tb = bid(TORSO_LINK)
    if tb < 0:
        raise SystemExit(f"no body named {TORSO_LINK}")
    print("=" * 70)
    print("TORSO COLLISION ENVELOPE  (measured, not assumed)")
    print("=" * 70)
    box, ngeom = torso_box(model, tb)
    if box is None:
        raise SystemExit(f"{TORSO_LINK} has no collidable geometry")
    lo, hi = box
    Rt = np.array(data.xmat[tb]).reshape(3, 3)
    org = np.array(data.xpos[tb])
    print(f"  {ngeom} collidable geoms on {TORSO_LINK}")
    print(f"  in the torso's own frame:")
    print(f"    x {lo[0]:+.3f} .. {hi[0]:+.3f}   (front surface at "
          f"x {hi[0]:+.3f})")
    print(f"    y {lo[1]:+.3f} .. {hi[1]:+.3f}")
    print(f"    z {lo[2]:+.3f} .. {hi[2]:+.3f}")
    front_world = org + Rt @ np.array([hi[0], 0., 0.])
    back_world = org + Rt @ np.array([lo[0], 0., 0.])
    print(f"  in world at the rest pose:")
    print(f"    chest surface  x {front_world[0]:+.3f}")
    print(f"    back surface   x {back_world[0]:+.3f}")
    print("  The old layout put the chest sites at x 0.116 and 0.126, which")
    print(f"  is {0.116 - front_world[0]:+.3f} m in front of the surface. That")
    print("  is why the chest channel read 0.00 N in every run so far.")

    cy = 0.5 * (lo[1] + hi[1])
    cz = 0.5 * (lo[2] + hi[2])
    hy = 0.5 * (hi[1] - lo[1]) * args.pad_shrink
    hz = 0.5 * (hi[2] - lo[2]) * args.pad_shrink

    placements = [
        # name, host link, site type, pos, size, quat
        ("chest", TORSO_LINK, "box",
         (hi[0], cy, cz), (args.pad_thick, hy, hz), None),
        ("back", TORSO_LINK, "box",
         (lo[0], cy, cz), (args.pad_thick, hy, hz), None),
    ]

    # ---- the arm pads, from the segment vectors --------------------------
    print()
    print("=" * 70)
    print("ARM PADS  (capsule along the bone, from model.body_pos)")
    print("=" * 70)
    print(f"  {'pad':17s} {'segment':>8s} {'capsule':>9s}  axis in the host "
          f"link's frame")
    for name, host, child in ARM_SEGMENTS:
        hb, cb = bid(host), bid(child)
        if hb < 0 or cb < 0:
            print(f"  {name:17s} SKIP: {host if hb < 0 else child} missing")
            continue
        seg = np.array(model.body_pos[cb], dtype=float)
        L = float(np.linalg.norm(seg))
        if L < 1e-6:
            print(f"  {name:17s} SKIP: zero-length segment")
            continue
        u = seg / L
        # The upper-arm pads grow toward the shoulder, not symmetrically.
        prox = args.upper_extend if name.endswith("upper_arm") else 0.0
        half = 0.5 * L + args.arm_extend + prox / 2.
        pos = seg * 0.5 - u * (prox / 2.)
        q = quat_z_to(u)
        if args.arm_radius is not None:
            r, how = args.arm_radius, "forced"
        else:
            limb, measured = arm_radius(model, hb, seg)
            r = limb + args.arm_sleeve
            how = (f"limb {limb:.3f} + sleeve {args.arm_sleeve:.3f}"
                   if measured else "default, limb not measurable")
        placements.append((name, host, "capsule", tuple(pos),
                           (r, half), tuple(q)))
        print(f"  {name:17s} {L:7.3f}m  r{r:.4f} "
              f"L{2*half:.3f}  ({u[0]:+.2f},{u[1]:+.2f},{u[2]:+.2f})  {how}")

    # ---- write the robot file -------------------------------------------
    tree = ET.parse(src)
    root = tree.getroot()
    by_name = {p[0]: p for p in placements}

    added = []
    for name in PATCHES:
        if name not in by_name:
            continue
        _, host, stype, pos, size, quat = by_name[name]
        body = find_body(root, host)
        if body is None:
            print(f"  SKIP {name}: no body {host} in the XML")
            continue
        attrs = {
            "name": f"bubble_{name}",
            "pos": " ".join(f"{v:.4f}" for v in pos),
            "size": " ".join(f"{v:.4f}" for v in size),
            "type": stype,
            "rgba": ("0.95 0.45 0.15 0.35" if name in ("chest", "back")
                     else "0.20 0.90 0.30 0.35"),
            "group": "3",
        }
        if quat is not None:
            attrs["quat"] = " ".join(f"{v:.5f}" for v in quat)
        ET.SubElement(body, "site", attrs)
        added.append(name)

    if not args.keep_hands:
        # Scan by NAME PATTERN over the whole tree, and say exactly what
        # was touched. The previous version edited a hand-written list of
        # six body names, reported "12 geoms", and the hands were still
        # there in the render -- the model keeps them under names that list
        # did not contain. Guessing names is how that happened; this does
        # not guess, and it prints its work so the claim can be checked.
        pat = ("wrist_pitch", "wrist_yaw", "rubber_hand", "hand",
               "finger", "thumb", "index", "middle", "palm")
        touched = []
        for body in root.iter("body"):
            bn = (body.get("name") or "").lower()
            if not any(k in bn for k in pat):
                continue
            for g in body.findall("geom"):
                g.set("contype", "0")
                g.set("conaffinity", "0")
                g.set("rgba", "0 0 0 0")
                g.set("group", "4")       # outside the viewer's default set
                g.attrib.pop("material", None)   # material would override rgba
                touched.append((body.get("name"),
                                g.get("name") or g.get("mesh") or "?"))
        print(f"\n  hands: {len(touched)} geoms hidden and made "
              f"non-collidable  (--keep-hands to leave them)")
        for bn, gn in touched:
            print(f"    {bn:28s} {gn}")
        if not touched:
            print("    none matched. Body names in this model:")
            for body in root.iter("body"):
                print(f"      {body.get('name')}")

    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    # Drop any touch sensors left behind by the old eight-patch layout.
    for s in list(sensor.findall("touch")):
        sensor.remove(s)
    for name in added:
        ET.SubElement(sensor, "touch",
                      {"name": f"touch_{name}", "site": f"bubble_{name}"})

    out = os.path.join(d, args.out)
    ET.indent(tree, space="  ")
    tree.write(out, encoding="utf-8", xml_declaration=False)
    print(f"\nwrote {out} with {len(added)} pads")

    # ---- verify on the compiled model -----------------------------------
    probe2 = os.path.join(d, "_verify.xml")
    with open(probe2, "w") as fh:
        fh.write(
            f'<mujoco model="v">\n  <include file="{args.out}"/>\n'
            '  <worldbody>\n'
            '    <body name="probebox" pos="0.6 0 0.95">\n'
            '      <freejoint name="probebox_free"/>\n'
            '      <geom name="probebox_geom" type="box" size="0.07 0.17 0.10"'
            ' mass="2"/>\n'
            '    </body>\n'
            '  </worldbody>\n</mujoco>\n')
    m2 = mujoco.MjModel.from_xml_path(probe2)
    d2 = mujoco.MjData(m2)
    mujoco.mj_forward(m2, d2)
    print("\nwhere the pads sit at the rest pose (world)")
    print(f"  {'pad':17s} {'x':>8s} {'y':>8s} {'z':>8s}  {'type':>8s}  zone")
    names = []
    for name in PATCHES:
        sid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_SITE, f"bubble_{name}")
        if sid < 0:
            continue
        names.append(name)
        p = d2.site_xpos[sid]
        st = int(m2.site_type[sid])
        tn = {mujoco.mjtGeom.mjGEOM_BOX: "box",
              mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
              mujoco.mjtGeom.mjGEOM_SPHERE: "sphere"}.get(st, str(st))
        sz = m2.site_size[sid]
        print(f"  {name:17s} {p[0]:+8.3f} {p[1]:+8.3f} {p[2]:+8.3f}  "
              f"{tn:>8s}  {sz[0]:.3f} {sz[1]:.3f} {sz[2]:.3f}")
    n_touch = sum(1 for s in range(m2.nsensor)
                  if m2.sensor_type[s] == mujoco.mjtSensor.mjSENS_TOUCH)
    print(f"\n  {n_touch} touch sensors compiled, expected {len(PATCHES)}")
    if n_touch != len(PATCHES):
        print("  MISMATCH -- the sensor block did not compile as expected.")

    # ---- self-check: does the chest pad actually feel a box? -------------
    # Measuring the envelope and placing a pad on it is not proof. Push a
    # box at the chest and confirm the contact point lands inside the pad.
    print()
    print("=" * 70)
    print("SELF-CHECK  push a box at the chest and watch the pad")
    print("=" * 70)
    bgid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_GEOM, "probebox_geom")
    bjid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_JOINT, "probebox_free")
    bq = int(m2.jnt_qposadr[bjid])
    chest_sid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_SITE, "bubble_chest")
    chest_sz = np.array(m2.site_size[chest_sid], dtype=float)
    q0 = m2.qpos0.copy()
    first_any, first_chest = None, None
    for back in np.arange(0.30, -0.02, -0.004):
        d2.qpos[:] = q0
        d2.qvel[:] = 0.
        d2.qpos[bq:bq + 7] = (float(back) + 0.07, 0., 0.95, 1., 0., 0., 0.)
        mujoco.mj_forward(m2, d2)
        inside = False
        touched = False
        for c in range(d2.ncon):
            con = d2.contact[c]
            if int(con.geom1) != bgid and int(con.geom2) != bgid:
                continue
            touched = True
            R = np.array(d2.site_xmat[chest_sid]).reshape(3, 3)
            q = R.T @ (np.array(con.pos) - np.array(d2.site_xpos[chest_sid]))
            if np.all(np.abs(q) <= chest_sz[:3]):
                inside = True
        if touched and first_any is None:
            first_any = float(back)
        if inside:
            first_chest = float(back)
            break
    if first_any is None:
        print("  the box never touched the robot at all, down to x 0.00.")
    else:
        print(f"  first contact with the robot   back face x {first_any:+.3f}")
    if first_chest is None:
        print("  the chest pad NEVER registered a contact.")
        print("  The pad is not on the surface. Try a thicker pad:")
        print("    python step9_place_sites.py --pad-thick 0.06")
    else:
        print(f"  first contact IN the chest pad  back face x "
              f"{first_chest:+.3f}")
        print(f"  The chest pad works. A box parked with its back face at or")
        print(f"  just behind x {first_chest:+.3f} will load that channel.")
        if first_any is not None and first_chest < first_any - 0.02:
            print(f"  NOTE: something else on the robot (an arm) meets the box")
            print(f"  {first_any - first_chest:+.3f} m earlier than the chest.")
    os.remove(probe2)

    print("\nnext:")
    print("  python carry_box.py")


if __name__ == "__main__":
    main()
