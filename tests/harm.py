"""Did a cleanup change the performance? Measurements before and after, and the limits they have to keep.

Used by ``real_take.py``. Everything is measured on the evaluated rig and on the character mesh
(world positions), not on what the add-on reports about itself.

The limits exist because a run once printed "slide 348 -> 9 mm, RESULT: ok" after it had pinned a
foot through a 35 cm step: fixing a number is not the same as leaving the performance alone.
"""

from __future__ import annotations

import bpy
import numpy as np

# The one-click defaults have to stay inside these on any take.
LIMITS = {
    "foot_cm": 5.5,          # a pin may hold a foot where it is, not carry it somewhere else (horizontal)
    "pelvis_cm": 1.0,        # how far the hips are lowered, 95th percentile
    "knee_turn": 10.0,       # degrees a bent knee may turn sideways about the hip-ankle line (knock-knees)
    "knee_accel": 1.5,       # knee acceleration: not above this many times the raw maximum ...
    "knee_accel_abs": 3000,  # ... when it also rises by this many deg/s^2
    "chains_cm": 0.5,        # IK and FK chain of a limb, where the take came with both in sync
    "mesh_cm": 0.5,          # where the character mesh rests, against the floor
}


def character_meshes(scene, rig):
    """Meshes deformed by ``rig``, or by a skeleton that is constrained to it."""
    out = []
    for ob in scene.objects:
        if ob.type != "MESH":
            continue
        for m in ob.modifiers:
            arm = m.object if m.type == "ARMATURE" else None
            if arm is not None and (arm == rig or any(getattr(c, "target", None) == rig
                                                     for pb in arm.pose.bones for c in pb.constraints)):
                out.append(ob)
                break
    return out


def moving_vertices(ob):
    """Mask of the vertices weighted to a deforming bone of the mesh's armature (None: all, or unknown).

    Loose vertices without a weight stay where they were modelled and say nothing about the feet
    (the OptiTrack actor's mesh has six at z = -1 cm, which read as "1 cm under the floor" on every frame).
    """
    arms = [m.object for m in ob.modifiers
            if m.type == "ARMATURE" and m.object is not None and m.object.type == "ARMATURE" and m.use_vertex_groups]
    if not arms or not ob.vertex_groups or not len(ob.data.vertices):
        return None
    bones = {b.name for a in arms for b in a.data.bones if b.use_deform}
    groups = {g.index for g in ob.vertex_groups if g.name in bones}
    mask = np.array([any(g.group in groups and g.weight > 0.0 for g in v.groups) for v in ob.data.vertices], dtype=bool)
    return None if (mask.all() or not mask.any()) else mask


def mesh_lowest(scene, meshes, frames) -> np.ndarray:
    """Lowest world z of the deformed meshes at each of ``frames`` (only vertices that follow a bone)."""
    out = np.full(len(frames), np.inf)
    cur = scene.frame_current
    masks = {ob.name: moving_vertices(ob) for ob in meshes}
    for i, f in enumerate(frames):
        scene.frame_set(int(f))
        dg = bpy.context.evaluated_depsgraph_get()
        for ob in meshes:
            ev = ob.evaluated_get(dg)
            me = ev.to_mesh()
            co = np.empty(len(me.vertices) * 3)
            me.vertices.foreach_get("co", co)
            M = np.array(ev.matrix_world)
            z = co.reshape(-1, 3) @ M[2, :3] + M[2, 3]
            mask = masks[ob.name]
            if mask is not None and len(mask) == len(z):
                z = z[mask]
            out[i] = min(out[i], float(z.min()))
            ev.to_mesh_clear()
    scene.frame_set(cur)
    return out


def chain_agreement(scene, rig, profile, fcurves, frames) -> dict[str, np.ndarray]:
    """Per limb: how far the end joint (foot, hand) sits in pure FK from where it sits in pure IK.

    ``{limb key: (len(frames),)}`` in world units. The switch curves are muted while measuring and
    put back afterwards; nothing is keyed.
    """
    limbs = [l for l in profile.limbs() if l.switch_bone and l.switch_bone in rig.pose.bones
             and "IK_FK" in rig.pose.bones[l.switch_bone].keys()]
    if not limbs:
        return {}
    switches = [fc for fc in fcurves if fc.data_path.endswith('["IK_FK"]')]
    muted = [fc.mute for fc in switches]
    values = [rig.pose.bones[l.switch_bone]["IK_FK"] for l in limbs]
    cur = scene.frame_current
    MW = np.array(rig.matrix_world)
    pos = {(state, l.key): np.zeros((len(frames), 3)) for state in (0.0, 1.0) for l in limbs}
    try:
        for fc in switches:
            fc.mute = True
        for state in (0.0, 1.0):
            for l in limbs:
                rig.pose.bones[l.switch_bone]["IK_FK"] = state
            rig.update_tag()
            for i, f in enumerate(frames):
                scene.frame_set(int(f))
                for l in limbs:
                    h = rig.pose.bones[l.org_end].head
                    pos[(state, l.key)][i] = MW[:3, :3] @ np.array(h) + MW[:3, 3]
    finally:
        for fc, m in zip(switches, muted):
            fc.mute = m
        for l, v in zip(limbs, values):
            rig.pose.bones[l.switch_bone]["IK_FK"] = v
        rig.update_tag()
        scene.frame_set(cur)
    return {l.key: np.linalg.norm(pos[(0.0, l.key)] - pos[(1.0, l.key)], axis=1) for l in limbs}


def plant_slide(soles: dict, pins, f0: int, fps: float):
    """How fast the grounded part of a pinned foot moves along the floor: ``(p90, max)`` in units per second.

    Inside the locked frames of the floor pins, per frame the sole point that moves least (a foot
    that rolls or turns on its ball is not sliding). ``soles``: ``{"L": (T, K, 3), ...}``.
    """
    speeds = []
    for S, pts in soles.items():
        if len(pts) < 3:
            continue
        v = np.zeros(pts.shape[:2])
        v[1:-1] = np.linalg.norm(pts[2:, :, :2] - pts[:-2, :, :2], axis=2) * (0.5 * fps)
        slow = v.min(axis=1)
        for p in pins:
            if p.owner == f"leg.{S}" and p.mode == "FLOOR" and p.enabled:
                a, b = max(p.lock_in - f0 + 1, 1), min(p.lock_out - f0 - 1, len(pts) - 2)
                if b >= a:
                    speeds.append(slow[a : b + 1])
    if not speeds:
        return 0.0, 0.0
    allv = np.concatenate(speeds)
    return float(np.percentile(allv, 90)), float(allv.max())


def knee_acceleration(profile, smp, fps: float):
    """Largest knee angular acceleration of the take in deg/s^2, and where: ``(value, "L 286")``."""
    worst, where = 0.0, ""
    for S, leg in profile.legs.items():
        a = smp.head_w(leg.org_upper) - smp.head_w(leg.org_lower)
        b = smp.head_w(leg.org_end) - smp.head_w(leg.org_lower)
        cos = np.einsum("ti,ti->t", a, b) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)
        ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        if len(ang) >= 3:
            acc = np.abs(np.diff(ang, 2)) * fps * fps
            if float(acc.max()) > worst:
                worst, where = float(acc.max()), f"{S} {smp.f0 + 1 + int(np.argmax(acc))}"
    return worst, where


def knee_planes(profile, smp):
    """Per leg: which way the knee points (unit vectors across the hip-ankle line) and how far it is off that line."""
    out = {}
    for S, leg in profile.legs.items():
        root, mid, end = smp.head_w(leg.org_upper), smp.head_w(leg.org_lower), smp.head_w(leg.org_end)
        axis = end - root
        axis /= np.linalg.norm(axis, axis=1, keepdims=True) + 1e-12
        perp = (mid - root) - np.sum((mid - root) * axis, axis=1, keepdims=True) * axis
        off = np.linalg.norm(perp, axis=1)
        out[S] = (perp / (off[:, None] + 1e-12), off)
    return out


def take(mods, scene, rig, f0: int, f1: int, meshes=None, samples: int = 240) -> dict:
    """Everything ``compare`` needs about the take as it is now. ``mods``: the add-on's modules
    (``action_io``, ``sampler``, ``ri`` = rig_inspector, ``s1`` = stage1_diagnostics)."""
    profile = mods.ri.inspect_rig(rig)
    smp = mods.sampler.sample_objects([(rig, mods.ri.solver_bone_names(profile, rig.data.bones))], f0, f1)[rig.name]
    T = f1 - f0 + 1
    sparse = np.arange(f0, f1 + 1, max(1, int(np.ceil(T / samples))))
    data = rig.mocap_cleanup
    kinds: dict[str, int] = {}
    for it in data.issues:
        kinds[it.kind] = kinds.get(it.kind, 0) + 1
    return {
        "profile": profile, "smp": smp, "unit": profile.unit, "f0": f0, "sparse": sparse,
        "joints": mods.s1.body_joints(profile, smp),
        "knee_accel": knee_acceleration(profile, smp, scene.render.fps / scene.render.fps_base),
        "knee_planes": knee_planes(profile, smp),
        "chains": chain_agreement(scene, rig, profile, list(mods.action_io.iter_fcurves(rig)), sparse),
        "mesh_low": mesh_lowest(scene, meshes, sparse) if meshes else None,
        "issues": kinds,
    }


def compare(before: dict, after: dict, floor_z: float, limits: dict | None = None, pins=None):
    """Rows ``(what, before, after, limit, ok)`` and the list of failed checks. ``pins``: the floor pins after
    the cleanup; where a foot is planted is where the mesh has to rest on the floor."""
    lim = dict(LIMITS, **(limits or {}))
    u, f0 = before["unit"] / 100.0, before["f0"]            # world units per cm of a 1.75 m character
    rows, failed = [], []

    def row(what, b, a, limit, ok):
        rows.append((what, b, a, limit, ok))
        if ok is False:
            failed.append(what)

    for name in sorted(n for n in before["joints"] if n.startswith("Foot")):
        d = after["joints"][name] - before["joints"][name]
        flat = np.linalg.norm(d[:, :2], axis=1) / u
        i = int(np.argmax(flat))
        row(f"{name} carried sideways", "", f"{flat[i]:.1f} cm at frame {f0 + i} ({100.0 * np.mean(flat > 10.0):.0f} % of frames over 10 cm)",
            f"{lim['foot_cm']} cm", bool(flat[i] <= lim["foot_cm"]))
        row(f"{name} lifted / lowered", "", f"+{max(d[:, 2].max(), 0.0) / u:.1f} / -{max(-d[:, 2].min(), 0.0) / u:.1f} cm", "", None)
    if "Hips" in before["joints"]:
        drop = (before["joints"]["Hips"][:, 2] - after["joints"]["Hips"][:, 2]) / u
        p95 = float(np.percentile(drop, 95))
        row("Hips lowered", "", f"p95 {p95:.2f} cm, max {drop.max():.2f} cm at frame {f0 + int(np.argmax(drop))}",
            f"p95 {lim['pelvis_cm']} cm", bool(p95 <= lim["pelvis_cm"]))
    for name in sorted(n for n in before["joints"] if n.startswith(("Knee", "Hand", "Head"))):
        d = np.linalg.norm(after["joints"][name] - before["joints"][name], axis=1) / u
        row(f"{name} moved", "", f"{d.max():.1f} cm at frame {f0 + int(np.argmax(d))} (mean {d.mean():.1f})", "", None)
    for S in sorted(before["knee_planes"]):
        (db, ob), (da, oa) = before["knee_planes"][S], after["knee_planes"][S]
        bent = (ob > 2.0 * u) & (oa > 2.0 * u)                  # a knee that is 2 cm off the line points somewhere
        if bent.any():
            turn = np.degrees(np.arccos(np.clip(np.sum(db * da, axis=1), -1.0, 1.0)))
            turn[~bent] = 0.0
            i = int(np.argmax(turn))
            row(f"Knee {S} turned sideways", "", f"{turn[i]:.1f} deg at frame {f0 + i} (p95 {np.percentile(turn[bent], 95):.1f})",
                f"{lim['knee_turn']} deg", bool(turn[i] <= lim["knee_turn"]))
    (kb, _wb), (ka, wa) = before["knee_accel"], after["knee_accel"]
    row("Knee acceleration, max", f"{kb:.0f}", f"{ka:.0f} deg/s^2 (knee {wa})",
        f"{lim['knee_accel']} x raw and +{lim['knee_accel_abs']} deg/s^2",
        not (ka > lim["knee_accel"] * kb and ka - kb > lim["knee_accel_abs"]))
    pb, pa = before["issues"].get("KNEE_POP", 0), after["issues"].get("KNEE_POP", 0)
    row("Joint pops flagged", str(pb), str(pa), "not more", pa <= pb)
    for key in sorted(before["chains"]):
        b, a = before["chains"][key].max() / u, after["chains"].get(key, np.zeros(1)).max() / u
        in_sync = b <= lim["chains_cm"]
        row(f"IK against FK, {key}", f"{b:.2f} cm", f"{a:.2f} cm", f"{lim['chains_cm']} cm" if in_sync else "(was not in sync)",
            bool(a <= lim["chains_cm"]) if in_sync else None)
    if after["mesh_low"] is not None:
        # The mesh has to rest on the floor where a foot is planted; in the air or mid-step it is above it,
        # and a median over every frame of a jumpy take says more about the jumps than about the floor.
        sparse = after["sparse"]
        planted = np.zeros(len(sparse), dtype=bool)
        for p in pins or ():
            if getattr(p, "enabled", True) and getattr(p, "mode", "FLOOR") == "FLOOR" and str(p.owner).startswith("leg."):
                planted |= (sparse >= p.lock_in) & (sparse <= p.lock_out)
        sel = planted if planted.sum() >= 5 else np.ones(len(sparse), dtype=bool)
        b = (np.median(before["mesh_low"][sel]) - floor_z) / u
        a = (np.median(after["mesh_low"][sel]) - floor_z) / u
        every = (np.median(after["mesh_low"]) - floor_z) / u
        where = f"{int(sel.sum())} planted frames" if planted.sum() >= 5 else "all frames"
        row("Character mesh rests at (floor = 0)", f"{b:+.2f} cm",
            f"{a:+.2f} cm on {where} (all frames {every:+.2f}), lowest {(after['mesh_low'].min() - floor_z) / u:+.2f} cm",
            f"+-{lim['mesh_cm']} cm", bool(abs(a) <= lim["mesh_cm"]))
    return rows, failed


def print_rows(rows) -> None:
    w = max(len(r[0]) for r in rows)
    for what, b, a, limit, ok in rows:
        mark = {True: "ok  ", False: "FAIL", None: "    "}[ok]
        arrow = f"{b} -> {a}" if b else a
        print(f"  {mark} {what:<{w}}  {arrow}" + (f"   [limit {limit}]" if limit else ""))
