"""Where the character stands: the sole of its mesh, and one vertical shift that puts a take on the floor.

The floor is what the viewer sees: the grid, or the Floor Height. The rig's sole points (heel and ball
from the foot bones) are not where the shoe is: a rig built for a game skeleton puts them at the toe joint,
2-3 cm above the mesh sole. ``sole_offset`` measures that gap on the character mesh at frames where a foot
rests; the solver then keeps the rig's soles that much above the floor, so the mesh lands on it (QA M3).

``shift_take`` moves the whole take up or down by one constant (on the rig's root control), so the feet
rest on the floor without bending a knee. It never changes the motion itself.
"""

from __future__ import annotations

import bpy
import numpy as np
from mathutils import Vector

from . import action_io


def character_meshes(scene, rig) -> list:
    """Visible meshes deformed by ``rig``, or by a skeleton constrained to it (a game skeleton under a control rig)."""
    out = []
    for ob in scene.objects:
        if ob.type != "MESH" or not ob.visible_get():
            continue
        for m in ob.modifiers:
            arm = m.object if m.type == "ARMATURE" and m.show_viewport else None
            if arm is not None and (arm == rig or any(getattr(c, "target", None) == rig
                                                     for pb in arm.pose.bones for c in pb.constraints)):
                out.append(ob)
                break
    return out


_moving_cache: dict = {}


def moving_vertices(ob):
    """Mask of the vertices that follow a bone: weighted to a deforming bone of the mesh's armature.

    Vertices without such a weight stay where the mesh was modelled whatever the take does, so they
    say nothing about where the feet are (an actor mesh with a few loose vertices at the origin sat
    "1 cm under the floor" on every frame, QA N3). ``None`` when every vertex follows a bone, or when
    it can't be told (no vertex groups, envelope deformation).
    """
    arms = [m.object for m in ob.modifiers
            if m.type == "ARMATURE" and m.object is not None and m.object.type == "ARMATURE" and m.use_vertex_groups]
    if not arms or not ob.vertex_groups or not len(ob.data.vertices):
        return None
    key = (ob.data.name, len(ob.data.vertices), tuple(a.name for a in arms))
    if key in _moving_cache:
        return _moving_cache[key]
    bones = {b.name for a in arms for b in a.data.bones if b.use_deform}
    groups = {g.index for g in ob.vertex_groups if g.name in bones}
    mask = np.array([any(g.group in groups and g.weight > 0.0 for g in v.groups) for v in ob.data.vertices], dtype=bool)
    mask = None if (mask.all() or not mask.any()) else mask
    _moving_cache[key] = mask
    return mask


def mesh_lowest(scene, meshes, frames) -> np.ndarray:
    """Lowest world z of the deformed meshes at each of ``frames`` (restores the current frame).

    Only vertices that follow a bone count (``moving_vertices``)."""
    out = np.full(len(frames), np.inf)
    cur = scene.frame_current
    _moving_cache.clear()                      # weights may have changed since the last call
    masks = {ob.name: moving_vertices(ob) for ob in meshes}
    try:
        for i, f in enumerate(frames):
            scene.frame_set(int(f))
            dg = bpy.context.evaluated_depsgraph_get()
            for ob in meshes:
                ev = ob.evaluated_get(dg)
                me = ev.to_mesh()
                try:
                    if len(me.vertices):
                        co = np.empty(len(me.vertices) * 3)
                        me.vertices.foreach_get("co", co)
                        M = np.array(ev.matrix_world)
                        z = co.reshape(-1, 3) @ M[2, :3] + M[2, 3]
                        mask = masks[ob.name]
                        if mask is not None and len(mask) == len(z):
                            z = z[mask]
                        out[i] = min(out[i], float(z.min()))
                finally:
                    ev.to_mesh_clear()
    finally:
        scene.frame_set(cur)
    return out


def sole_offset(scene, rig, soles: dict, f0: int, rest_z: float, unit: float, samples: int = 12):
    """How far the character mesh rests below the rig's sole points: ``(offset, n_frames)``.

    Measured where the lowest sole point is at its resting height (``rest_z`` within 5 mm): there the
    lowest point of the mesh is the planted shoe. ``(0.0, 0)`` without a visible character mesh.
    """
    meshes = character_meshes(scene, rig)
    if not meshes or not soles:
        return 0.0, 0
    low = np.min([p[..., 2].min(axis=1) for p in soles.values()], axis=0)
    rest = np.flatnonzero(np.abs(low - rest_z) < 0.005 * unit)
    if len(rest) == 0:
        rest = np.argsort(np.abs(low - rest_z))[:samples]
    pick = rest[np.linspace(0, len(rest) - 1, min(samples, len(rest))).round().astype(int)]
    mesh_low = mesh_lowest(scene, meshes, [f0 + int(i) for i in pick])
    ok = np.isfinite(mesh_low)
    if not ok.any():
        return 0.0, 0
    return float(np.median(low[pick][ok] - mesh_low[ok])), int(ok.sum())


def mesh_rest(scene, rig, soles: dict, f0: int, rest_z: float, unit: float, samples: int = 16):
    """Where the character mesh rests: median lowest point at frames where a sole rests (``rest_z`` within
    5 mm). None without a visible character mesh."""
    meshes = character_meshes(scene, rig)
    if not meshes or not soles:
        return None
    low = np.min([p[..., 2].min(axis=1) for p in soles.values()], axis=0)
    rest = np.flatnonzero(np.abs(low - rest_z) < 0.005 * unit)
    if len(rest) == 0:
        return None
    pick = rest[np.linspace(0, len(rest) - 1, min(samples, len(rest))).round().astype(int)]
    ml = mesh_lowest(scene, meshes, [f0 + int(i) for i in pick])
    ml = ml[np.isfinite(ml)]
    return float(np.median(ml)) if len(ml) else None


def shift_take(rig, root: str, dz: float) -> bool:
    """Move the whole take by ``dz`` (world units, up) on the root control. False if there is no usable root.

    A pose bone's location lives in its rest frame, so one constant in that frame is the same world shift on
    every frame, whatever the root's own rotation does."""
    pb = rig.pose.bones.get(root) if root else None
    if pb is None or pb.parent is not None or abs(dz) < 1e-9:
        return False
    M = rig.matrix_world.to_3x3() @ pb.bone.matrix_local.to_3x3()
    d = M.inverted() @ Vector((0.0, 0.0, dz))
    cb = action_io.ensure_channelbag(rig)
    path = action_io.bone_data_path(root, "location")
    f0, _f1 = action_io.get_action_frame_range(rig)
    for i in range(3):
        if abs(d[i]) < 1e-12:
            continue
        fc = action_io.find_fcurve(cb, path, i)
        if fc is not None and len(fc.keyframe_points):
            n = len(fc.keyframe_points)
            for attr in ("co", "handle_left", "handle_right"):
                co = np.empty(2 * n)
                fc.keyframe_points.foreach_get(attr, co)
                co[1::2] += d[i]
                fc.keyframe_points.foreach_set(attr, co)
            fc.update()
        else:
            fc = action_io.ensure_fcurve(cb, path, i, group_name=root)
            fc.keyframe_points.insert(f0, float(pb.location[i]) + d[i], options={"FAST"})
            fc.update()
    return True
