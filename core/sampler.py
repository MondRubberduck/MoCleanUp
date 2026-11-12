"""Frame-stepping pose sampler and exact pose <-> channel conversion.

The cleanup stages reason about *evaluated* world-space motion (after parent switches, constraints,
IK and drivers), so the rig is sampled once per operation by stepping the scene. Corrections are
then computed in NumPy and converted back to local channels with ``Bone.convert_local_to_pose``,
using the sampled per-frame parent matrices. No "static parent" assumption is made: an animated
``root`` control or a parent-switched IK control is handled exactly.
"""

from __future__ import annotations

import bpy
import numpy as np
from mathutils import Matrix, Vector

from .math_quat import quat_from_matrix, quat_hemisphere_continuous


class RigSample:
    """Evaluated matrices of selected bones for every frame of ``[f0, f1]``."""

    def __init__(self, obj_name: str, f0: int, f1: int, names: list[str], pose: np.ndarray, obj_world: np.ndarray):
        self.obj_name = obj_name
        self.f0 = int(f0)
        self.f1 = int(f1)
        self.T = self.f1 - self.f0 + 1
        self.names = list(names)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.pose = pose              # (T, B, 4, 4) armature space
        self.obj_world = obj_world    # (T, 4, 4)
        self._obj_world_inv = None

    def has(self, name: str) -> bool:
        return name in self.index

    def pose_m(self, name: str) -> np.ndarray:
        return self.pose[:, self.index[name]]

    def world_m(self, name: str) -> np.ndarray:
        return self.obj_world @ self.pose[:, self.index[name]]

    def head_w(self, name: str) -> np.ndarray:
        return self.world_m(name)[:, :3, 3]

    def point_w(self, name: str, local_point) -> np.ndarray:
        """World trajectory ``(T, 3)`` of a point given in the bone's local space."""
        p = np.append(np.asarray(local_point, dtype=np.float64), 1.0)
        return (self.world_m(name) @ p)[:, :3]

    @property
    def obj_world_inv(self) -> np.ndarray:
        if self._obj_world_inv is None:
            self._obj_world_inv = np.linalg.inv(self.obj_world)
        return self._obj_world_inv

    def to_pose_space(self, world_mats: np.ndarray) -> np.ndarray:
        return self.obj_world_inv @ world_mats


ISOLATED_SCENE = "mocap_cleanup_sampling"
last_mode = ""                       # "isolated" or "scene": how the last call evaluated the rig
_trusted: set[frozenset] = set()     # object sets whose isolated evaluation matched the scene
_scene_bound: set[frozenset] = set() # object sets that only evaluate correctly inside their scene
_probe = 0


def reset_cache() -> None:
    _trusted.clear()
    _scene_bound.clear()


def _read(obj, entry, t: int) -> None:
    _o, _names, idx, buf, pose, ow = entry
    ow[t] = np.array(obj.matrix_world, dtype=np.float64)
    if idx is not None:
        obj.pose.bones.foreach_get("matrix", buf)
        # Blender stores matrices column-major: transpose to math (row, col) layout.
        pose[t] = buf.reshape(-1, 4, 4)[idx].transpose(0, 2, 1)


def _sample_isolated(entries, f0: int, T: int, scene) -> None:
    """Evaluate the requested objects in a temporary scene that holds nothing else.

    Whatever they depend on (parents, constraint targets, driver sources) is still evaluated, the
    rest of the scene (other characters, meshes, modifiers) is not.
    """
    tmp = bpy.data.scenes.new(ISOLATED_SCENE)
    try:
        tmp.render.fps, tmp.render.fps_base = scene.render.fps, scene.render.fps_base
        tmp.frame_start, tmp.frame_end = scene.frame_start, scene.frame_end
        for obj in {e[0] for e in entries}:
            tmp.collection.objects.link(obj)
        layer = tmp.view_layers[0]
        for t in range(T):
            tmp.frame_set(f0 + t)
            deps = layer.depsgraph
            for entry in entries:
                _read(entry[0].evaluated_get(deps), entry, t)
    finally:
        bpy.data.scenes.remove(tmp)


def _sample_scene(entries, f0: int, T: int, scene, frames=None, hide_other_meshes: bool = True):
    """Step the user's scene. ``frames``: only these indices, returned as ``{t: [(world, pose), ...]}``."""
    keep = {e[0].name for e in entries}
    hidden = []
    if hide_other_meshes and frames is None:
        # Deforming meshes are expensive to evaluate and irrelevant for bone matrices.
        for ob in scene.objects:
            if ob.type == "MESH" and ob.name not in keep and not ob.hide_viewport and ob.modifiers:
                hidden.append(ob)
                ob.hide_viewport = True
    saved_frame, saved_sub = scene.frame_current, scene.frame_subframe
    probes = {}
    try:
        for t in (range(T) if frames is None else frames):
            scene.frame_set(f0 + t)
            if frames is None:
                for entry in entries:
                    _read(entry[0], entry, t)
            else:
                rows = []
                for obj, _names, idx, buf, _pose, _ow in entries:
                    world = np.array(obj.matrix_world, dtype=np.float64)
                    mats = None
                    if idx is not None:
                        obj.pose.bones.foreach_get("matrix", buf)
                        mats = buf.reshape(-1, 4, 4)[idx].transpose(0, 2, 1).astype(np.float64)
                    rows.append((world, mats))
                probes[t] = rows
    finally:
        for ob in hidden:
            ob.hide_viewport = False
        scene.frame_set(saved_frame, subframe=saved_sub)
    return probes


def _agrees(entries, probes: dict) -> bool:
    for t, rows in probes.items():
        for (_obj, _names, idx, _buf, pose, ow), (world, mats) in zip(entries, rows):
            if np.abs(world - ow[t]).max() > 1e-4 * max(1.0, float(np.abs(world).max())):
                return False
            if mats is not None and len(mats) and np.abs(mats - pose[t]).max() > 1e-4 * max(1.0, float(np.abs(mats).max())):
                return False
    return True


def sample_objects(requests: list[tuple[object, list[str] | None]], f0: int, f1: int, scene=None,
                   hide_other_meshes: bool = True, isolate: bool = True) -> dict[str, RigSample]:
    """Record the evaluated matrices of all requested objects for every frame of ``[f0, f1]``.

    ``requests``: ``(object, bone names)`` pairs; bone names are ignored for non-armature objects.
    Returns ``{object.name: RigSample}``.

    The objects are evaluated on their own in a temporary scene (several times faster in a
    production scene with more characters and meshes) and a few frames are compared with the real
    scene. If they differ - physics, drivers reading the scene - the whole scene is stepped instead.
    """
    global last_mode, _probe
    scene = scene or bpy.context.scene
    T = f1 - f0 + 1
    entries = []
    seen = set()
    for obj, names in requests:
        if obj is None or obj.name in seen:
            continue
        seen.add(obj.name)
        if obj.type == "ARMATURE" and names:
            all_names = [pb.name for pb in obj.pose.bones]
            lookup = {n: i for i, n in enumerate(all_names)}
            names = [n for n in dict.fromkeys(names) if n in lookup]
            idx = np.array([lookup[n] for n in names], dtype=np.int64)
            buf = np.empty(len(all_names) * 16, dtype=np.float32)
            pose = np.empty((T, len(names), 4, 4), dtype=np.float64)
        else:
            names, idx, buf, pose = [], None, None, np.empty((T, 0, 4, 4), dtype=np.float64)
        entries.append((obj, names, idx, buf, pose, np.empty((T, 4, 4), dtype=np.float64)))

    key = frozenset(seen)
    done = False
    if isolate and entries and T > 8 and key not in _scene_bound:
        try:
            _sample_isolated(entries, f0, T, scene)
            if key in _trusted:                      # keep spot-checking one (changing) frame
                _probe += 1
                frames = [(_probe * 7919) % T]
            else:
                frames = sorted({0, T // 2, T - 1})
            done = _agrees(entries, _sample_scene(entries, f0, T, scene, frames=frames))
        except Exception:
            done = False
        if done:
            _trusted.add(key)
        else:
            _trusted.discard(key)
            _scene_bound.add(key)
            print(f"MoCap Cleanup: {sorted(seen)} evaluate differently outside their scene; sampling the whole scene")
    if not done:
        _sample_scene(entries, f0, T, scene, hide_other_meshes=hide_other_meshes)
    last_mode = "isolated" if done else "scene"
    return {obj.name: RigSample(obj.name, f0, f1, names, pose, ow) for obj, names, idx, buf, pose, ow in entries}


def sample_frames(obj, names: list[str], frames, scene=None) -> RigSample:
    """Evaluated matrices of ``names`` at a few frames (the user's scene is stepped to each).

    Returns a ``RigSample`` whose time axis is the list of frames, in the order given.
    """
    scene = scene or bpy.context.scene
    all_names = [pb.name for pb in obj.pose.bones]
    lookup = {n: i for i, n in enumerate(all_names)}
    names = [n for n in dict.fromkeys(names) if n in lookup]
    idx = np.array([lookup[n] for n in names], dtype=np.int64)
    buf = np.empty(len(all_names) * 16, dtype=np.float32)
    frames = [int(f) for f in frames]
    probes = _sample_scene([(obj, names, idx, buf, None, None)], 0, 0, scene, frames=frames)
    world = np.stack([probes[f][0][0] for f in frames])
    pose = np.stack([probes[f][0][1] for f in frames])
    return RigSample(obj.name, 0, len(frames) - 1, names, pose, world)


# ---------------------------------------------------------------------------
# Pose matrix -> local channels
# ---------------------------------------------------------------------------

def pose_to_basis(rig, bone_name: str, pose_mats: np.ndarray, parent_pose_mats: np.ndarray | None):
    """Convert desired pose-space matrices ``(T, 4, 4)`` of a bone into local channels.

    ``parent_pose_mats`` are the (already updated) pose matrices of the bone's parent.
    Returns ``(loc (T, 3), quat (T, 4))``; inherit-rotation / inherit-scale / local-location flags
    are honoured by ``Bone.convert_local_to_pose``.
    """
    bone = rig.data.bones[bone_name]
    rest = bone.matrix_local
    parent = bone.parent
    T = len(pose_mats)
    loc = np.empty((T, 3), dtype=np.float64)
    rot = np.empty((T, 3, 3), dtype=np.float64)
    conv = bone.convert_local_to_pose
    if parent is not None and parent_pose_mats is not None:
        prest = parent.matrix_local
        for t in range(T):
            b = conv(Matrix(pose_mats[t].tolist()), rest, parent_matrix=Matrix(parent_pose_mats[t].tolist()),
                     parent_matrix_local=prest, invert=True)
            loc[t] = b.translation
            rot[t] = b.to_3x3()
    else:
        for t in range(T):
            b = conv(Matrix(pose_mats[t].tolist()), rest, invert=True)
            loc[t] = b.translation
            rot[t] = b.to_3x3()
    quat, _ = quat_hemisphere_continuous(quat_from_matrix(rot))
    return loc, quat


def parent_name(rig, bone_name: str) -> str:
    p = rig.data.bones[bone_name].parent
    return p.name if p is not None else ""


# ---------------------------------------------------------------------------
# Dependency probing
# ---------------------------------------------------------------------------

_PROBE = Matrix.Translation((0.137, -0.211, 0.173)) @ Matrix.Rotation(0.31, 4, Vector((0.36, 0.48, 0.8)))


def probe_follow(rig, drivers: list[str], followers: list[str], tol: float = 2e-4) -> dict[str, dict[str, str]]:
    """Find out how the *parent* of each follower control reacts when a driver control moves.

    Returns ``result[follower][driver]`` in ``{'RIGID', 'NONE', 'PARTIAL'}``. ``RIGID`` means the
    follower's parent matrix is carried along by exactly the driver's pose-space delta (parented or
    parent-switched to it); ``NONE`` means it does not react. The probe is done at the current frame
    and fully restored.
    """
    pbones = rig.pose.bones
    view_layer = bpy.context.view_layer
    result: dict[str, dict[str, str]] = {f: {} for f in followers}
    probe_np = np.array(_PROBE, dtype=np.float64)
    scale = max(float(np.mean([abs(s) for s in rig.matrix_world.to_scale()])), 1e-9)

    def parent_mats():
        out = {}
        for f in followers:
            pb = pbones.get(f)
            if pb is not None and pb.parent is not None:
                out[f] = np.array(pb.parent.matrix, dtype=np.float64)
        return out

    view_layer.update()
    before = parent_mats()
    for d in drivers:
        pbd = pbones.get(d)
        if pbd is None:
            continue
        saved = pbd.matrix_basis.copy()
        try:
            pbd.matrix = _PROBE @ pbd.matrix
            view_layer.update()
            after = parent_mats()
        finally:
            pbd.matrix_basis = saved
            view_layer.update()
        for f in followers:
            if f == d or f not in before:
                result[f][d] = "NONE"
                continue
            size = max(1.0, float(np.abs(before[f][:3, 3]).max()) / scale)
            err_rigid = float(np.abs(after[f] - probe_np @ before[f]).max())
            err_none = float(np.abs(after[f] - before[f]).max())
            if err_none <= tol * size:
                result[f][d] = "NONE"
            elif err_rigid <= tol * size * 5.0:
                result[f][d] = "RIGID"
            else:
                result[f][d] = "PARTIAL"
    return result
