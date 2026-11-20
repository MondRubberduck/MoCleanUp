"""Frame-stepping pose sampler and exact pose <-> channel conversion.

The cleanup stages reason about *evaluated* world-space motion (after parent switches, constraints,
IK and drivers), so the rig is sampled once per operation by stepping the scene. Corrections are
then computed in NumPy and converted back to local channels with ``Bone.convert_local_to_pose``,
using the sampled per-frame parent matrices. No "static parent" assumption is made: an animated
``root`` control or a parent-switched IK control is handled exactly.
"""

from __future__ import annotations

import re
import zlib

import bpy
import numpy as np
from mathutils import Matrix, Vector

from . import action_io
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
last_reuse = ""                      # "full", "partial" or "cached": what the last call had to evaluate
last_frames = 0                      # ... and on how many frames
_trusted: set[frozenset] = set()     # object sets whose isolated evaluation matched the scene
_scene_bound: set[frozenset] = set() # object sets that only evaluate correctly inside their scene
_probe = 0
_kept: dict[tuple, dict] = {}        # (reuse name, object names) -> the sample kept under that name
_no_reuse: set[tuple] = set()        # ... for which updating single frames did not reproduce the kept sample
KEEP_SETS = 3                        # characters whose samples are kept at the same time


def reset_cache() -> None:
    _trusted.clear()
    _scene_bound.clear()
    _kept.clear()
    _no_reuse.clear()


# ---------------------------------------------------------------------------
# Which frames changed
# ---------------------------------------------------------------------------

def _plain(value):
    if isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "to_dict"):
        return repr(value.to_dict())
    if hasattr(value, "to_list"):
        return tuple(value.to_list())
    return type(value).__name__


def _static_signature(obj, action) -> tuple:
    """What the pose depends on besides curves and un-keyed channels: the action and the object's transform.
    Constraint settings are not watched."""
    return (action.name if action is not None else "", tuple(round(v, 6) for row in obj.matrix_world for v in row),
            len(obj.pose.bones) if obj.type == "ARMATURE" else 0)


SAME = 1e-5                          # a curve within this of what it was has not changed (channel units)
_MIX = np.uint64(0x9E3779B97F4A7C15)
_COLS = np.array([0x9E3779B97F4A7C15, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9, 0x27D4EB2F165667C5, 0x85EBCA77C2B2AE63,
                  0xD6E8FEB86659FD93, 0xA0761D6478BD642F], dtype=np.uint64)


_CHANNELS = {"location": (0, 3), "rotation_quaternion": (3, 4), "rotation_euler": (7, 3), "rotation_axis_angle": (10, 4),
             "scale": (14, 3)}
_BONE_PATH = re.compile(r'^pose\.bones\["(.*)"\]\.(\w+)$')


class CurveState:
    """What ``obj``'s pose depends on at every frame of a range, to tell later which frames are still the same.

    * a row of values per curve that has one key per frame (a baked take) or is constant;
    * a 64-bit sum per frame over the keys of all other (sparse) curves;
    * the values of the channels and custom properties that are not keyed, and the signature.
    """

    def __init__(self, obj, f0: int, f1: int):
        T = f1 - f0 + 1
        self.T = T
        rows: dict[tuple, np.ndarray] = {}
        sparse = np.zeros(T, dtype=np.uint64)
        keyed: set[tuple] = set()
        with np.errstate(over="ignore"):
            for fc in action_io.iter_fcurves(obj):
                pts = fc.keyframe_points
                n = len(pts)
                if not n:
                    continue
                ident = (fc.data_path, fc.array_index)
                keyed.add(ident)
                co = np.empty(n * 2, dtype=np.float32)
                pts.foreach_get("co", co)
                co = co.reshape(n, 2)
                x = co[:, 0]
                if n == 1 or bool(np.all(co[:, 1] == co[0, 1])):
                    rows[ident] = np.full(T, co[0, 1], dtype=np.float32)           # constant, however it is keyed
                    continue
                if float(x[-1] - x[0]) == n - 1 and bool(np.all(np.diff(x) == 1.0)):
                    # one key per frame: the value on each frame, held before and after the keys
                    row = np.empty(T, dtype=np.float32)
                    first = int(round(float(x[0]))) - f0
                    a, b = max(first, 0), min(first + n, T)
                    row[: max(min(first, T), 0)] = co[0, 1]
                    row[max(first + n, 0) :] = co[-1, 1]
                    if b > a:
                        row[a:b] = co[a - first : b - first, 1]
                    rows[ident] = row
                    continue
                # sparse keys: a sum per frame over the keys that decide the value there (the key, its handles and
                # interpolation, over the span to its neighbours). Whole numbers that wrap around, so the sum
                # does not depend on the order of the curves.
                seed = np.uint64(zlib.crc32(f"{fc.data_path}|{fc.array_index}".encode()))
                mult = (seed * _MIX + np.uint64(0x632BE59BD9B4E019)) | np.uint64(1)
                data = np.empty((n, 7), dtype=np.float32)
                data[:, 0:2] = co
                buf = np.empty(n * 2, dtype=np.float32)
                pts.foreach_get("handle_left", buf)
                data[:, 2:4] = buf.reshape(n, 2)
                pts.foreach_get("handle_right", buf)
                data[:, 4:6] = buf.reshape(n, 2)
                kind = np.empty(n, dtype=np.int32)
                pts.foreach_get("interpolation", kind)
                data[:, 6] = kind
                key = (data.view(np.uint32).astype(np.uint64) * _COLS[None, :]).sum(axis=1, dtype=np.uint64) * mult
                lo = np.concatenate([[0], np.clip(np.floor(x[:-1]).astype(np.int64) - f0, 0, T)])
                hi = np.concatenate([np.clip(np.ceil(x[1:]).astype(np.int64) - f0 + 1, 0, T), [T]])
                steps = np.zeros(T + 1, dtype=np.uint64)
                np.add.at(steps, lo, key)
                np.subtract.at(steps, hi, key)
                sparse += np.cumsum(steps, dtype=np.uint64)[:T]
        self.signature = _static_signature(obj, action_io.get_active_action(obj))
        self.keys = tuple(sorted(rows))
        self.index = {k: i for i, k in enumerate(self.keys)}
        self.values = np.stack([rows[k] for k in self.keys]) if rows else np.zeros((0, T), dtype=np.float32)
        # rows that are the four components of one quaternion: q and -q are the same rotation
        quats = [[self.index.get((path, i)) for i in range(4)] for path, j in self.keys
                 if j == 0 and path.endswith("rotation_quaternion")]
        self.quats = np.array([q for q in quats if None not in q], dtype=np.int64).reshape(-1, 4)
        self.sparse = sparse
        # what is not keyed: channels of the pose bones, custom properties
        self.static: dict[tuple, float] = {}
        self.other: tuple = ()
        if obj.type == "ARMATURE":
            bones = obj.pose.bones
            n = len(bones)
            flat = np.empty((n, 17), dtype=np.float32)
            for prop, (col, width) in _CHANNELS.items():
                buf = np.empty(n * width, dtype=np.float32)
                bones.foreach_get(prop, buf)
                flat[:, col : col + width] = buf.reshape(n, width)
            self._flat = flat
            self._rows = {pb.name: i for i, pb in enumerate(bones)}
            other = []
            for pb in bones:
                for name in pb.keys():
                    ident = (action_io.bone_prop_data_path(pb.name, name), 0)
                    if ident in keyed:
                        continue
                    value = pb[name]
                    if isinstance(value, (bool, int, float)):
                        self.static[ident] = float(value)
                    else:
                        other.append((pb.name, name, _plain(value)))
            self.other = tuple(other)
        else:
            self._flat = np.zeros((0, 17), dtype=np.float32)
            self._rows = {}
        self._keyed = keyed

    def _unkeyed(self, ident: tuple):
        """Value of a channel that has no curve here (None: not one this class knows)."""
        if ident in self._keyed:
            return None
        if ident in self.static:
            return self.static[ident]
        m = _BONE_PATH.match(ident[0])
        if m and m.group(2) in _CHANNELS and m.group(1) in self._rows:
            col, width = _CHANNELS[m.group(2)]
            if ident[1] < width:
                return float(self._flat[self._rows[m.group(1)], col + ident[1]])
        return None

    def _row(self, ident: tuple):
        i = self.index.get(ident)
        if i is not None:
            return self.values[i]
        value = self._unkeyed(ident)
        return None if value is None else np.full(self.T, value, dtype=np.float32)

    def changed_since(self, then: "CurveState") -> np.ndarray:
        """Frames ``(T,)`` on which something is not what it was in ``then``.

        Curve values are compared with a tolerance (``SAME``): what the cleanup writes again comes out of an
        evaluated pose and differs in the last digits on frames nothing happened to. A channel that got its
        first curve is compared with the value it had without one. Anything that cannot be compared frame by
        frame - the signature, sparse keys, an un-keyed control that was moved - counts for every frame it can
        reach."""
        T = self.T
        everything = np.ones(T, dtype=bool)
        if self.signature != then.signature or self.other != then.other or self._flat.shape != then._flat.shape or T != then.T:
            return everything
        free = np.ones(self._flat.shape, dtype=bool)                  # channels without a curve in both
        for state in (self, then):
            for path, i in state._keyed:
                m = _BONE_PATH.match(path)
                if m and m.group(2) in _CHANNELS and m.group(1) in state._rows and i < _CHANNELS[m.group(2)][1]:
                    free[state._rows[m.group(1)], _CHANNELS[m.group(2)][0] + i] = False
        if np.any((np.abs(self._flat - then._flat) > SAME) & free):
            return everything
        for ident in self.static.keys() & then.static.keys():
            if abs(self.static[ident] - then.static[ident]) > SAME:
                return everything
        changed = self.sparse != then.sparse
        if self.keys == then.keys:
            if len(self.keys):
                diff = np.abs(self.values - then.values) > SAME
                if len(self.quats):
                    # the same rotation with the other sign is no change
                    flipped = (np.abs(self.values[self.quats] + then.values[self.quats]) <= SAME).all(axis=1)   # (Q, T)
                    diff[self.quats] &= ~flipped[:, None, :]
                changed |= diff.any(axis=0)
            return changed
        for ident in set(self.keys) | set(then.keys):
            a, b = self._row(ident), then._row(ident)
            if a is None or b is None:
                return everything
            changed |= np.abs(a - b) > SAME
        return changed

    def take(self, now: "CurveState", idx: np.ndarray) -> "CurveState":
        """The state after the frames ``idx`` were sampled again in the state ``now``: those frames as they are
        now, the others as they were when they were sampled."""
        keep = np.ones(self.T, dtype=bool)
        keep[idx] = False
        if now.keys == self.keys:
            now.values[:, keep] = self.values[:, keep]
        else:
            for i, ident in enumerate(now.keys):
                old = self._row(ident)
                if old is not None:
                    now.values[i, keep] = old[keep]
        now.sparse[keep] = self.sparse[keep]
        return now


def _read(obj, entry, t: int) -> None:
    _o, _names, idx, buf, pose, ow = entry
    ow[t] = np.array(obj.matrix_world, dtype=np.float64)
    if idx is not None:
        obj.pose.bones.foreach_get("matrix", buf)
        # Blender stores matrices column-major: transpose to math (row, col) layout.
        pose[t] = buf.reshape(-1, 4, 4)[idx].transpose(0, 2, 1)


def _sample_isolated(entries, f0: int, T: int, scene, frames=None) -> None:
    """Evaluate the requested objects in a temporary scene that holds nothing else.

    Whatever they depend on (parents, constraint targets, driver sources) is still evaluated, the
    rest of the scene (other characters, meshes, modifiers) is not. ``frames``: these scene frames
    (``T`` of them) instead of ``f0 .. f0 + T - 1``.
    """
    tmp = bpy.data.scenes.new(ISOLATED_SCENE)
    try:
        tmp.render.fps, tmp.render.fps_base = scene.render.fps, scene.render.fps_base
        tmp.frame_start, tmp.frame_end = scene.frame_start, scene.frame_end
        for obj in {e[0] for e in entries}:
            tmp.collection.objects.link(obj)
        layer = tmp.view_layers[0]
        for t in range(T):
            tmp.frame_set(f0 + t if frames is None else int(frames[t]))
            deps = layer.depsgraph
            for entry in entries:
                _read(entry[0].evaluated_get(deps), entry, t)
    finally:
        bpy.data.scenes.remove(tmp)


def _sample_scene(entries, f0: int, T: int, scene, frames=None, hide_other_meshes: bool = True):
    # (with ``frames`` the default only hides meshes for more than three of them: probes stay cheap and exact)
    """Step the user's scene. ``frames``: only these indices, returned as ``{t: [(world, pose), ...]}``."""
    keep = {e[0].name for e in entries}
    hidden = []
    if hide_other_meshes and (frames is None or len(frames) > 3):
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


def _entries(requests, T: int):
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
    return entries, frozenset(seen)


def sample_objects(requests: list[tuple[object, list[str] | None]], f0: int, f1: int, scene=None,
                   hide_other_meshes: bool = True, isolate: bool = True, reuse: str = "") -> dict[str, RigSample]:
    """Record the evaluated matrices of all requested objects for every frame of ``[f0, f1]``.

    ``requests``: ``(object, bone names)`` pairs; bone names are ignored for non-armature objects.
    Returns ``{object.name: RigSample}``.

    The objects are evaluated on their own in a temporary scene (several times faster in a
    production scene with more characters and meshes) and a few frames are compared with the real
    scene. If they differ - physics, drivers reading the scene - the whole scene is stepped instead.

    ``reuse``: a name for this caller's sample. Evaluating a control rig is what takes the time (7 s for a
    3-minute take), and after a small edit almost every frame is as it was. The sample taken under this name
    is kept, and the next call evaluates only the frames whose curves are not what they were when that frame
    was sampled (``CurveState``), plus one that did not change, as a check. The samples returned are then the kept ones, updated in place: do not hold
    on to them across calls. ``last_reuse`` / ``last_frames`` say what had to be evaluated.
    """
    global last_reuse, last_frames
    scene = scene or bpy.context.scene
    T = f1 - f0 + 1
    if not reuse:
        last_reuse, last_frames = "full", T
        return _sample_full(requests, f0, f1, scene, hide_other_meshes, isolate)

    objects = {}
    for obj, _names in requests:
        if obj is not None:
            objects.setdefault(obj.name, obj)
    key = (reuse, tuple(sorted(objects)))
    now = {name: CurveState(obj, f0, f1) for name, obj in objects.items()}
    wanted = {e[0].name: tuple(e[1]) for e in _entries(requests, 0)[0]}
    kept = _kept.get(key)
    if kept is not None and kept["range"] == (f0, f1) and kept["names"] == wanted and key not in _no_reuse:
        changed = np.zeros(T, dtype=bool)
        for name, state in now.items():
            changed |= state.changed_since(kept["curves"][name])
        n = int(changed.sum())
        if n == 0:
            last_reuse, last_frames = "cached", 0
            return kept["samples"]
        if n <= 0.5 * T and _update_frames(kept, requests, f0, np.flatnonzero(changed), ~changed, scene, isolate):
            idx = np.flatnonzero(changed)
            for name, state in now.items():                   # frames left alone keep what they were sampled at
                kept["curves"][name] = kept["curves"][name].take(state, idx)
            last_reuse, last_frames = "partial", n
            return kept["samples"]
    samples = _sample_full(requests, f0, f1, scene, hide_other_meshes, isolate)
    _kept.pop(key, None)                                      # (newest last)
    _kept[key] = {"range": (f0, f1), "names": wanted, "curves": now, "samples": samples}
    sets = list(dict.fromkeys(k[1] for k in _kept))           # oldest first
    for old in sets[:-KEEP_SETS]:
        for k in [k for k in _kept if k[1] == old]:
            del _kept[k]
    last_reuse, last_frames = "full", T
    return samples


def _update_frames(kept: dict, requests, f0: int, idx: np.ndarray, same: np.ndarray, scene, isolate: bool) -> bool:
    """Evaluate the frames ``idx`` again and put them into the kept sample. False if a frame that should be as
    it was is not (a rig whose pose depends on other frames): then the whole take has to be sampled."""
    global _probe
    check = np.flatnonzero(same)
    probe = None
    if len(check):
        _probe += 1
        probe = int(check[(_probe * 7919) % len(check)])
        idx = np.append(idx, probe)
    entries, key = _entries(requests, len(idx))
    if isolate and key in _trusted and key not in _scene_bound:
        _sample_isolated(entries, f0, len(idx), scene, frames=[f0 + int(i) for i in idx])
    else:
        rows = _sample_scene(entries, f0, len(idx), scene, frames=[int(i) for i in idx], hide_other_meshes=len(idx) > 3)
        for k, i in enumerate(idx):
            for entry, (world, mats) in zip(entries, rows[int(i)]):
                entry[5][k] = world
                if mats is not None:
                    entry[4][k] = mats
    samples = kept["samples"]
    if probe is not None:
        for obj, _names, _idx, _buf, pose, ow in entries:
            smp = samples[obj.name]
            for new, old in ((ow[-1], smp.obj_world[probe]), (pose[-1], smp.pose[probe])):
                if new.size and np.abs(new - old).max() > 1e-4 * max(1.0, float(np.abs(new).max())):
                    _no_reuse.add(next(k for k, v in _kept.items() if v is kept))
                    print(f"MoCap Cleanup: the pose of {obj.name!r} depends on other frames; sampling the whole take")
                    return False
        idx = idx[:-1]
    n = len(idx)
    for obj, _names, _idx, _buf, pose, ow in entries:
        smp = samples[obj.name]
        smp.pose[idx] = pose[:n]
        smp.obj_world[idx] = ow[:n]
        smp._obj_world_inv = None
    return True


def _sample_full(requests, f0: int, f1: int, scene, hide_other_meshes: bool, isolate: bool) -> dict[str, RigSample]:
    global last_mode, _probe
    T = f1 - f0 + 1
    entries, key = _entries(requests, T)
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
            print(f"MoCap Cleanup: {sorted(key)} evaluate differently outside their scene; sampling the whole scene")
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


def sample_many_frames(obj, names: list[str], frames, scene=None) -> RigSample:
    """``sample_frames`` for many frames: ``obj`` is evaluated on its own in a temporary scene when
    ``sample_objects`` found that this matches the user's scene, else the scene is stepped."""
    scene = scene or bpy.context.scene
    frames = [int(f) for f in frames]
    if frozenset({obj.name}) not in _trusted or not frames:
        return sample_frames(obj, names, frames, scene)
    all_names = [pb.name for pb in obj.pose.bones]
    lookup = {n: i for i, n in enumerate(all_names)}
    names = [n for n in dict.fromkeys(names) if n in lookup]
    idx = np.array([lookup[n] for n in names], dtype=np.int64)
    T = len(frames)
    entry = (obj, names, idx, np.empty(len(all_names) * 16, dtype=np.float32),
             np.empty((T, len(names), 4, 4), dtype=np.float64), np.empty((T, 4, 4), dtype=np.float64))
    _sample_isolated([entry], 0, T, scene, frames=frames)
    return RigSample(obj.name, 0, T - 1, names, entry[4], entry[5])


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
