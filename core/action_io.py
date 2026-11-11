"""Slotted Action / Channelbag F-Curve I/O with NumPy buffers (Blender 5.x).

``Action -> layers[0] -> strips[0] -> channelbag(slot) -> fcurves``. Everything the add-on reads
or writes goes through this module, so the rest of the code never touches the Action API directly.
"""

from __future__ import annotations

import re
import zlib

import bpy
import numpy as np
from mathutils import Euler, Quaternion

POSE_BONE_PATH_RE = re.compile(r'^pose\.bones\["((?:[^"\\]|\\.)+)"\]\.(.+)$')
POSE_BONE_PROP_RE = re.compile(r'^pose\.bones\["((?:[^"\\]|\\.)+)"\]\["((?:[^"\\]|\\.)+)"\]$')


# ---------------------------------------------------------------------------
# Channelbag access
# ---------------------------------------------------------------------------

def _find_slot(obj, action):
    anim = getattr(obj, "animation_data", None)
    if anim is not None and anim.action == action and anim.action_slot is not None:
        return anim.action_slot
    if not action.slots:
        return None
    slot = action.slots.get("OB" + obj.name)
    if slot is not None:
        return slot
    for s in action.slots:
        if s.target_id_type in {"OBJECT", "UNSPECIFIED"}:
            return s
    return action.slots[0]


def get_active_action(obj):
    anim = getattr(obj, "animation_data", None)
    return anim.action if anim is not None else None


def get_active_channelbag(obj, action=None):
    """Channelbag of ``action`` (default: the object's assigned action) or None."""
    if action is None:
        action = get_active_action(obj)
        if action is None:
            return None
    slot = _find_slot(obj, action)
    if slot is None or not action.layers or not action.layers[0].strips:
        return None
    return action.layers[0].strips[0].channelbag(slot, ensure=False)


def ensure_channelbag(obj, action=None):
    """Return a channelbag for ``obj`` in ``action``, creating action / layer / strip / slot as needed."""
    if action is None:
        if obj.animation_data is None:
            obj.animation_data_create()
        anim = obj.animation_data
        if anim.action is None:
            anim.action = bpy.data.actions.new(name=f"{obj.name}Action")
        action = anim.action
    layer = action.layers[0] if action.layers else action.layers.new(name="Layer")
    strip = layer.strips[0] if layer.strips else layer.strips.new(type="KEYFRAME")
    slot = _find_slot(obj, action)
    if slot is None:
        slot = action.slots.new(id_type="OBJECT", name=obj.name)
    anim = obj.animation_data
    if anim is not None and anim.action == action and anim.action_slot is None:
        try:
            anim.action_slot = slot
        except Exception:
            pass
    return strip.channelbag(slot, ensure=True)


def iter_fcurves(obj, action=None):
    cb = get_active_channelbag(obj, action)
    if cb is not None:
        yield from cb.fcurves


def get_action_frame_range(obj, action=None) -> tuple[int, int]:
    """Integer ``(first, last)`` keyed frame of the action (scene range when there are no keys)."""
    if action is None:
        action = get_active_action(obj)
    if action is not None:
        r = action.curve_frame_range
        f0, f1 = int(round(r[0])), int(round(r[1]))
        if f1 > f0:
            return f0, f1
    scene = bpy.context.scene
    return int(scene.frame_start), int(scene.frame_end)


def bone_data_path(bone_name: str, prop: str) -> str:
    return f'pose.bones["{bpy.utils.escape_identifier(bone_name)}"].{prop}'


def bone_prop_data_path(bone_name: str, prop_name: str) -> str:
    return f'pose.bones["{bpy.utils.escape_identifier(bone_name)}"]["{bpy.utils.escape_identifier(prop_name)}"]'


def find_fcurve(channelbag, data_path: str, array_index: int = 0):
    if channelbag is None:
        return None
    return channelbag.fcurves.find(data_path, index=array_index)


def ensure_fcurve(channelbag, data_path: str, array_index: int = 0, group_name: str | None = None):
    fc = channelbag.fcurves.find(data_path, index=array_index)
    if fc is None:
        fc = channelbag.fcurves.new(data_path, index=array_index, group_name=group_name or "")
    return fc


# ---------------------------------------------------------------------------
# Raw key buffers
# ---------------------------------------------------------------------------

def read_fcurve_raw_points(fcurve) -> tuple[np.ndarray, np.ndarray]:
    n = len(fcurve.keyframe_points)
    if n == 0:
        return np.empty(0), np.empty(0)
    buf = np.empty(n * 2, dtype=np.float32)
    fcurve.keyframe_points.foreach_get("co", buf)
    pts = buf.reshape(n, 2).astype(np.float64)
    return pts[:, 0], pts[:, 1]


def sample_fcurve_range(fcurve, frame_start: int, frame_end: int, default_val: float = 0.0) -> np.ndarray:
    """Value at every integer frame in ``[frame_start, frame_end]`` (fast path for baked curves)."""
    T = frame_end - frame_start + 1
    if fcurve is None or len(fcurve.keyframe_points) == 0:
        return np.full(T, default_val, dtype=np.float64)
    kf, kv = read_fcurve_raw_points(fcurve)
    if len(kf) >= T:
        i0 = int(np.searchsorted(kf, frame_start - 1e-3))
        if i0 + T <= len(kf) and abs(kf[i0] - frame_start) < 1e-3 and abs(kf[i0 + T - 1] - frame_end) < 1e-3:
            if np.allclose(kf[i0 : i0 + T], np.arange(frame_start, frame_end + 1), atol=1e-3):
                return kv[i0 : i0 + T].copy()
    ev = fcurve.evaluate
    return np.fromiter((ev(float(f)) for f in range(frame_start, frame_end + 1)), dtype=np.float64, count=T)


_ENUM_CACHE: dict[tuple[str, str], int] = {}


def _enum_value(prop: str, identifier: str) -> int:
    key = (prop, identifier)
    if key not in _ENUM_CACHE:
        _ENUM_CACHE[key] = bpy.types.Keyframe.bl_rna.properties[prop].enum_items[identifier].value
    return _ENUM_CACHE[key]


def write_fcurve_range(channelbag, data_path: str, array_index: int, frame_start: int, values: np.ndarray,
                       group_name: str | None = None, interpolation: str = "LINEAR"):
    """Write one key per frame for ``[frame_start, frame_start + len(values) - 1]``.

    Keys outside that range are kept (with their interpolation); keys inside are replaced.
    """
    values = np.asarray(values, dtype=np.float64)
    T = len(values)
    if T == 0:
        return None
    frame_end = frame_start + T - 1
    fc = ensure_fcurve(channelbag, data_path, array_index, group_name=group_name)
    kp = fc.keyframe_points
    n_old = len(kp)
    new_f = np.arange(frame_start, frame_end + 1, dtype=np.float64)
    interp_val = _enum_value("interpolation", interpolation)
    interp = np.full(T, interp_val, dtype=np.int32)
    all_f, all_v = new_f, values

    if n_old:
        old_f, old_v = read_fcurve_raw_points(fc)
        before = old_f < frame_start - 1e-3
        after = old_f > frame_end + 1e-3
        if before.any() or after.any():
            old_i = np.empty(n_old, dtype=np.int32)
            kp.foreach_get("interpolation", old_i)
            all_f = np.concatenate([old_f[before], new_f, old_f[after]])
            all_v = np.concatenate([old_v[before], values, old_v[after]])
            interp = np.concatenate([old_i[before], interp, old_i[after]])

    total = len(all_f)
    if n_old != total:
        kp.clear()
        kp.add(total)
    co = np.empty((total, 2), dtype=np.float32)
    co[:, 0] = all_f
    co[:, 1] = all_v
    kp.foreach_set("co", co.ravel())
    kp.foreach_set("interpolation", interp)
    fc.update()
    return fc


def write_fcurve_sparse(channelbag, data_path: str, array_index: int, frames, values,
                        handle_left=None, handle_right=None, group_name: str | None = None):
    """Replace the curve with sparse Bezier keys. Handles ``(n, 2)`` are optional (aligned when given)."""
    frames = np.asarray(frames, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    n = len(frames)
    fc = ensure_fcurve(channelbag, data_path, array_index, group_name=group_name)
    kp = fc.keyframe_points
    kp.clear()
    if n == 0:
        return fc
    kp.add(n)
    co = np.empty((n, 2), dtype=np.float32)
    co[:, 0] = frames
    co[:, 1] = values
    kp.foreach_set("co", co.ravel())
    kp.foreach_set("interpolation", np.full(n, _enum_value("interpolation", "BEZIER"), dtype=np.int32))
    if handle_left is not None and handle_right is not None:
        ht = np.full(n, _enum_value("handle_left_type", "ALIGNED"), dtype=np.int32)
        kp.foreach_set("handle_left_type", ht)
        kp.foreach_set("handle_right_type", ht)
        kp.foreach_set("handle_left", np.asarray(handle_left, dtype=np.float32).ravel())
        kp.foreach_set("handle_right", np.asarray(handle_right, dtype=np.float32).ravel())
    else:
        ht = np.full(n, _enum_value("handle_left_type", "AUTO_CLAMPED"), dtype=np.int32)
        kp.foreach_set("handle_left_type", ht)
        kp.foreach_set("handle_right_type", ht)
    fc.update()
    return fc


# ---------------------------------------------------------------------------
# Bone channels
# ---------------------------------------------------------------------------

def read_bone_local_channels(obj, bone_name: str, frame_start: int, frame_end: int, action=None) -> dict:
    """Local channels of a pose bone on every frame.

    Returns ``loc (T, 3)``, ``quat (T, 4)`` (unit, raw sign), ``raw_rot`` (stored rotation channels),
    ``scale (T, 3)``, ``rotation_mode`` and ``has_loc`` / ``has_rot`` / ``has_scale`` flags.
    """
    pb = obj.pose.bones.get(bone_name)
    if pb is None:
        raise KeyError(f"PoseBone '{bone_name}' not found on '{obj.name}'")
    cb = get_active_channelbag(obj, action)
    T = frame_end - frame_start + 1
    rot_mode = pb.rotation_mode

    def read_vec(prop, defaults):
        path = bone_data_path(bone_name, prop)
        out = np.empty((T, len(defaults)), dtype=np.float64)
        found = False
        for i, dv in enumerate(defaults):
            fc = find_fcurve(cb, path, i)
            if fc is not None and len(fc.keyframe_points) > 0:
                found = True
            out[:, i] = sample_fcurve_range(fc, frame_start, frame_end, default_val=float(dv))
        return out, found

    loc, has_loc = read_vec("location", tuple(pb.location))
    scale, has_scale = read_vec("scale", tuple(pb.scale))

    if rot_mode == "QUATERNION":
        raw_rot, has_rot = read_vec("rotation_quaternion", tuple(pb.rotation_quaternion))
        norms = np.linalg.norm(raw_rot, axis=1, keepdims=True)
        quat = raw_rot / np.where(norms < 1e-12, 1.0, norms)
        quat[norms[:, 0] < 1e-12] = (1.0, 0.0, 0.0, 0.0)
    elif rot_mode == "AXIS_ANGLE":
        raw_rot, has_rot = read_vec("rotation_axis_angle", tuple(pb.rotation_axis_angle))
        quat = np.empty((T, 4), dtype=np.float64)
        for t in range(T):
            q = Quaternion(raw_rot[t, 1:4], raw_rot[t, 0])
            quat[t] = (q.w, q.x, q.y, q.z)
    else:
        raw_rot, has_rot = read_vec("rotation_euler", tuple(pb.rotation_euler))
        quat = np.empty((T, 4), dtype=np.float64)
        for t in range(T):
            q = Euler(raw_rot[t], rot_mode).to_quaternion()
            quat[t] = (q.w, q.x, q.y, q.z)
        # Euler -> quaternion has no sign memory; make the derived trajectory continuous.
        dots = np.sum(quat[:-1] * quat[1:], axis=1)
        sign = np.where(np.cumsum(np.concatenate([[False], dots < 0.0])) % 2 == 1, -1.0, 1.0)
        quat *= sign[:, None]

    return {
        "loc": loc,
        "quat": quat,
        "raw_rot": raw_rot,
        "scale": scale,
        "rotation_mode": rot_mode,
        "has_loc": has_loc,
        "has_rot": has_rot,
        "has_scale": has_scale,
    }


def write_bone_local_channels(obj, bone_name: str, frame_start: int, loc=None, quat=None, action=None,
                              interpolation: str = "LINEAR", enforce_continuity: bool = True):
    """Write dense location ``(T, 3)`` and / or rotation ``(T, 4)`` curves in the bone's rotation mode."""
    pb = obj.pose.bones.get(bone_name)
    if pb is None:
        raise KeyError(f"PoseBone '{bone_name}' not found on '{obj.name}'")
    cb = ensure_channelbag(obj, action)
    rot_mode = pb.rotation_mode

    if loc is not None:
        loc = np.asarray(loc, dtype=np.float64)
        path = bone_data_path(bone_name, "location")
        for i in range(3):
            write_fcurve_range(cb, path, i, frame_start, loc[:, i], group_name=bone_name, interpolation=interpolation)

    if quat is None:
        return
    quat = np.array(quat, dtype=np.float64, copy=True)
    T = len(quat)
    if enforce_continuity and T > 1:
        dots = np.sum(quat[:-1] * quat[1:], axis=1)
        sign = np.where(np.cumsum(np.concatenate([[False], dots < 0.0])) % 2 == 1, -1.0, 1.0)
        quat *= sign[:, None]

    if rot_mode == "QUATERNION":
        path = bone_data_path(bone_name, "rotation_quaternion")
        for i in range(4):
            write_fcurve_range(cb, path, i, frame_start, quat[:, i], group_name=bone_name, interpolation=interpolation)
    elif rot_mode == "AXIS_ANGLE":
        path = bone_data_path(bone_name, "rotation_axis_angle")
        aa = np.empty((T, 4), dtype=np.float64)
        for t in range(T):
            axis, angle = Quaternion(quat[t]).to_axis_angle()
            aa[t] = (angle, axis.x, axis.y, axis.z)
        for i in range(4):
            write_fcurve_range(cb, path, i, frame_start, aa[:, i], group_name=bone_name, interpolation=interpolation)
    else:
        path = bone_data_path(bone_name, "rotation_euler")
        eul = np.empty((T, 3), dtype=np.float64)
        prev = None
        for t in range(T):
            q = Quaternion(quat[t])
            e = q.to_euler(rot_mode, prev) if prev is not None else q.to_euler(rot_mode)
            prev = e
            eul[t] = (e.x, e.y, e.z)
        for i in range(3):
            write_fcurve_range(cb, path, i, frame_start, eul[:, i], group_name=bone_name, interpolation=interpolation)


def read_bone_prop_curve(obj, bone_name: str, prop_name: str, frame_start: int, frame_end: int, action=None) -> np.ndarray:
    """Per-frame value of a custom property such as ``pose.bones["thigh_parent.L"]["IK_FK"]``."""
    pb = obj.pose.bones.get(bone_name)
    default = 0.0
    if pb is not None and prop_name in pb.keys():
        try:
            default = float(pb[prop_name])
        except (TypeError, ValueError):
            default = 0.0
    fc = find_fcurve(get_active_channelbag(obj, action), bone_prop_data_path(bone_name, prop_name), 0)
    return sample_fcurve_range(fc, frame_start, frame_end, default_val=default)


def write_bone_prop_curve(obj, bone_name: str, prop_name: str, frame_start: int, values, action=None):
    cb = ensure_channelbag(obj, action)
    return write_fcurve_range(cb, bone_prop_data_path(bone_name, prop_name), 0, frame_start, values, group_name=bone_name)


def get_animated_bone_names(obj, action=None, transforms_only: bool = True) -> list[str]:
    """Pose bones that own transform F-Curves in the action, in armature order."""
    names = set()
    for fc in iter_fcurves(obj, action):
        m = POSE_BONE_PATH_RE.match(fc.data_path)
        if not m:
            continue
        if transforms_only and m.group(2) not in {
            "location", "rotation_quaternion", "rotation_euler", "rotation_axis_angle", "scale"}:
            continue
        names.add(bpy.utils.unescape_identifier(m.group(1)))
    return [pb.name for pb in obj.pose.bones if pb.name in names]


def selected_pose_bone_names(obj) -> list[str]:
    out = []
    for pb in obj.pose.bones:
        sel = pb.select if hasattr(pb, "select") else pb.bone.select
        if sel:
            out.append(pb.name)
    return out


def fcurve_bone_name(fcurve) -> str | None:
    m = POSE_BONE_PATH_RE.match(fcurve.data_path) or POSE_BONE_PROP_RE.match(fcurve.data_path)
    return bpy.utils.unescape_identifier(m.group(1)) if m else None


def channels_checksum(obj, data_paths: list[str], action=None) -> str:
    """CRC of the key buffers of all curves whose data path is in ``data_paths`` (order independent)."""
    wanted = set(data_paths)
    parts = []
    for fc in iter_fcurves(obj, action):
        if fc.data_path not in wanted:
            continue
        n = len(fc.keyframe_points)
        buf = np.empty(n * 2, dtype=np.float32)
        if n:
            fc.keyframe_points.foreach_get("co", buf)
        parts.append((fc.data_path, fc.array_index, zlib.crc32(buf.tobytes())))
    parts.sort()
    return format(zlib.crc32(repr(parts).encode()), "08x") + f":{len(parts)}"


def action_checksum(obj, action=None) -> str:
    """CRC over every F-Curve key buffer of the action slot used by ``obj``."""
    parts = []
    for fc in iter_fcurves(obj, action):
        n = len(fc.keyframe_points)
        buf = np.empty(n * 2, dtype=np.float32)
        if n:
            fc.keyframe_points.foreach_get("co", buf)
        parts.append((fc.data_path, fc.array_index, zlib.crc32(buf.tobytes())))
    parts.sort()
    return format(zlib.crc32(repr(parts).encode()), "08x") + f":{len(parts)}"


def rotation_data_path(obj, bone_name: str) -> str:
    mode = obj.pose.bones[bone_name].rotation_mode
    prop = {"QUATERNION": "rotation_quaternion", "AXIS_ANGLE": "rotation_axis_angle"}.get(mode, "rotation_euler")
    return bone_data_path(bone_name, prop)
