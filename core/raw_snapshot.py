"""Non-destructive bookkeeping: raw snapshot of the take and the contact-solver base layer.

* **Raw snapshot** - a hidden copy of the action made before the first edit. Used for the ghost
  curves in the Graph Editor and for "Restore Raw" (whole take, frame range or selected bones).
* **Solve base** - the pre-solve curves of the channels the contact / kinematics solver writes.
  Every re-solve starts from this base again, so pins can be moved, edited or deleted without
  baking errors on top of errors. A checksum of the solved channels tells whether the curves were
  edited by hand (or by another tool) since the last solve; in that case the edit becomes the new base.
"""

from __future__ import annotations

import json

import bpy
import numpy as np

from .action_io import (
    channels_checksum,
    ensure_channelbag,
    ensure_fcurve,
    fcurve_bone_name,
    get_action_frame_range,
    get_active_action,
    get_active_channelbag,
    sample_fcurve_range,
    write_fcurve_range,
)
from .math_quat import blend_window_weights

RAW_LINK_KEY = "mc_raw_snapshot"
RAW_FLAG_KEY = "mc_is_raw_snapshot"
BASE_LINK_KEY = "mc_solve_base"
BASE_FLAG_KEY = "mc_is_solve_base"
BASE_MISSING_KEY = "mc_missing_paths"
SOLVED_CRC_KEY = "mc_solved_crc"
HELPER_FLAGS = (RAW_FLAG_KEY, BASE_FLAG_KEY, "mc_is_residual")


def is_helper_action(action) -> bool:
    return action is not None and any(action.get(k) for k in HELPER_FLAGS)


LINK_KEYS = (RAW_LINK_KEY, BASE_LINK_KEY, "mc_residual")


def purge_orphans() -> int:
    """Remove helper actions (raw snapshot, solve base, residual detail) that no take links to any more,
    because their take was deleted or another tool replaced it. They carry a fake user, so Blender would keep
    them forever otherwise. Returns how many went."""
    linked = set()
    for a in bpy.data.actions:
        if not is_helper_action(a):
            linked |= {str(a.get(k)) for k in LINK_KEYS if a.get(k)}
    gone = 0
    for a in list(bpy.data.actions):
        if is_helper_action(a) and a.name not in linked and a.library is None and a.users - int(a.use_fake_user) == 0:
            bpy.data.actions.remove(a)
            gone += 1
    return gone


# ---------------------------------------------------------------------------
# F-Curve copy helper
# ---------------------------------------------------------------------------

_KEY_ARRAYS = (
    ("co", np.float32, 2),
    ("handle_left", np.float32, 2),
    ("handle_right", np.float32, 2),
    ("interpolation", np.int32, 1),
    ("handle_left_type", np.int32, 1),
    ("handle_right_type", np.int32, 1),
)


def copy_fcurve_keys(src, dst) -> None:
    """Make ``dst`` an exact copy of ``src``'s keyframes (values, handles, interpolation)."""
    n = len(src.keyframe_points)
    kp = dst.keyframe_points
    if len(kp) != n:
        kp.clear()
        if n:
            kp.add(n)
    for attr, dtype, width in _KEY_ARRAYS:
        if n == 0:
            break
        buf = np.empty(n * width, dtype=dtype)
        src.keyframe_points.foreach_get(attr, buf)
        kp.foreach_set(attr, buf)
    dst.extrapolation = src.extrapolation
    dst.update()


# ---------------------------------------------------------------------------
# Raw snapshot
# ---------------------------------------------------------------------------

def get_raw_snapshot(obj):
    act = get_active_action(obj)
    if act is None:
        return None
    if act.get(RAW_FLAG_KEY):
        return act
    name = act.get(RAW_LINK_KEY, "")
    snap = bpy.data.actions.get(name) if name else None
    if snap is not None and snap.get(RAW_FLAG_KEY):
        return snap
    return None


def ensure_raw_snapshot(obj, force_overwrite: bool = False):
    """Create the raw snapshot of the active action on first use and return it."""
    act = get_active_action(obj)
    if act is None or is_helper_action(act):
        return None
    existing = get_raw_snapshot(obj)
    if existing is not None and not force_overwrite:
        return existing
    if existing is not None:
        bpy.data.actions.remove(existing)
    purge_orphans()
    snap = act.copy()
    snap.name = f".{act.name}.raw"
    snap.use_fake_user = True
    snap[RAW_FLAG_KEY] = True
    for key in (RAW_LINK_KEY, BASE_LINK_KEY, SOLVED_CRC_KEY):
        if key in snap:
            del snap[key]
    act[RAW_LINK_KEY] = snap.name
    return snap


def delete_raw_snapshot(obj) -> bool:
    act = get_active_action(obj)
    snap = get_raw_snapshot(obj)
    if snap is None or snap == act:
        return False
    bpy.data.actions.remove(snap)
    if act is not None and RAW_LINK_KEY in act:
        del act[RAW_LINK_KEY]
    return True


def _reset_channel(obj, data_path: str, index: int) -> None:
    """Put a transform channel back to its rest value (location / Euler 0, quaternion identity, scale 1)."""
    try:
        prop = obj.path_resolve(data_path)
    except ValueError:
        return
    tail = data_path.rsplit(".", 1)[-1]
    if tail == "scale":
        value = 1.0
    elif tail == "rotation_quaternion":
        value = 1.0 if index == 0 else 0.0
    elif tail == "rotation_axis_angle":
        value = 1.0 if index == 2 else 0.0
    elif tail in ("location", "rotation_euler"):
        value = 0.0
    else:
        return
    try:
        prop[index] = value
    except (TypeError, IndexError):
        pass


def restore_region_to_raw(obj, frame_start: int | None = None, frame_end: int | None = None,
                          blend_factor: float = 1.0, falloff_frames: int = 4,
                          bone_names: list[str] | None = None) -> int:
    """Blend ``[frame_start, frame_end]`` back toward the raw snapshot with C2 falloffs.

    Returns the number of restored curves. Sparse curves are baked to one key per frame.
    """
    snap = get_raw_snapshot(obj)
    act = get_active_action(obj)
    if snap is None or act is None or snap == act:
        return 0
    cb_cur = ensure_channelbag(obj, act)
    cb_raw = get_active_channelbag(obj, snap)
    if cb_raw is None:
        return 0
    f0, f1 = get_action_frame_range(obj, snap)
    r0 = f0 if frame_start is None else max(f0, int(frame_start))
    r1 = f1 if frame_end is None else min(f1, int(frame_end))
    if r1 < r0:
        return 0
    T = f1 - f0 + 1
    w = blend_window_weights(T, r0 - f0, r1 - f0, falloff_frames, falloff_frames) * float(blend_factor)
    wanted = set(bone_names) if bone_names else None
    count = 0
    if frame_start is None and frame_end is None and wanted is None and blend_factor >= 1.0:
        # the whole take: curves the capture never had (e.g. the root shift of Feet on Floor) go as well
        raw_keys = {(fc.data_path, fc.array_index) for fc in cb_raw.fcurves}
        for fc in [fc for fc in cb_cur.fcurves if (fc.data_path, fc.array_index) not in raw_keys]:
            path, index = fc.data_path, fc.array_index
            cb_cur.fcurves.remove(fc)
            _reset_channel(obj, path, index)     # a channel without a curve keeps its last value otherwise
            count += 1
    for fc_raw in cb_raw.fcurves:
        if wanted is not None and fcurve_bone_name(fc_raw) not in wanted:
            continue
        raw_vals = sample_fcurve_range(fc_raw, f0, f1)
        fc_cur = cb_cur.fcurves.find(fc_raw.data_path, index=fc_raw.array_index)
        cur_vals = raw_vals if fc_cur is None else sample_fcurve_range(fc_cur, f0, f1)
        group = fc_raw.group.name if fc_raw.group else None
        write_fcurve_range(cb_cur, fc_raw.data_path, fc_raw.array_index, f0, (1.0 - w) * cur_vals + w * raw_vals,
                           group_name=group)
        count += 1
    return count


# ---------------------------------------------------------------------------
# Solver base layer
# ---------------------------------------------------------------------------

def get_solve_base(obj):
    act = get_active_action(obj)
    if act is None:
        return None
    name = act.get(BASE_LINK_KEY, "")
    base = bpy.data.actions.get(name) if name else None
    return base if (base is not None and base.get(BASE_FLAG_KEY)) else None


def solve_is_intact(obj, data_paths: list[str]) -> bool:
    """True when the solver channels still hold exactly what the last solve wrote."""
    act = get_active_action(obj)
    if act is None or get_solve_base(obj) is None:
        return False
    return act.get(SOLVED_CRC_KEY, "") == channels_checksum(obj, data_paths)


def store_solve_base(obj, data_paths: list[str]):
    """Snapshot the current curves of ``data_paths`` as the new solver base."""
    act = get_active_action(obj)
    if act is None:
        return None
    old = get_solve_base(obj)
    if old is not None:
        bpy.data.actions.remove(old)
    base = bpy.data.actions.new(f".{act.name}.base")
    base.use_fake_user = True
    base[BASE_FLAG_KEY] = True
    cb_base = ensure_channelbag(obj, base)
    cb_cur = get_active_channelbag(obj, act)
    wanted = set(data_paths)
    present = set()
    if cb_cur is not None:
        for fc in cb_cur.fcurves:
            if fc.data_path in wanted:
                copy_fcurve_keys(fc, ensure_fcurve(cb_base, fc.data_path, fc.array_index))
                present.add(fc.data_path)
    base[BASE_MISSING_KEY] = json.dumps(sorted(wanted - present))
    act[BASE_LINK_KEY] = base.name
    if SOLVED_CRC_KEY in act:
        del act[SOLVED_CRC_KEY]
    return base


def restore_solve_base(obj, data_paths: list[str]) -> bool:
    """Write the base curves back to the active action (removes curves the solver had to create)."""
    act = get_active_action(obj)
    base = get_solve_base(obj)
    if act is None or base is None:
        return False
    cb_cur = ensure_channelbag(obj, act)
    cb_base = get_active_channelbag(obj, base)
    wanted = set(data_paths)
    base_keys = set()
    if cb_base is not None:
        for fc in cb_base.fcurves:
            if fc.data_path not in wanted:
                continue
            base_keys.add((fc.data_path, fc.array_index))
            bone = fcurve_bone_name(fc)
            copy_fcurve_keys(fc, ensure_fcurve(cb_cur, fc.data_path, fc.array_index, group_name=bone))
    for fc in list(cb_cur.fcurves):
        if fc.data_path in wanted and (fc.data_path, fc.array_index) not in base_keys:
            cb_cur.fcurves.remove(fc)
    return True


def mark_solved(obj, data_paths: list[str]) -> None:
    act = get_active_action(obj)
    if act is not None:
        act[SOLVED_CRC_KEY] = channels_checksum(obj, data_paths)


def drop_solve_base(obj) -> None:
    """Forget the base (the current curves become plain animation again)."""
    act = get_active_action(obj)
    base = get_solve_base(obj)
    if base is not None:
        bpy.data.actions.remove(base)
    if act is not None:
        for key in (BASE_LINK_KEY, SOLVED_CRC_KEY):
            if key in act:
                del act[key]


def unbake_solve(obj, data_paths: list[str]) -> bool:
    """Prepare for an upstream edit (filter, drift, repair): put the pre-solve curves back.

    Returns True when a solve was active and has been unbaked (the caller should re-solve after
    its edit). If the solved curves were modified by hand, they are kept and become plain data.
    """
    if get_solve_base(obj) is None:
        return False
    if solve_is_intact(obj, data_paths):
        restore_solve_base(obj, data_paths)
        drop_solve_base(obj)
        return True
    drop_solve_base(obj)
    return False
