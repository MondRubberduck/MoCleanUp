"""Deferred re-solve after pin / solver setting edits ("Live Update")."""

from __future__ import annotations

import bpy

from ..core import raw_snapshot

_pending: set[str] = set()
_busy = False


def schedule(rig, only_if_solved: bool = False) -> None:
    scene = bpy.context.scene
    st = getattr(scene, "mocap_cleanup", None)
    if st is None or not st.auto_solve or _busy or rig is None or rig.type != "ARMATURE":
        return
    if only_if_solved and raw_snapshot.get_solve_base(rig) is None:
        return
    _pending.add(rig.name)
    if not bpy.app.background and not bpy.app.timers.is_registered(_timer):
        bpy.app.timers.register(_timer, first_interval=0.03)


def flush() -> int:
    """Run pending solves now (the timer does this in interactive sessions). Returns the number of solves."""
    global _busy
    from . import common
    names = list(_pending)
    _pending.clear()
    done = 0
    _busy = True
    try:
        for name in names:
            rig = bpy.data.objects.get(name)
            if rig is None or rig.type != "ARMATURE":
                continue
            try:
                common.run_solve(bpy.context, rig)
                done += 1
            except Exception as exc:                      # never let a timer raise
                common.set_result(rig, f"Solve failed: {exc}")
    finally:
        _busy = False
    return done


def _timer():
    flush()
    return None


def unregister():
    _pending.clear()
    if bpy.app.timers.is_registered(_timer):
        bpy.app.timers.unregister(_timer)
