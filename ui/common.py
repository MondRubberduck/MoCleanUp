"""Shared operator / panel helpers: rig lookup, edit bookkeeping, analysis and solve entry points."""

from __future__ import annotations

import bpy

from .. import props
from ..core import action_io, raw_snapshot, runtime
from ..core.rig_inspector import inspect_rig
from ..stages import solver
from ..stages import stage1_diagnostics as s1
from ..stages import stage6_simplifier as s6
from ..stages.stage2_filters import scene_fps


class MocapError(RuntimeError):
    pass


def get_settings(context=None):
    return (context or bpy.context).scene.mocap_cleanup


def get_rig(context=None):
    """The rig to work on: the picked one, else the active armature (or the armature of the active mesh)."""
    context = context or bpy.context
    st = context.scene.mocap_cleanup
    rig = st.rig
    if rig is not None and rig.name in context.scene.objects and rig.type == "ARMATURE":
        return rig
    ob = getattr(context, "active_object", None) or context.view_layer.objects.active
    if ob is None:
        return None
    if ob.type == "ARMATURE":
        return ob
    if ob.parent is not None and ob.parent.type == "ARMATURE":
        return ob.parent
    return None


def profile_of(rig):
    key = (rig.name, len(rig.data.bones), tuple(round(v, 5) for v in rig.matrix_world.to_scale()))
    cached = runtime.PROFILES.get(rig.name)
    if cached is not None and cached[0] == key:
        return cached[1]
    profile = inspect_rig(rig)
    runtime.PROFILES[rig.name] = (key, profile)
    return profile


def require_rig(context, need_action: bool = True, need_rigify: bool = False):
    rig = get_rig(context)
    if rig is None:
        raise MocapError("Pick an armature (or make it the active object)")
    if need_action and action_io.get_active_action(rig) is None:
        raise MocapError(f"'{rig.name}' has no animation")
    if need_action and raw_snapshot.is_helper_action(action_io.get_active_action(rig)):
        raise MocapError("The assigned action is a MoCap Cleanup helper action (raw snapshot / layer)")
    if need_rigify and not profile_of(rig).is_supported:
        raise MocapError(profile_of(rig).message)
    return rig


def frame_scope(context, rig) -> tuple[int | None, int | None]:
    st = context.scene.mocap_cleanup
    scene = context.scene
    if st.range_mode == "PREVIEW" and scene.use_preview_range:
        return int(scene.frame_preview_start), int(scene.frame_preview_end)
    return None, None


def bone_scope(context, rig) -> list[str] | None:
    st = context.scene.mocap_cleanup
    if not st.only_selected:
        return None
    names = action_io.selected_pose_bone_names(rig)
    if not names:
        raise MocapError("'Selected Bones Only' is on, but no pose bone is selected")
    return names


def tag_redraw() -> None:
    props._tag_redraw()


def begin_edit(rig, upstream: bool = True) -> bool:
    """Call before changing curves. Returns True when the contact solve has to be redone afterwards."""
    if s6.has_residual(rig):
        raise MocapError("The take has a residual detail layer. Merge it first (Keys stage)")
    raw_snapshot.ensure_raw_snapshot(rig)
    return solver.unbake_for_edit(rig) if upstream else False


def end_edit(context, rig, resolve: bool = False) -> None:
    data = rig.mocap_cleanup
    data.analysis_stale = True
    if resolve:
        run_solve(context, rig)
    rig.update_tag()
    scene = context.scene
    scene.frame_set(scene.frame_current)
    tag_redraw()


def run_solve(context, rig, use_cache: bool = True):
    if s6.has_residual(rig):
        raise MocapError("The take has a residual detail layer. Merge it first (Keys stage)")
    st = context.scene.mocap_cleanup
    data = rig.mocap_cleanup
    res = solver.solve(rig, data.pin_data(), props.solve_settings(st, scene_fps(context.scene)), use_cache=use_cache)
    data.last_result = res.summary()
    data.analysis_stale = True
    tag_redraw()
    return res


def sync_floor(context, estimate) -> float:
    """With 'Floor from Feet' on (and no terrain object) the floor height follows the take."""
    st = context.scene.mocap_cleanup
    if st.floor_auto and st.floor_object is None and estimate is not None and abs(st.floor_z - estimate) > 1e-7:
        with props.suspend_updates():
            st.floor_z = float(estimate)
    return st.floor_z


def run_analyze(context, rig):
    st = context.scene.mocap_cleanup
    # After a solve the limbs sit inside the soft limit on purpose: only mark what exceeds it.
    hyperext = st.hyperext_ratio
    if raw_snapshot.get_solve_base(rig) is not None and st.use_knee_fix:
        hyperext = max(hyperext, st.max_extension + 0.0005)
    # The floor follows the feet of the take as it was captured. Once a solve has put the feet on a
    # floor, that floor stays: estimating it again from corrected feet would push it up every time.
    follow = st.floor_auto and raw_snapshot.get_solve_base(rig) is None
    solved = raw_snapshot.get_solve_base(rig) is not None
    report = s1.analyze(rig, floor_z=st.floor_z, floor_object=st.floor_object, spike_z=st.spike_sensitivity,
                        jitter_z=st.jitter_sensitivity, hyperext=hyperext, auto_floor=follow,
                        pins=rig.mocap_cleanup.pin_data() if solved else None)
    if follow:
        sync_floor(context, report.floor_estimate)
    runtime.REPORTS[rig.name] = report
    data = rig.mocap_cleanup
    with props.suspend_updates():
        data.issues.clear()
        for it in report.issues:
            item = data.issues.add()
            item.name = it.label
            item.kind, item.frame_start, item.frame_end = it.kind, it.frame_start, it.frame_end
            item.bone, item.severity, item.message = it.bone, it.severity, it.message
        data.active_issue = min(data.active_issue, max(len(data.issues) - 1, 0))
    data.analysis_summary = report.summary()
    data.analysis_stale = False
    data.suggested_cutoff = report.suggested_cutoff
    data.has_floor_estimate = report.floor_estimate is not None
    data.floor_estimate = report.floor_estimate or 0.0
    tag_redraw()
    return report


def fresh_report(context, rig):
    """The cached report if it still matches the curves, else a new analysis."""
    report = runtime.REPORTS.get(rig.name)
    if report is None or report.checksum != action_io.action_checksum(rig):
        report = run_analyze(context, rig)
    return report


def select_only_bone(rig, bone_name: str) -> bool:
    pb = rig.pose.bones.get(bone_name)
    if pb is None:
        return False
    for other in rig.pose.bones:
        if hasattr(other, "select"):
            other.select = other == pb
        else:
            other.bone.select = other == pb
    rig.data.bones.active = pb.bone
    return True


def focus_issue(context, rig) -> None:
    data = rig.mocap_cleanup
    if _focus_guard or not (0 <= data.active_issue < len(data.issues)):
        return
    issue = data.issues[data.active_issue]
    scene = context.scene
    if scene.frame_current != issue.frame_start:
        scene.frame_current = issue.frame_start
    if issue.bone and rig.mode == "POSE":
        select_only_bone(rig, issue.bone)
    tag_redraw()


_focus_guard = False


def owner_of_bone(profile, bone_name: str) -> str:
    """Pin owner key for a pose bone name ('' when it does not belong to a limb or the torso)."""
    limb = profile.limb_for_ctrl(bone_name)
    if limb is not None:
        return limb.key
    for limb in profile.limbs():
        if bone_name in (limb.pole_ctrl, limb.ik_base_ctrl, limb.toe_ctrl, limb.heel_ctrl, limb.org_upper, limb.org_lower,
                         limb.org_toe, limb.switch_bone):
            return limb.key
    low = bone_name.lower()
    side = "L" if low.endswith((".l", "_l")) else ("R" if low.endswith((".r", "_r")) else "")
    if side:
        if any(k in low for k in ("foot", "toe", "shin", "thigh", "heel", "leg")):
            return "leg." + side
        if any(k in low for k in ("hand", "forearm", "upper_arm", "finger", "thumb", "palm", "arm", "f_")):
            return "arm." + side
    if any(k in low for k in ("torso", "hips", "spine", "chest", "pelvis")):
        return "torso"
    return ""
