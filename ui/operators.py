"""Operators of the six cleanup stages."""

from __future__ import annotations

import bpy
import numpy as np
from bpy.props import BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty
from bpy.types import Operator

from .. import props
from ..core import action_io, floor_fit, raw_snapshot, runtime, sampler
from ..core.rig_inspector import solver_bone_names
from ..stages import solver
from ..stages import stage1_diagnostics as s1
from ..stages import stage2_filters as s2
from ..stages import stage3_drift as s3
from ..stages import stage4_contacts_pins as s4
from ..stages import stage6_simplifier as s6
from . import autosolve, common
from .common import MocapError


class _MocapOperator(Operator):
    """Base class: error handling and a poll that needs an armature."""

    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return common.get_rig(context) is not None

    def run(self, context):
        raise NotImplementedError

    def execute(self, context):
        try:
            msg = self.run(context)
        except Exception as exc:
            if isinstance(exc, (MocapError, solver.SolverError)) or type(exc) is RuntimeError:
                self.report({"ERROR"}, str(exc))               # something the user can act on
                return {"CANCELLED"}
            # Unexpected (a bug, or a rig this add-on does not understand). Curves may already have
            # been edited: finish the operator so that Undo covers it, and say where the details are.
            import traceback
            traceback.print_exc()
            self.report({"ERROR"}, f"{type(exc).__name__}: {exc}. Details are in the system console; "
                                   "Undo or Restore Raw brings the take back")
            try:
                scene = context.scene
                scene.frame_set(scene.frame_current)
            except Exception:
                pass
            common.tag_redraw()
            return {"FINISHED"}
        if msg:
            self.report({"INFO"}, msg)
            rig = common.get_rig(context)
            if rig is not None:
                common.set_result(rig, msg)
        common.tag_redraw()
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# General
# ---------------------------------------------------------------------------

class MOCAP_OT_use_active_rig(_MocapOperator):
    bl_idname = "mocap.use_active_rig"
    bl_label = "Use Active"
    bl_description = "Work on the active armature (for a character mesh or a skeleton under a control rig: the rig)"

    @classmethod
    def poll(cls, context):
        return common.resolve_rig(context.active_object) is not None

    def run(self, context):
        ob = context.active_object
        rig = common.resolve_rig(ob) or ob
        context.scene.mocap_cleanup.rig = rig
        if rig != ob:
            return f"Working on '{rig.name}', the control rig that drives '{ob.name}'"
        return f"Working on '{rig.name}'"


PRESETS = {
    "OPTICAL": dict(filter_method="MODIFIED_SINC", cutoff_hz=10.0, filter_adaptive=False, spike_sensitivity=6.5,
                    filter_max_shift=0.015, contact_speed=0.12,
                    pipe_repair=True, pipe_filter=True, pipe_dedrift=False, pipe_level=False, pipe_contacts=True,
                    pipe_solve=True, use_ballistic=False, use_knee_fix=True, use_pelvis=True),
    "IMU": dict(filter_method="WHITTAKER", cutoff_hz=8.0, filter_adaptive=False, spike_sensitivity=8.0,
                filter_max_shift=0.02, contact_speed=0.18,
                pipe_repair=True, pipe_filter=True, pipe_dedrift=True, pipe_level=True, pipe_contacts=True,
                pipe_solve=True, use_ballistic=False, use_knee_fix=True, use_pelvis=True, drift_smooth_hz=1.5),
    "VIDEO": dict(filter_method="WHITTAKER", cutoff_hz=5.0, filter_adaptive=False, spike_sensitivity=5.5,
                  filter_max_shift=0.04, contact_speed=0.25,
                  pipe_repair=True, pipe_filter=True, pipe_dedrift=True, pipe_level=True, pipe_contacts=True,
                  pipe_solve=True, use_ballistic=True, use_knee_fix=True, use_pelvis=True, drift_smooth_hz=1.0),
}


class MOCAP_OT_apply_preset(_MocapOperator):
    bl_idname = "mocap.apply_preset"
    bl_label = "Apply Source Preset"
    bl_description = "Tune detectors, filters and the one-click pipeline for the kind of capture system"

    @classmethod
    def poll(cls, context):
        return True

    def run(self, context):
        st = context.scene.mocap_cleanup
        with props.suspend_updates():
            for key, value in PRESETS[st.preset].items():
                setattr(st, key, value)
        return f"Preset '{st.bl_rna.properties['preset'].enum_items[st.preset].name}' applied"


# ---------------------------------------------------------------------------
# Stage 1
# ---------------------------------------------------------------------------

class MOCAP_OT_analyze(_MocapOperator):
    bl_idname = "mocap.analyze"
    bl_label = "Analyze"
    bl_description = "Scan the take for flips, spikes, dropouts, jitter, over-extension, sliding and floor penetration"

    def run(self, context):
        rig = common.require_rig(context)
        report = common.run_analyze(context, rig)
        return "Analysis: " + report.summary()


class MOCAP_OT_issue_jump(Operator):
    bl_idname = "mocap.issue_jump"
    bl_label = "Jump to Issue"
    bl_description = "Go to the next / previous marked problem"
    bl_options = {"REGISTER"}

    direction: IntProperty(default=1)

    @classmethod
    def poll(cls, context):
        rig = common.get_rig(context)
        return rig is not None and len(rig.mocap_cleanup.issues) > 0

    def execute(self, context):
        rig = common.get_rig(context)
        data = rig.mocap_cleanup
        cur = context.scene.frame_current
        starts = [(it.frame_start, i) for i, it in enumerate(data.issues)]
        active = data.active_issue
        if self.direction > 0:
            later = [(f, i) for f, i in starts if f > cur or (f == cur and i > active)]
            target = min(later)[1] if later else min(starts)[1]
        else:
            earlier = [(f, i) for f, i in starts if f < cur or (f == cur and i < active)]
            target = max(earlier)[1] if earlier else max(starts)[1]
        data.active_issue = target
        issue = data.issues[target]
        for area in context.screen.areas if context.screen else []:
            if area.type == "GRAPH_EDITOR":
                region = next((r for r in area.regions if r.type == "WINDOW"), None)
                try:
                    with context.temp_override(area=area, region=region):
                        bpy.ops.graph.view_frame()
                except Exception:
                    pass
        self.report({"INFO"}, f"Frames {issue.frame_start}-{issue.frame_end}: {issue.message}")
        return {"FINISHED"}


class MOCAP_OT_repair(_MocapOperator):
    bl_idname = "mocap.repair"
    bl_label = "Repair"
    bl_description = "Fix rotation flips, cut out spikes and dropouts and rebuild the motion underneath"

    def run(self, context):
        rig = common.require_rig(context)
        st = context.scene.mocap_cleanup
        fs, fe = common.frame_scope(context, rig)
        bones = common.bone_scope(context, rig)
        resolve = common.begin_edit(rig)
        report = common.fresh_report(context, rig)
        res = s1.repair(rig, report, flips=st.repair_flips, spikes=st.repair_spikes, frozen=st.repair_frozen,
                        frame_start=fs, frame_end=fe, bone_names=bones)
        common.end_edit(context, rig, resolve)
        common.run_analyze(context, rig)
        return (f"Repaired {res['flips']} flipped bones, {res['spike_frames']} spike frames, "
                f"{res['frozen_frames']} dropout frames")


class MOCAP_OT_fill_gap(_MocapOperator):
    bl_idname = "mocap.fill_gap"
    bl_label = "Rebuild Range"
    bl_description = ("Throw away a frame range and reconstruct it from the surrounding motion "
                      "(selected bones, or everything when 'Selected Bones Only' is off)")

    frame_start: IntProperty(name="Start")
    frame_end: IntProperty(name="End")

    def invoke(self, context, event):
        scene = context.scene
        if scene.use_preview_range:
            self.frame_start, self.frame_end = scene.frame_preview_start, scene.frame_preview_end
        else:
            self.frame_start = self.frame_end = scene.frame_current
        return context.window_manager.invoke_props_dialog(self)

    def run(self, context):
        rig = common.require_rig(context)
        if self.frame_end < self.frame_start:
            raise MocapError("The range is empty")
        bones = common.bone_scope(context, rig)
        resolve = common.begin_edit(rig)
        n = s1.fill_range(rig, self.frame_start, self.frame_end, bones)
        common.end_edit(context, rig, resolve)
        return f"Rebuilt frames {self.frame_start}-{self.frame_end} ({n} bone-frames)"


class MOCAP_OT_issue_fix(_MocapOperator):
    bl_idname = "mocap.issue_fix"
    bl_label = "Fix This Issue"
    bl_description = "Apply the matching tool to the active issue only"

    @classmethod
    def poll(cls, context):
        rig = common.get_rig(context)
        return rig is not None and 0 <= rig.mocap_cleanup.active_issue < len(rig.mocap_cleanup.issues)

    def run(self, context):
        rig = common.require_rig(context)
        st = context.scene.mocap_cleanup
        data = rig.mocap_cleanup
        issue = data.issues[data.active_issue]
        kind, a, b, bone, text = issue.kind, issue.frame_start, issue.frame_end, issue.bone, issue.message
        tool = s1.ISSUE_FIX.get(kind, "")
        bones = [bone] if bone else None
        if tool == "REPAIR":
            crc = action_io.action_checksum(rig)
            resolve = common.begin_edit(rig)
            report = common.fresh_report(context, rig)
            s1.repair(rig, report, frame_start=a, frame_end=b, bone_names=bones)
            common.end_edit(context, rig, resolve)
            if action_io.action_checksum(rig) == crc:
                common.run_analyze(context, rig)
                raise MocapError(f"Could not repair {text}: nothing there to rebuild (a lasting step in the take, "
                                 "not a one-frame glitch). Key it by hand or leave it")
            common.run_analyze(context, rig)
            still = any(i.kind == kind and i.bone == bone and i.frame_start <= b and i.frame_end >= a for i in data.issues)
            return (f"Repair changed the curves, but '{text}' is still flagged" if still
                    else f"Repaired {text}")
        elif tool == "FILTER":
            resolve = common.begin_edit(rig)
            s2.filter_action(rig, st.filter_method, st.cutoff_hz, bone_names=bones, frame_start=a, frame_end=b,
                             blend_frames=max(st.restore_blend, 3), euro_beta=st.euro_beta, unit=common.profile_of(rig).unit,
                             max_shift=st.filter_max_shift)
            common.end_edit(context, rig, resolve)
            msg = f"Filtered frames {a}-{b}" + (f" of {bone}" if bone else "")
        elif tool == "CONTACTS":
            common.require_rig(context, need_rigify=True)
            owner = common.owner_of_bone(common.profile_of(rig), bone)
            if not owner:
                raise MocapError("Cannot tell which limb this issue belongs to")
            with props.suspend_updates():
                if not any(p.owner == owner and p.lock_in <= b and p.lock_out >= a for p in data.pins):
                    pin = data.pins.add()
                    pin.from_data(s4.KinematicPin(owner=owner, mode="FLOOR", lock_in=a, lock_out=b,
                                                  blend_in=st.pin_blend_in, blend_out=st.pin_blend_out,
                                                  name=f"Plant {a}-{b}"))
                    data.active_pin = len(data.pins) - 1
            common.begin_edit(rig, upstream=False)
            res = common.run_solve(context, rig)
            msg = res.summary()
        elif tool == "SOLVE":
            common.require_rig(context, need_rigify=True)
            common.begin_edit(rig, upstream=False)
            res = common.run_solve(context, rig)
            msg = res.summary()
        else:
            raise MocapError("No automatic fix for this kind of issue")
        common.run_analyze(context, rig)
        return msg


# ---------------------------------------------------------------------------
# Stage 2
# ---------------------------------------------------------------------------

class MOCAP_OT_filter(_MocapOperator):
    bl_idname = "mocap.filter"
    bl_label = "Apply Filter"
    bl_description = "Low-pass the take (rotations are filtered on SO(3), never per channel)"

    def run(self, context):
        rig = common.require_rig(context)
        st = context.scene.mocap_cleanup
        fs, fe = common.frame_scope(context, rig)
        bones = common.bone_scope(context, rig)
        if not (st.filter_location or st.filter_rotation):
            raise MocapError("Enable Location and / or Rotation")
        resolve = common.begin_edit(rig)
        confidence = common.fresh_report(context, rig).confidence if st.filter_adaptive else None
        n = s2.filter_action(rig, st.filter_method, st.cutoff_hz, bone_names=bones, frame_start=fs, frame_end=fe,
                             blend_frames=st.restore_blend, filter_location=st.filter_location,
                             filter_rotation=st.filter_rotation, strength=st.filter_strength, confidence=confidence,
                             adaptive=st.filter_adaptive, euro_beta=st.euro_beta, unit=common.profile_of(rig).unit,
                             max_shift=st.filter_max_shift)
        common.end_edit(context, rig, resolve)
        label = st.bl_rna.properties["filter_method"].enum_items[st.filter_method].name
        return f"{label} filter at {st.cutoff_hz:.1f} Hz on {n} bones"


class MOCAP_OT_estimate_cutoff(_MocapOperator):
    bl_idname = "mocap.estimate_cutoff"
    bl_label = "Suggest Cutoff"
    bl_description = "Residual analysis (Winter): find the cutoff where the filter starts to remove motion instead of noise"

    def run(self, context):
        rig = common.require_rig(context)
        report = common.fresh_report(context, rig)
        if report.suggested_cutoff <= 0.0:
            raise MocapError("Not enough moving location channels to estimate a cutoff")
        context.scene.mocap_cleanup.cutoff_hz = round(report.suggested_cutoff, 1)
        return f"Suggested cutoff: {report.suggested_cutoff:.1f} Hz"


class MOCAP_OT_restore_raw(_MocapOperator):
    bl_idname = "mocap.restore_raw"
    bl_label = "Restore Raw"
    bl_description = "Blend the take (or the range / selected bones) back to the untouched capture"

    factor: FloatProperty(name="Amount", default=1.0, min=0.0, max=1.0, subtype="FACTOR")

    def run(self, context):
        rig = common.require_rig(context)
        st = context.scene.mocap_cleanup
        if raw_snapshot.get_raw_snapshot(rig) is None:
            raise MocapError("There is no raw snapshot yet (nothing has been changed)")
        if s6.has_residual(rig):
            s6.merge_residual(rig)
        fs, fe = common.frame_scope(context, rig)
        bones = common.bone_scope(context, rig)
        whole = fs is None and bones is None and self.factor >= 1.0
        resolve = solver.unbake_for_edit(rig)
        n = raw_snapshot.restore_region_to_raw(rig, fs, fe, self.factor, st.restore_blend, bones)
        if whole:
            # the take is the capture again: nothing left to restore, no solve on top of it
            raw_snapshot.drop_solve_base(rig)
            raw_snapshot.delete_raw_snapshot(rig)
            raw_snapshot.purge_orphans()
        common.end_edit(context, rig, resolve and not whole, sync=not whole)      # the whole take: exactly as it was
        return f"Restored {n} curves" + ("" if not whole else " (pins are kept; press Solve to apply them again)")


class MOCAP_OT_snapshot(_MocapOperator):
    bl_idname = "mocap.snapshot"
    bl_label = "Raw Snapshot"
    bl_description = "Manage the untouched copy of the take"

    action: EnumProperty(items=(("REFRESH", "Set Current as Raw", "The current animation becomes the new reference"),
                                ("DELETE", "Delete", "Remove the raw snapshot")))

    def run(self, context):
        rig = common.require_rig(context)
        runtime.GHOSTS.clear()
        if self.action == "REFRESH":
            if s6.has_residual(rig):
                raise MocapError("Merge the residual detail layer first")
            raw_snapshot.ensure_raw_snapshot(rig, force_overwrite=True)
            common.remember_chains(rig)
            return "The current animation is the new raw reference"
        return "Raw snapshot deleted" if raw_snapshot.delete_raw_snapshot(rig) else "There was no raw snapshot"


# ---------------------------------------------------------------------------
# Stage 3
# ---------------------------------------------------------------------------

def _floor_pins(rig):
    pins = [p for p in rig.mocap_cleanup.pin_data() if p.enabled and p.mode == "FLOOR" and p.owner.startswith("leg")]
    return pins or None


class MOCAP_OT_dedrift(_MocapOperator):
    bl_idname = "mocap.dedrift"
    bl_label = "Remove Drift"
    bl_description = ("Planted feet do not move: remove the slow sliding of the whole body "
                      "(inertial suits, video mocap). Uses the foot pins when there are any")

    def run(self, context):
        rig = common.require_rig(context, need_rigify=True)
        st = context.scene.mocap_cleanup
        resolve = common.begin_edit(rig)
        res = s3.remove_drift(rig, _floor_pins(rig), floor_z=common.floor_for(context, rig), floor_object=st.floor_object,
                              smooth_hz=st.drift_smooth_hz, strength=st.drift_strength)
        common.end_edit(context, rig, resolve)
        return "Drift: " + res.summary()


class MOCAP_OT_level_floor(_MocapOperator):
    bl_idname = "mocap.level_floor"
    bl_label = "Level Floor"
    bl_description = "Remove invisible stairs and ramps: every step lands on the floor again"

    def run(self, context):
        rig = common.require_rig(context, need_rigify=True)
        st = context.scene.mocap_cleanup
        resolve = common.begin_edit(rig)
        res = s3.level_floor(rig, None, floor_z=common.floor_for(context, rig), floor_object=st.floor_object,
                             smooth_hz=st.drift_smooth_hz, strength=st.drift_strength)
        common.end_edit(context, rig, resolve)
        return "Floor: " + res.summary()


class MOCAP_OT_floor_from_feet(_MocapOperator):
    bl_idname = "mocap.floor_from_feet"
    bl_label = "Floor from Feet"
    bl_description = "Set the floor height to where the character's feet rest in this take (the take is not moved)"

    def run(self, context):
        rig = common.require_rig(context, need_rigify=True)
        rest, k, _n = _feet_rest(context, rig)
        context.scene.mocap_cleanup.floor_z = rest - k
        rig.mocap_cleanup.analysis_stale = True
        return f"Floor height set to {rest - k:.4f}"


class MOCAP_OT_close_loop(_MocapOperator):
    bl_idname = "mocap.close_loop"
    bl_label = "Close Loop"
    bl_description = "Make the last frame match the first one (pose and velocity) for seamless cycles"

    def run(self, context):
        rig = common.require_rig(context)
        st = context.scene.mocap_cleanup
        resolve = common.begin_edit(rig)
        stats = s3.close_loop(rig, blend_frames=st.loop_blend, keep_travel=st.loop_keep_travel)
        common.end_edit(context, rig, resolve)
        return (f"Loop closed: seam was {stats['max_gap'] * 100:.1f} cm / {np.degrees(stats['max_rot_gap']):.1f} deg, "
                f"travel kept {stats['travel']:.2f} m")


# ---------------------------------------------------------------------------
# Stage 4
# ---------------------------------------------------------------------------

class MOCAP_OT_detect_contacts(_MocapOperator):
    bl_idname = "mocap.detect_contacts"
    bl_label = "Detect Contacts"
    bl_description = "Find foot plants (height, stillness of the contact point, settling) and create floor pins for them"

    def run(self, context):
        rig = common.require_rig(context, need_rigify=True)
        st = context.scene.mocap_cleanup
        had_solve = common.begin_edit(rig)             # detect on the motion without the previous solve
        n = detect_contacts(context, rig)
        msg = f"{n} contacts detected"
        if st.auto_solve or had_solve:
            res = common.run_solve(context, rig)
            msg += " - " + res.summary()
        common.end_edit(context, rig)
        return msg


def _feet_rest(context, rig):
    """Where the take's soles rest, ``(rig sole height, mesh offset below it, frames measured)``. Stores the offset."""
    profile = common.profile_of(rig)
    f0, f1 = action_io.get_action_frame_range(rig)
    smp = sampler.sample_objects([(rig, solver_bone_names(profile, rig.data.bones))], f0, f1)[rig.name]
    soles = s3._sole_points(profile, smp)
    if not soles:
        raise MocapError("The rig has no foot geometry")
    rest = s4.estimate_floor_height(soles, s2.scene_fps(context.scene), profile.unit)
    k, n = floor_fit.sole_offset(context.scene, rig, soles, f0, rest, profile.unit)
    rig.mocap_cleanup.sole_offset = k
    return rest, k, n


def put_on_floor(context, rig) -> float | None:
    """Feet on Floor: one vertical shift of the whole take so the resting mesh soles stand on the floor.
    Returns the shift, None if the take could not be moved."""
    st = context.scene.mocap_cleanup
    profile = common.profile_of(rig)
    rest, k, _n = _feet_rest(context, rig)
    dz = st.floor_z + k - rest if st.floor_auto else 0.0
    if abs(dz) <= 0.001 * profile.unit:
        return 0.0
    return dz if floor_fit.shift_take(rig, profile.root_ctrl, dz) else None


def settle_on_floor(context, rig) -> float:
    """After a solve: the solve flattens planted feet and lifts toes, which changes how far the shoe reaches
    below the rig's sole points. Measure where the mesh rests now and shift the take by the rest; the sole
    offset moves with it, so the solver's floor stays exactly where the soles are. Returns the shift."""
    st = context.scene.mocap_cleanup
    data = rig.mocap_cleanup
    profile = common.profile_of(rig)
    f0, f1 = action_io.get_action_frame_range(rig)
    smp = sampler.sample_objects([(rig, solver_bone_names(profile, rig.data.bones))], f0, f1)[rig.name]
    soles = s3._sole_points(profile, smp)
    if not soles:
        return 0.0
    rest = s4.estimate_floor_height(soles, s2.scene_fps(context.scene), profile.unit)
    m = floor_fit.mesh_rest(context.scene, rig, soles, f0, rest, profile.unit)
    if m is None:
        return 0.0
    e = m - st.floor_z
    if abs(e) <= 0.0005 * profile.unit or not floor_fit.shift_take(rig, profile.root_ctrl, -e):
        return 0.0
    data.sole_offset -= e
    return -e


def _floor_note(dz, k, unit) -> str:
    parts = []
    if dz is None:
        parts.append("the feet do not rest on the floor (no root control to move the take)")
    elif abs(dz) > 0.001 * unit:
        parts.append(f"take {'raised' if dz > 0 else 'lowered'} {abs(dz) * 100:.1f} cm onto the floor")
    if abs(k) > 0.002 * unit:
        parts.append(f"mesh sole {abs(k) * 100:.1f} cm {'below' if k > 0 else 'above'} the rig's")
    return ", ".join(parts)


def detect_contacts(context, rig) -> int:
    """Replace the detected pins of ``rig`` by a fresh detection (manual pins are kept). Returns their number."""
    st = context.scene.mocap_cleanup
    data = rig.mocap_cleanup
    profile = common.profile_of(rig)
    f0, f1 = action_io.get_action_frame_range(rig)
    fps = s2.scene_fps(context.scene)
    smp = sampler.sample_objects([(rig, solver_bone_names(profile, rig.data.bones))], f0, f1)[rig.name]
    points = s3._sole_points(profile, smp)
    # Feet on Floor: plants are found where the feet rest in the take (Clean Up Take has put them on the floor by
    # then); otherwise against the floor as set
    if points and st.floor_auto and st.floor_object is None:
        z = s4.estimate_floor_height(points, fps, profile.unit)
    else:
        z = common.floor_for(context, rig)
    floor = s4.Floor(z, st.floor_object, context.evaluated_depsgraph_get())
    if st.contact_hands:
        for arm in profile.arms.values():
            p = smp.head_w(arm.org_end).copy()
            p[:, 2] -= 0.5 * arm.end_radius
            points[arm.key] = p[:, None, :]
    if not points:
        raise MocapError("The rig has no foot geometry")
    pins, _ = s4.detect_contacts(points, floor, f0, fps, profile.unit, height_tol=st.contact_height,
                                 speed_tol=st.contact_speed, blend_in=st.pin_blend_in, blend_out=st.pin_blend_out,
                                 min_duration=st.contact_min_time, travel_tol=st.contact_travel)
    manual = [p.to_data() for p in data.pins if not p.auto]
    # a manual pin wins over a detected one on the same limb and frames
    pins = [p for p in pins if not any(m.owner == p.owner and m.lock_in <= p.lock_out and m.lock_out >= p.lock_in
                                      for m in manual)]
    with props.suspend_updates():
        data.pins.clear()
        for p in sorted(manual + pins, key=lambda q: (q.owner, q.lock_in)):
            data.pins.add().from_data(p)
        data.active_pin = 0
    return len(pins)


class MOCAP_OT_pin_add(_MocapOperator):
    bl_idname = "mocap.pin_add"
    bl_label = "Pin Here"
    bl_description = ("Pin the limb of the selected control from the current frame on: to the floor, in world space, "
                      "or to another object / bone (select the other object first, the rig last)")

    mode: EnumProperty(name="Mode", items=tuple((k, n, d) for k, n, d, _i, _v in props.MODE_ITEMS))
    owner: EnumProperty(name="Limb", items=(("AUTO", "From Selection", ""),) + props.OWNER_ITEMS, default="AUTO")

    def run(self, context):
        rig = common.require_rig(context, need_rigify=True)
        if s6.has_residual(rig):
            raise MocapError("The take has a residual detail layer. Merge it first (Keys stage)")
        st = context.scene.mocap_cleanup
        data = rig.mocap_cleanup
        profile = common.profile_of(rig)
        scene = context.scene
        owners = []
        if self.owner != "AUTO":
            owners = [self.owner]
        else:
            for name in action_io.selected_pose_bone_names(rig):
                o = common.owner_of_bone(profile, name)
                if o and o not in owners:
                    owners.append(o)
        if not owners:
            raise MocapError("Select a hand, foot or torso control of the rig (or choose the limb in the operator panel)")
        f0, f1 = action_io.get_action_frame_range(rig)
        if st.range_mode == "PREVIEW" and scene.use_preview_range:
            a, b = scene.frame_preview_start, scene.frame_preview_end
        else:
            a = scene.frame_current
            b = min(a + st.pin_length - 1, max(f1, a))
        target, target_bone = None, ""
        if self.mode == "TARGET":
            others = [o for o in context.selected_objects if o != rig]
            if not others:
                raise MocapError("Select the object to follow as well (the rig has to be the active object)")
            target = others[0]
            if target.type == "ARMATURE" and target.data.bones.active is not None:
                target_bone = target.data.bones.active.name
        with props.suspend_updates():
            for owner in owners:
                pin = data.pins.add()
                pin.from_data(s4.KinematicPin(
                    owner=owner, mode=self.mode, lock_in=a, lock_out=b, blend_in=st.pin_blend_in, blend_out=st.pin_blend_out,
                    reference_frame=-1 if self.mode == "FLOOR" else scene.frame_current,
                    name=f"{props.OWNER_LABELS[owner]} {self.mode.title()} {a}-{b}"))
                if target is not None:
                    pin.target_object = target
                    pin.target_bone = target_bone
                data.active_pin = len(data.pins) - 1
        msg = f"Pinned {', '.join(props.OWNER_LABELS[o] for o in owners)} on frames {a}-{b}"
        if st.auto_solve:
            common.begin_edit(rig, upstream=False)
            msg += " - " + common.run_solve(context, rig).summary()
        return msg


class MOCAP_OT_pin_remove(_MocapOperator):
    bl_idname = "mocap.pin_remove"
    bl_label = "Remove Pin"
    bl_description = "Remove the active pin"

    def run(self, context):
        rig = common.require_rig(context, need_action=False)
        data = rig.mocap_cleanup
        if not (0 <= data.active_pin < len(data.pins)):
            raise MocapError("No pin selected")
        with props.suspend_updates():
            data.pins.remove(data.active_pin)
            data.active_pin = max(0, min(data.active_pin, len(data.pins) - 1))
        autosolve.schedule(rig, only_if_solved=True)
        if bpy.app.background:
            autosolve.flush()
        return "Pin removed"


class MOCAP_OT_pin_clear(_MocapOperator):
    bl_idname = "mocap.pin_clear"
    bl_label = "Clear Pins"
    bl_description = "Remove pins"

    auto_only: BoolProperty(name="Detected Only", default=False)

    def run(self, context):
        rig = common.require_rig(context, need_action=False)
        data = rig.mocap_cleanup
        with props.suspend_updates():
            keep = [p.to_data() for p in data.pins if self.auto_only and not p.auto]
            n = len(data.pins) - len(keep)
            data.pins.clear()
            for p in keep:
                data.pins.add().from_data(p)
            data.active_pin = 0
        autosolve.schedule(rig, only_if_solved=True)
        if bpy.app.background:
            autosolve.flush()
        return f"{n} pins removed"


class MOCAP_OT_pin_edge(_MocapOperator):
    bl_idname = "mocap.pin_edge"
    bl_label = "Set Pin Frame"
    bl_description = "Set the start, end or reference frame of the active pin to the current frame"

    edge: EnumProperty(items=(("IN", "In", ""), ("OUT", "Out", ""), ("REF", "Reference", "")))

    def run(self, context):
        rig = common.require_rig(context, need_action=False)
        data = rig.mocap_cleanup
        if not (0 <= data.active_pin < len(data.pins)):
            raise MocapError("No pin selected")
        pin = data.pins[data.active_pin]
        f = context.scene.frame_current
        if self.edge == "IN":
            with props.suspend_updates():
                pin.lock_out = max(pin.lock_out, f)
            pin.lock_in = f
        elif self.edge == "OUT":
            with props.suspend_updates():
                pin.lock_in = min(pin.lock_in, f)
            pin.lock_out = f
        else:
            with props.suspend_updates():
                pin.use_reference = True
            pin.reference_frame = f
        if bpy.app.background:
            autosolve.flush()
        return ""


class MOCAP_OT_solve(_MocapOperator):
    bl_idname = "mocap.solve"
    bl_label = "Solve"
    bl_description = ("Apply pins, floor, knee / elbow fixes and physics to the control curves. Always starts from "
                      "the motion before the previous solve, so nothing accumulates")

    resample: BoolProperty(name="Resample Rig", default=False,
                           description="Evaluate the rig again even if nothing seems to have changed")

    def run(self, context):
        rig = common.require_rig(context, need_rigify=True)
        common.begin_edit(rig, upstream=False)
        res = common.run_solve(context, rig, use_cache=not self.resample)
        common.end_edit(context, rig)
        for w in res.warnings[:3]:
            self.report({"WARNING"}, w)
        return res.summary()


class MOCAP_OT_clear_solve(_MocapOperator):
    bl_idname = "mocap.clear_solve"
    bl_label = "Remove Solve"
    bl_description = "Take the pin / floor / knee corrections off the curves again (pins stay in the list)"

    def run(self, context):
        rig = common.require_rig(context)
        if s6.has_residual(rig):
            raise MocapError("Merge the residual detail layer first")
        done = solver.clear_solve(rig)
        common.end_edit(context, rig)
        return "Solve removed" if done else "There was no solve to remove"


# ---------------------------------------------------------------------------
# Stage 6
# ---------------------------------------------------------------------------

class MOCAP_OT_simplify(_MocapOperator):
    bl_idname = "mocap.simplify"
    bl_label = "Reduce Keys"
    bl_description = "Replace the dense curves by sparse Bezier keys within the tolerances"

    def run(self, context):
        rig = common.require_rig(context)
        st = context.scene.mocap_cleanup
        data = rig.mocap_cleanup
        bones = common.bone_scope(context, rig)
        forced: dict[str, list[int]] = {}
        if st.simplify_keep_contacts and common.profile_of(rig).is_supported:
            profile = common.profile_of(rig)
            for p in data.pin_data():
                if not p.enabled:
                    continue
                ctrl = profile.torso_ctrl if p.owner == "torso" else getattr(profile.limb_by_key(p.owner), "ik_ctrl", "")
                if ctrl:
                    forced.setdefault(ctrl, []).extend([p.start, p.lock_in, p.lock_out, p.end])
        wm = context.window_manager
        wm.progress_begin(0, 100)
        try:
            res = s6.simplify(rig, bone_names=bones, tol_loc=st.simplify_tol_loc, tol_rot_deg=st.simplify_tol_rot,
                              residual=st.simplify_residual, forced_frames=forced,
                              progress=lambda f: wm.progress_update(int(100 * f)))
        finally:
            wm.progress_end()
        runtime.clear_rig(rig.name)
        common.end_edit(context, rig)
        return res.summary()


class _ResidualOperator(_MocapOperator):
    """Only available while the take has a residual detail layer (greyed out otherwise, QA N3)."""

    @classmethod
    def poll(cls, context):
        rig = common.get_rig(context)
        if rig is None:
            return False
        if not s6.has_residual(rig):
            cls.poll_message_set("There is no residual detail layer")
            return False
        return True


class MOCAP_OT_merge_residual(_ResidualOperator):
    bl_idname = "mocap.merge_residual"
    bl_label = "Merge Detail Layer"
    bl_description = "Bake sparse keys + residual detail back into dense curves (needed before running other stages again)"

    def run(self, context):
        rig = common.require_rig(context)
        if not s6.has_residual(rig):
            raise MocapError("There is no residual detail layer")
        n = s6.merge_residual(rig)
        runtime.clear_rig(rig.name)
        common.end_edit(context, rig)
        return f"Merged the detail layer into {n} curves"


class MOCAP_OT_toggle_residual(_ResidualOperator):
    bl_idname = "mocap.toggle_residual"
    bl_label = "Toggle Detail"
    bl_description = "Show the animation with / without the residual mocap detail"

    def run(self, context):
        rig = common.require_rig(context)
        if not s6.has_residual(rig):
            raise MocapError("There is no residual detail layer")
        s6.set_residual_enabled(rig, not s6.residual_enabled(rig))
        context.scene.frame_set(context.scene.frame_current)
        return "Detail layer " + ("on" if s6.residual_enabled(rig) else "off")


# ---------------------------------------------------------------------------
# One click
# ---------------------------------------------------------------------------

class MOCAP_OT_run_all(_MocapOperator):
    bl_idname = "mocap.run_all"
    bl_label = "Clean Up Take"
    bl_description = "Run the enabled stages in the right order: repair, filter, drift, contacts, solve"

    def run(self, context):
        rig = common.require_rig(context)
        st = context.scene.mocap_cleanup
        rigify = common.profile_of(rig).is_supported
        steps = []
        common.begin_edit(rig)
        report = common.run_analyze(context, rig)
        before = report.joints
        if st.pipe_repair:
            # unattended: only what is beyond doubt. Doubtful spikes stay in the issue list.
            res = s1.repair(rig, report, flips=st.repair_flips, spikes=st.repair_spikes, frozen=st.repair_frozen,
                            unsure=False)
            steps.append(f"repair ({res['spike_frames'] + res['frozen_frames']} frames)")
        if st.pipe_filter:
            confidence = common.run_analyze(context, rig).confidence if st.filter_adaptive else None
            s2.filter_action(rig, st.filter_method, st.cutoff_hz, strength=st.filter_strength, confidence=confidence,
                             adaptive=st.filter_adaptive, filter_location=st.filter_location,
                             filter_rotation=st.filter_rotation, euro_beta=st.euro_beta, unit=common.profile_of(rig).unit,
                             max_shift=st.filter_max_shift)
            steps.append(f"filter {st.cutoff_hz:.0f} Hz")
        fit = None
        if rigify and st.floor_object is None:
            fit = put_on_floor(context, rig)
            steps.append("")                       # the floor note, filled in when the take has settled
        if rigify and st.pipe_level:
            r = s3.level_floor(rig, None, floor_z=common.floor_for(context, rig), floor_object=st.floor_object)
            steps.append(f"floor {r.max_vertical * 100:.0f} cm")
        if rigify and st.pipe_dedrift:
            r = s3.remove_drift(rig, None, floor_z=common.floor_for(context, rig), floor_object=st.floor_object,
                                smooth_hz=st.drift_smooth_hz)
            steps.append(f"drift {r.max_horizontal * 100:.0f} cm")
        if rigify and st.pipe_contacts:
            steps.append(f"{detect_contacts(context, rig)} contacts")
        if rigify and st.pipe_solve:
            res = common.run_solve(context, rig)
            steps.append("solve")
            if fit is not None and st.floor_auto and st.floor_object is None:
                fit += settle_on_floor(context, rig)
        if "" in steps:
            note = _floor_note(fit, rig.mocap_cleanup.sole_offset, common.profile_of(rig).unit)
            i = steps.index("")
            steps[i:i + 1] = [note] if note else []
        # the solve synced the chains; settling the take on the floor moves both of them alike
        common.end_edit(context, rig, sync=not (rigify and st.pipe_solve))
        after = common.run_analyze(context, rig)
        # say how far the cleanup moved the body: the one number that tells whether it is still the same take.
        # The floor shift moves every joint alike and is reported on its own, so it is not part of it.
        if fit:
            before = {k: v + np.array([0.0, 0.0, fit]) for k, v in before.items()}
        change = s1.largest_change(before, after.joints, after.f0)
        moved = f" Largest change: {change[0]} {change[1] * 100:.1f} cm at frame {change[2]}." if change and change[1] > 5e-4 else ""
        return "Done: " + ", ".join(steps) + "." + moved + " Left: " + rig.mocap_cleanup.analysis_summary


CLASSES = (
    MOCAP_OT_use_active_rig, MOCAP_OT_apply_preset, MOCAP_OT_analyze, MOCAP_OT_issue_jump, MOCAP_OT_repair,
    MOCAP_OT_fill_gap, MOCAP_OT_issue_fix, MOCAP_OT_filter, MOCAP_OT_estimate_cutoff, MOCAP_OT_restore_raw,
    MOCAP_OT_snapshot, MOCAP_OT_dedrift, MOCAP_OT_level_floor, MOCAP_OT_floor_from_feet, MOCAP_OT_close_loop,
    MOCAP_OT_detect_contacts, MOCAP_OT_pin_add, MOCAP_OT_pin_remove, MOCAP_OT_pin_clear, MOCAP_OT_pin_edge,
    MOCAP_OT_solve, MOCAP_OT_clear_solve, MOCAP_OT_simplify, MOCAP_OT_merge_residual, MOCAP_OT_toggle_residual,
    MOCAP_OT_run_all,
)
