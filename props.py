"""Add-on properties: scene settings, per-rig pins and the issue list."""

from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    EnumProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import PropertyGroup

from .core import runtime
from .stages.solver import SolveSettings
from .stages.stage4_contacts_pins import KinematicPin

OWNER_ITEMS = (
    ("leg.L", "Foot L", "Left foot"),
    ("leg.R", "Foot R", "Right foot"),
    ("arm.L", "Hand L", "Left hand"),
    ("arm.R", "Hand R", "Right hand"),
    ("torso", "Hips", "Torso control (whole upper body)"),
)
OWNER_LABELS = {k: v for k, v, _ in OWNER_ITEMS}
MODE_ITEMS = (
    ("FLOOR", "Floor", "Plant on the floor: the contact point stays put, heel and toe may roll", "IMPORT", 0),
    ("WORLD", "World", "Hold the position (and rotation) in world space", "WORLD", 1),
    ("TARGET", "Target", "Follow another object or a bone of another character", "CON_CHILDOF", 2),
)
STAGE_ITEMS = (
    ("DIAGNOSE", "Diagnose", "1. Find flips, spikes, dropouts, jitter, sliding feet and popping knees", "VIEWZOOM", 0),
    ("FILTER", "Filter", "2. Remove jitter without touching clean motion", "SMOOTHCURVE", 1),
    ("DRIFT", "Drift", "3. Global trajectory: drift, floor level, loops", "TRACKING", 2),
    ("CONTACTS", "Pins", "4. Plant feet, pin hands to the world or to another character, keep everything above the floor", "PINNED", 3),
    ("PHYSICS", "Body", "5. Knee and elbow pops, soft IK, jump arcs, hands inside the body", "ARMATURE_DATA", 4),
    ("SIMPLIFY", "Keys", "6. Reduce the take to editable keys, keep the detail on a layer", "KEYFRAME", 5),
)
FILTER_ITEMS = (
    ("WHITTAKER", "Whittaker", "Penalized least squares (Eilers 2003): no edge artifacts, supports confidence weights"),
    ("MODIFIED_SINC", "Modified Sinc", "Schmid et al. 2022: keeps peak heights, strong stop band"),
    ("BUTTERWORTH", "Butterworth", "Zero-phase Butterworth, the biomechanics standard"),
    ("BSPLINE", "B-Spline", "Least-squares cubic B-spline (Yan 2022)"),
    ("ONE_EURO", "1 Euro", "Speed adaptive: heavy smoothing at rest, light smoothing in fast motion"),
)
RANGE_ITEMS = (
    ("ALL", "Whole Take", "Work on the whole action"),
    ("PREVIEW", "Preview Range", "Work on the scene preview range (Ctrl+P in the timeline)"),
)
PRESET_ITEMS = (
    ("OPTICAL", "Optical", "Marker based systems (Vicon, OptiTrack): spikes, dropouts, light filtering"),
    ("IMU", "Inertial Suit", "Rokoko, Xsens, mocopi: drift, floor level, foot locking"),
    ("VIDEO", "Video / AI", "Markerless capture: stronger smoothing, contacts, physical plausibility"),
)


def _tag_redraw(*_):
    wm = bpy.context.window_manager
    if wm is None:
        return
    for win in wm.windows:
        for area in win.screen.areas:
            if area.type in {"VIEW_3D", "GRAPH_EDITOR", "DOPESHEET_EDITOR"}:
                area.tag_redraw()


def _redraw_update(self, context):
    _tag_redraw()


_suspended = 0


class suspend_updates:
    """Context manager: property updates inside do not trigger a re-solve (bulk edits)."""

    def __enter__(self):
        global _suspended
        _suspended += 1

    def __exit__(self, *exc):
        global _suspended
        _suspended -= 1
        return False


def _pin_update(self, context):
    """A pin changed: re-solve (deferred, so dragging a slider stays responsive)."""
    if _suspended:
        return
    rig = self.id_data
    if self.lock_out < self.lock_in:
        with suspend_updates():
            self.lock_out = self.lock_in
    from .ui import autosolve
    autosolve.schedule(rig)
    _tag_redraw()


def _solve_setting_update(self, context):
    if _suspended:
        return
    from .ui import autosolve
    from .ui.common import get_rig
    rig = get_rig(context)
    if rig is not None:
        autosolve.schedule(rig, only_if_solved=True)
    _tag_redraw()


def _poll_armature(self, obj):
    return obj.type == "ARMATURE"


def _poll_mesh(self, obj):
    return obj.type == "MESH"


class MCPin(PropertyGroup):
    owner: EnumProperty(name="Limb", items=OWNER_ITEMS, update=_pin_update)
    mode: EnumProperty(name="Mode", items=MODE_ITEMS, default="FLOOR", update=_pin_update)
    lock_in: IntProperty(name="In", description="First fully locked frame", default=1, update=_pin_update)
    lock_out: IntProperty(name="Out", description="Last fully locked frame", default=10, update=_pin_update)
    blend_in: IntProperty(name="Ease In", description="Frames to blend from the animation into the lock", default=4,
                          min=0, soft_max=30, update=_pin_update)
    blend_out: IntProperty(name="Ease Out", description="Frames to blend from the lock back to the animation", default=5,
                           min=0, soft_max=30, update=_pin_update)
    use_reference: BoolProperty(name="Reference Frame", default=False, update=_pin_update,
                                description="Take the locked pose from one frame instead of the automatic "
                                            "(most stationary) placement")
    reference_frame: IntProperty(name="Frame", description="Frame whose pose is held", default=1, update=_pin_update)
    lock_location: BoolProperty(name="Location", default=True, update=_pin_update)
    lock_rotation: FloatProperty(name="Rotation", description="0: the limb end pivots freely (ball joint), "
                                 "1: rotation is locked too", default=1.0, min=0.0, max=1.0, subtype="FACTOR",
                                 update=_pin_update)
    target_object: PointerProperty(name="Target", type=bpy.types.Object, update=_pin_update)
    target_bone: StringProperty(name="Bone", update=_pin_update)
    enabled: BoolProperty(name="Enabled", default=True, update=_pin_update)
    auto: BoolProperty(name="Detected", description="Created by the contact detector", default=False)

    def to_data(self) -> KinematicPin:
        return KinematicPin(
            owner=self.owner, mode=self.mode, lock_in=self.lock_in, lock_out=max(self.lock_out, self.lock_in),
            blend_in=self.blend_in, blend_out=self.blend_out,
            reference_frame=self.reference_frame if self.use_reference else -1,
            lock_location=self.lock_location, lock_rotation=self.lock_rotation,
            target_object=self.target_object.name if self.target_object else "",
            target_bone=self.target_bone, enabled=self.enabled, auto=self.auto, name=self.name)

    def from_data(self, p: KinematicPin) -> None:
        with suspend_updates():
            for key in ("owner", "mode", "lock_in", "lock_out", "blend_in", "blend_out", "lock_location",
                        "lock_rotation", "target_bone", "enabled", "auto", "name"):
                setattr(self, key, getattr(p, key))
            self.use_reference = p.reference_frame >= 0
            self.reference_frame = max(p.reference_frame, 0)
            self.target_object = bpy.data.objects.get(p.target_object) if p.target_object else None


class MCIssue(PropertyGroup):
    kind: StringProperty()
    frame_start: IntProperty()
    frame_end: IntProperty()
    bone: StringProperty()
    severity: FloatProperty()
    message: StringProperty()


def _issue_index_update(self, context):
    from .ui.common import focus_issue
    focus_issue(context, self.id_data)


class MCRigData(PropertyGroup):
    pins: CollectionProperty(type=MCPin)
    active_pin: IntProperty(name="Active Pin", default=0, update=_redraw_update)
    issues: CollectionProperty(type=MCIssue)
    active_issue: IntProperty(name="Active Issue", default=0, update=_issue_index_update)
    analysis_summary: StringProperty()
    analysis_stale: BoolProperty(default=False)
    suggested_cutoff: FloatProperty(default=0.0)
    floor_estimate: FloatProperty(default=0.0)
    has_floor_estimate: BoolProperty(default=False)
    last_result: StringProperty()

    def pin_data(self) -> list[KinematicPin]:
        return [p.to_data() for p in self.pins]


class MCSettings(PropertyGroup):
    rig: PointerProperty(name="Rig", type=bpy.types.Object, poll=_poll_armature,
                         description="Animated Rigify / Rig Bridge control rig (empty: the active armature)")
    stage: EnumProperty(name="Stage", items=STAGE_ITEMS, default="DIAGNOSE")
    preset: EnumProperty(name="Source", items=PRESET_ITEMS, default="OPTICAL")
    range_mode: EnumProperty(name="Range", items=RANGE_ITEMS, default="ALL")
    only_selected: BoolProperty(name="Selected Bones Only", default=False,
                                description="Limit curve edits to the selected pose bones")

    # floor
    floor_z: FloatProperty(name="Floor Height", default=0.0, unit="LENGTH", update=_solve_setting_update)
    floor_auto: BoolProperty(name="Floor from Feet", default=True, update=_solve_setting_update,
                             description="Use the height at which the feet rest in this take as the floor (refreshed "
                                         "whenever the take is analyzed or contacts are detected). Turn off to set the "
                                         "height yourself, e.g. to bring a floating take down onto the ground")
    floor_object: PointerProperty(name="Terrain", type=bpy.types.Object, poll=_poll_mesh, update=_solve_setting_update,
                                  description="Optional mesh to walk on (stairs, slopes); the plane is the fallback")

    # stage 1
    spike_sensitivity: FloatProperty(name="Spike Threshold", default=6.5, min=3.0, max=20.0,
                                     description="How far an outlier must stick out of its neighbourhood (lower = more spikes)")
    jitter_sensitivity: FloatProperty(name="Jitter Threshold", default=3.5, min=1.5, max=5.5,
                                      description="Noise level relative to the take that is marked as jitter")
    hyperext_ratio: FloatProperty(name="Reach Limit", default=0.985, min=0.9, max=1.0, precision=3,
                                  description="Limb extension above which a frame is marked")
    repair_flips: BoolProperty(name="Flips", default=True)
    repair_spikes: BoolProperty(name="Spikes", default=True)
    repair_frozen: BoolProperty(name="Dropouts", default=True)

    # stage 2
    filter_method: EnumProperty(name="Method", items=FILTER_ITEMS, default="WHITTAKER")
    cutoff_hz: FloatProperty(name="Cutoff", default=8.0, min=0.5, max=30.0, step=50, precision=1,
                             description="Motion faster than this (in Hz) is removed")
    filter_strength: FloatProperty(name="Strength", default=1.0, min=0.0, max=1.0, subtype="FACTOR")
    filter_adaptive: BoolProperty(name="Only Where Needed", default=False,
                                  description="Filter only frames the diagnosis marked as noisy; everything else "
                                              "stays bit-exact (run Analyze first)")
    filter_location: BoolProperty(name="Location", default=True)
    filter_rotation: BoolProperty(name="Rotation", default=True)
    euro_beta: FloatProperty(name="Speed Response", default=1.0, min=0.0, max=20.0,
                             description="1 Euro filter: how quickly the cutoff opens up with speed")
    restore_blend: IntProperty(name="Falloff", default=4, min=0, max=60, description="Blend frames at range borders")

    # stage 3
    drift_smooth_hz: FloatProperty(name="Drift Bandwidth", default=1.5, min=0.1, max=6.0, precision=2,
                                   description="Drift slower than this (Hz) is removed; faster foot sliding is "
                                               "left to the pins")
    drift_strength: FloatProperty(name="Strength", default=1.0, min=0.0, max=1.0, subtype="FACTOR")
    loop_blend: IntProperty(name="Blend Frames", default=10, min=2, max=120)
    loop_keep_travel: BoolProperty(name="Keep Travel", default=True,
                                   description="Leave the net forward travel of the body alone (walk / run cycles)")

    # stage 4
    contact_height: FloatProperty(name="Height Tolerance", default=0.04, min=0.005, max=0.3, unit="LENGTH",
                                  description="How close to the floor a foot has to be (for a 1.75 m character)")
    contact_speed: FloatProperty(name="Speed Tolerance", default=0.45, min=0.05, max=3.0,
                                 description="How still the contact point has to be (m/s for a 1.75 m character)")
    contact_min_time: FloatProperty(name="Min Duration", default=0.1, min=0.03, max=1.0, unit="TIME_ABSOLUTE")
    contact_hands: BoolProperty(name="Hands", default=False, description="Also detect hands resting on the floor")
    pin_blend_in: IntProperty(name="Ease In", default=4, min=0, max=60)
    pin_blend_out: IntProperty(name="Ease Out", default=5, min=0, max=60)
    pin_length: IntProperty(name="Length", default=20, min=1, max=10000, description="Length of a new pin in frames")
    use_pins: BoolProperty(name="Pins", default=True, update=_solve_setting_update)
    use_floor: BoolProperty(name="Floor Clipping", default=True, update=_solve_setting_update,
                            description="Soles, toes and wrists never dip below the floor")
    clip_hands: BoolProperty(name="Hands", default=True, update=_solve_setting_update,
                             description="Wrists stay above the floor as well")
    clip_toes: BoolProperty(name="Toes", default=True, update=_solve_setting_update,
                            description="Bend toes that would go through the floor, and lay the toes of a planted foot onto it")
    floor_cushion: FloatProperty(name="Cushion", default=0.1, min=0.0, max=1.0, unit="TIME_ABSOLUTE",
                                 update=_solve_setting_update,
                                 description="Time over which a limb is eased away from the floor")
    flat_tolerance: FloatProperty(name="Flatten", default=4.0, min=0.0, max=30.0, update=_solve_setting_update,
                                  description="Planted feet tilted less than this (degrees) lie flat")
    auto_solve: BoolProperty(name="Live Update", default=True,
                             description="Re-solve automatically when a pin or a solver setting changes")

    # stage 5
    use_knee_fix: BoolProperty(name="Knee / Elbow Pops", default=True, update=_solve_setting_update)
    max_extension: FloatProperty(name="Max Extension", default=0.995, min=0.9, max=1.0, precision=3,
                                 update=_solve_setting_update, description="Limbs never stretch beyond this ratio")
    softness: FloatProperty(name="Softness", default=0.03, min=0.001, max=0.2, precision=3, update=_solve_setting_update,
                            description="Width of the soft zone below the limit")
    use_pelvis: BoolProperty(name="Pelvis Gives Way", default=True, update=_solve_setting_update,
                             description="Planted feet stay planted: the hips drop a little instead")
    pelvis_max_drop: FloatProperty(name="Max Drop", default=0.08, min=0.0, max=0.5, unit="LENGTH",
                                   update=_solve_setting_update)
    pelvis_smooth: FloatProperty(name="Smoothing", default=0.17, min=0.0, max=1.0, unit="TIME_ABSOLUTE",
                                 update=_solve_setting_update)
    pelvis_horizontal: BoolProperty(name="Also Sideways", default=False, update=_solve_setting_update)
    use_heel_peel: BoolProperty(name="Heel Peel", default=False, update=_solve_setting_update,
                                description="Lift the heel of an over-stretched planted foot before moving the hips")
    heel_peel_max: FloatProperty(name="Max Angle", default=30.0, min=0.0, max=60.0, update=_solve_setting_update)
    fix_elbows: BoolProperty(name="Arms", default=True, update=_solve_setting_update)
    use_pole_fix: BoolProperty(name="Pole Whips", default=True, update=_solve_setting_update,
                               description="Remove sudden sideways flips of knees and elbows")
    use_ballistic: BoolProperty(name="Ballistic Jumps", default=False, update=_solve_setting_update,
                                description="Airborne phases follow a true gravity parabola")
    ballistic_strength: FloatProperty(name="Strength", default=1.0, min=0.0, max=1.0, subtype="FACTOR",
                                      update=_solve_setting_update)
    gravity_scale: FloatProperty(name="Gravity", default=1.0, min=0.05, max=4.0, update=_solve_setting_update)
    use_capsules: BoolProperty(name="Body Collision", default=False, update=_solve_setting_update,
                               description="Push hands out of torso, head and thighs")
    body_thickness: FloatProperty(name="Thickness", default=1.0, min=0.3, max=3.0, update=_solve_setting_update)

    # stage 6
    simplify_tol_loc: FloatProperty(name="Location", default=0.002, min=0.0001, max=0.1, unit="LENGTH", precision=4)
    simplify_tol_rot: FloatProperty(name="Rotation", default=0.5, min=0.01, max=10.0, description="Degrees")
    simplify_residual: BoolProperty(name="Residual Detail Layer", default=True,
                                    description="Keep what the sparse keys cannot express on an NLA layer")
    simplify_keep_contacts: BoolProperty(name="Keys at Contacts", default=True,
                                         description="Always keep a key where a pin starts or ends")

    # one click pipeline
    pipe_repair: BoolProperty(name="Repair", default=True)
    pipe_filter: BoolProperty(name="Filter", default=True)
    pipe_dedrift: BoolProperty(name="De-Drift", default=False)
    pipe_level: BoolProperty(name="Level Floor", default=False)
    pipe_contacts: BoolProperty(name="Contacts", default=True)
    pipe_solve: BoolProperty(name="Solve", default=True)

    # overlays
    show_heatmap: BoolProperty(name="Error Marks", default=True, update=_redraw_update)
    show_ghost: BoolProperty(name="Raw Ghost", default=True, update=_redraw_update)
    show_pin_tracks: BoolProperty(name="Pin Tracks", default=True, update=_redraw_update,
                                  description="Show the pins of each limb as bars at the bottom of the Dope Sheet. "
                                              "Drag a bar to move it, its edges to resize, its ramps to change the blend")
    show_pin_tracks_graph: BoolProperty(name="Also in Graph Editor", default=False, update=_redraw_update,
                                        description="Show (and edit) the pin tracks in the Graph Editor as well")
    show_hud: BoolProperty(name="Viewport HUD", default=True, update=_redraw_update)
    show_com: BoolProperty(name="Centre of Mass", default=True, update=_redraw_update)
    show_floor: BoolProperty(name="Floor", default=True, update=_redraw_update)
    show_capsules: BoolProperty(name="Body Capsules", default=False, update=_redraw_update)
    marks_only_selected: BoolProperty(name="Selected Only", default=False, update=_redraw_update,
                                      description="Show error marks of the selected bones only")


def solve_settings(st: MCSettings, fps: float) -> SolveSettings:
    """Solver settings from the scene properties (times -> frames)."""

    def frames(seconds: float, lo: int = 0) -> int:
        return max(lo, int(round(seconds * fps)))

    return SolveSettings(
        floor_z=st.floor_z, floor_object=st.floor_object.name if st.floor_object else "", use_floor=st.use_floor,
        floor_cushion=frames(st.floor_cushion), clip_hands=st.clip_hands, clip_toes=st.clip_toes,
        flat_tolerance_deg=st.flat_tolerance, use_pins=st.use_pins,
        use_knee_fix=st.use_knee_fix, max_extension=st.max_extension, softness=st.softness,
        use_pelvis=st.use_pelvis, pelvis_max_drop=st.pelvis_max_drop, pelvis_smooth=frames(st.pelvis_smooth),
        pelvis_horizontal=st.pelvis_horizontal, use_heel_peel=st.use_heel_peel, heel_peel_max_deg=st.heel_peel_max,
        fix_elbows=st.fix_elbows, use_pole_fix=st.use_pole_fix,
        use_ballistic=st.use_ballistic, ballistic_strength=st.ballistic_strength, gravity_scale=st.gravity_scale,
        use_capsules=st.use_capsules, body_thickness=st.body_thickness,
        ik_switch_blend=max(2, frames(0.13)),
    )


CLASSES = (MCPin, MCIssue, MCRigData, MCSettings)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.mocap_cleanup = PointerProperty(type=MCSettings)
    bpy.types.Object.mocap_cleanup = PointerProperty(type=MCRigData)


def unregister():
    del bpy.types.Object.mocap_cleanup
    del bpy.types.Scene.mocap_cleanup
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    runtime.clear_all()
