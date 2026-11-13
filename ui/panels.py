"""Sidebar panels (3D View, Graph Editor, Dope Sheet): one stage at a time."""

from __future__ import annotations

import textwrap

import bpy
from bpy.types import Panel, UIList

from .. import props
from ..core import action_io, raw_snapshot, runtime
from ..stages import stage1_diagnostics as s1
from ..stages import stage6_simplifier as s6
from . import common

ISSUE_ICONS = {
    "FLIP": "FILE_REFRESH", "SPIKE": "ERROR", "JITTER": "RNDCURVE", "FROZEN": "FREEZE",
    "HYPEREXT": "CON_KINEMATIC", "KNEE_POP": "CANCEL", "PENETRATION": "TRIA_DOWN_BAR", "SKATE": "TRACKING_FORWARDS",
}
ISSUE_NAMES = {
    "FLIP": "Flip", "SPIKE": "Spike", "JITTER": "Jitter", "FROZEN": "Frozen", "HYPEREXT": "Over-extended",
    "KNEE_POP": "Pop", "PENETRATION": "Below floor", "SKATE": "Slide",
}
MODE_ICONS = {"FLOOR": "IMPORT", "WORLD": "PINNED", "TARGET": "CON_CHILDOF"}


def _wrapped(layout, context, text: str, icon: str = "NONE") -> None:
    """A label that wraps to the width of the sidebar."""
    region = getattr(context, "region", None)
    scale = context.preferences.system.ui_scale or 1.0
    chars = max(18, int((region.width if region else 300 * scale) / (6.4 * scale)) - 8)
    col = layout.column(align=True)
    col.scale_y = 0.8
    for i, line in enumerate(textwrap.wrap(text, chars) or [""]):
        col.label(text=line, icon=icon if i == 0 else "NONE")


class MOCAP_UL_issues(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        row = layout.row(align=True)
        row.active = not item.idle                   # on a control that does not move the character there
        row.label(text="", icon=ISSUE_ICONS.get(item.kind, "DOT"))
        frames = f"{item.frame_start}" if item.frame_start == item.frame_end else f"{item.frame_start}-{item.frame_end}"
        sub = row.row()
        sub.ui_units_x = 3.6
        sub.label(text=frames)
        # short enough for a narrow sidebar; the whole message is shown below the list
        row.label(text=f"{ISSUE_NAMES.get(item.kind, item.kind.title())}  {item.bone}")

    def filter_items(self, context, data, propname):
        items = getattr(data, propname)
        show_idle = context.scene.mocap_cleanup.show_idle_issues
        flt = [self.bitflag_filter_item if (show_idle or not it.idle) else 0 for it in items]
        if self.filter_name:
            needle = self.filter_name.lower()
            flt = [f if (needle in it.message.lower() or needle in it.kind.lower()
                         or needle in ISSUE_NAMES.get(it.kind, "").lower()) else 0
                   for f, it in zip(flt, items)]
        return flt, []


class MOCAP_UL_pins(UIList):
    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        # checkbox | hold icon, limb and frames in one label (a list row gives each label an equal share of a
        # narrow sidebar, which cut both short); the hold is spelled out in the details below the list
        layout.prop(item, "enabled", text="")
        tail = ""
        if item.mode == "TARGET":
            tail = "  > " + (item.target_object.name if item.target_object else "?")
        elif item.mode == "WORLD" and not (item.lock_location and item.lock_rotation > 0.0):
            tail = "  Position" if item.lock_location else "  Rotation"
        owner = props.OWNER_LABELS.get(item.owner, item.owner)
        layout.label(text=f"{owner}   {item.lock_in}-{item.lock_out}{tail}", icon=MODE_ICONS.get(item.mode, "PINNED"))


def _header(layout, st, rig):
    col = layout.column(align=True)
    row = col.row(align=True)
    row.prop(st, "rig", text="")
    row.operator("mocap.use_active_rig", text="", icon="EYEDROPPER")
    if rig is None:
        col.label(text="Pick an animated rig or skeleton", icon="INFO")
        return False
    act = action_io.get_active_action(rig)
    if act is None:
        col.label(text=f"'{rig.name}' has no animation", icon="INFO")
        return False
    profile = common.profile_of(rig)
    f0, f1 = action_io.get_action_frame_range(rig)
    info = col.row()
    info.label(text=f"{act.name}  [{f0}-{f1}]", icon="ACTION")
    if not profile.is_supported:
        box = layout.box()
        box.label(text="No control rig or humanoid skeleton found:", icon="ERROR")
        box.label(text="curve tools only (diagnose, filter, keys)")
    elif profile.kind == "SKELETON":
        sub = col.row()
        sub.scale_y = 0.8
        sub.label(text="Skeleton: no control rig needed", icon="ARMATURE_DATA")
    row = layout.row(align=True)
    row.prop(st, "preset", text="")
    row.operator("mocap.apply_preset", text="", icon="CHECKMARK")
    row.operator("mocap.workspace_setup", text="", icon="WORKSPACE")
    screen = getattr(bpy.context, "screen", None)
    if not (screen is not None and screen.get("mocap_editor", False)):
        row = layout.row()
        row.scale_y = 1.2
        row.operator("mocap.editor_open", icon="WINDOW")
    return True


def _run_all_footer(layout) -> None:
    """The one-click button comes after the steps it runs, not before Analyze (GUI feedback)."""
    box = layout.box()
    row = box.row(align=True)
    row.scale_y = 1.3
    row.operator("mocap.run_all", text="Clean Up Take (all steps)", icon="PLAY")


def _floor_rows(col, st) -> None:
    col.prop(st, "floor_object")
    if st.floor_object is None:
        col.prop(st, "floor_auto")
        col.prop(st, "floor_z")


def _scope(layout, st):
    row = layout.row(align=True)
    row.prop(st, "range_mode", text="")
    row.prop(st, "only_selected", text="", icon="RESTRICT_SELECT_OFF")


def _draw_diagnose(layout, context, st, rig):
    data = rig.mocap_cleanup
    col = layout.column(align=True)
    row = col.row(align=True)
    row.scale_y = 1.2
    row.operator("mocap.analyze", icon="VIEWZOOM")
    row.operator("mocap.issue_jump", text="", icon="TRIA_LEFT").direction = -1
    row.operator("mocap.issue_jump", text="", icon="TRIA_RIGHT").direction = 1
    if data.analysis_summary:
        box = layout.box()
        sub = box.column(align=True)
        sub.scale_y = 0.85
        if data.analysis_stale:
            sub.label(text="Out of date - analyze again", icon="TIME")
        text = data.analysis_summary
        for part in [p.strip() for p in text.split(",")]:
            line = sub.row()
            line.active = "unused controls" not in part     # not counted: they do not move the character
            line.label(text=part)
        if data.suggested_cutoff > 0.0:
            sub.label(text=f"Suggested filter cutoff: {data.suggested_cutoff:.1f} Hz")
        rest = data.floor_estimate - data.sole_offset
        if (data.has_floor_estimate and not st.floor_auto and st.floor_object is None
                and abs(rest - st.floor_z) > 0.02 * common.profile_of(rig).unit):
            row = box.row(align=True)
            row.label(text=f"Feet rest at {rest:.3f}", icon="INFO")
            row.operator("mocap.floor_from_feet", text="Use")
    if len(data.issues):
        layout.template_list("MOCAP_UL_issues", "", data, "issues", data, "active_issue", rows=6)
        n_idle = sum(1 for it in data.issues if it.idle)
        if n_idle:
            layout.prop(st, "show_idle_issues", text=f"Show {n_idle} on Unused Controls")
        if 0 <= data.active_issue < len(data.issues):
            issue = data.issues[data.active_issue]
            fix = s1.ISSUE_FIX.get(issue.kind, "")
            label = {"REPAIR": "Repair This", "FILTER": "Filter This", "SOLVE": "Solve (Floor / Knees)",
                     "CONTACTS": "Plant This Foot"}.get(fix, "Fix")
            _wrapped(layout, context, issue.message, ISSUE_ICONS.get(issue.kind, "DOT"))
            layout.operator("mocap.issue_fix", text=label, icon="TOOL_SETTINGS")
    box = layout.box()
    box.label(text="Structural Repair", icon="MODIFIER")
    row = box.row(align=True)
    row.prop(st, "repair_flips", toggle=True)
    row.prop(st, "repair_spikes", toggle=True)
    row.prop(st, "repair_frozen", toggle=True)
    _scope(box, st)
    row = box.row(align=True)
    row.operator("mocap.repair", icon="CHECKMARK")
    row.operator("mocap.fill_gap", text="Rebuild Range", icon="IPO_BEZIER")
    col = layout.column(align=True)
    col.use_property_split = True
    col.use_property_decorate = False
    col.prop(st, "spike_sensitivity")
    col.prop(st, "jitter_sensitivity")
    col.prop(st, "hyperext_ratio")


def _draw_filter(layout, context, st, rig):
    data = rig.mocap_cleanup
    col = layout.column()
    col.use_property_split = True
    col.use_property_decorate = False
    col.prop(st, "filter_method")
    row = col.row(align=True)
    row.prop(st, "cutoff_hz", text="Cutoff (Hz)")
    row.operator("mocap.estimate_cutoff", text="", icon="EYEDROPPER")
    if st.filter_method == "ONE_EURO":
        col.prop(st, "euro_beta")
    col.prop(st, "filter_strength")
    col.prop(st, "filter_max_shift")
    col.prop(st, "filter_adaptive")
    row = layout.row(align=True)
    row.prop(st, "filter_location", toggle=True)
    row.prop(st, "filter_rotation", toggle=True)
    _scope(layout, st)
    row = layout.row()
    row.scale_y = 1.3
    row.operator("mocap.filter", icon="SMOOTHCURVE")
    if st.filter_adaptive and (not data.analysis_summary or data.analysis_stale):
        layout.label(text="Analysis is refreshed automatically", icon="INFO")
    _restore_box(layout, st, rig)


def _restore_box(layout, st, rig):
    box = layout.box()
    snap = raw_snapshot.get_raw_snapshot(rig)
    box.label(text="Raw Take" + ("" if snap else " (untouched)"), icon="GHOST_ENABLED" if snap else "GHOST_DISABLED")
    col = box.column(align=True)
    col.enabled = snap is not None
    row = col.row(align=True)
    row.operator("mocap.restore_raw", icon="LOOP_BACK")
    row.prop(st, "restore_blend", text="Falloff")
    row = col.row(align=True)
    row.operator("mocap.snapshot", text="Set Current as Raw").action = "REFRESH"
    row.operator("mocap.snapshot", text="", icon="TRASH").action = "DELETE"


def _draw_drift(layout, context, st, rig):
    col = layout.column(align=True)
    col.use_property_split = True
    col.use_property_decorate = False
    _floor_rows(col, st)
    if not st.floor_auto and st.floor_object is None:
        col.operator("mocap.floor_from_feet", text="Set from Feet Once", icon="TRIA_DOWN_BAR")
    box = layout.box()
    box.label(text="Drift (inertial suits, video)", icon="TRACKING")
    col = box.column(align=True)
    col.use_property_split = True
    col.use_property_decorate = False
    col.prop(st, "drift_smooth_hz", text="Bandwidth (Hz)")
    col.prop(st, "drift_strength")
    row = box.row(align=True)
    row.operator("mocap.dedrift", icon="CON_LOCLIMIT")
    row.operator("mocap.level_floor", icon="IPO_LINEAR")
    box = layout.box()
    box.label(text="Cycle", icon="FILE_REFRESH")
    col = box.column(align=True)
    col.use_property_split = True
    col.use_property_decorate = False
    col.prop(st, "loop_blend")
    col.prop(st, "loop_keep_travel")
    box.operator("mocap.close_loop", icon="FILE_REFRESH")


def _solve_row(layout, st, rig):
    data = rig.mocap_cleanup
    row = layout.row(align=True)
    row.scale_y = 1.4
    row.operator("mocap.solve", icon="PLAY")
    row.operator("mocap.clear_solve", text="", icon="X")
    row.prop(st, "auto_solve", text="", icon="FILE_REFRESH")
    if raw_snapshot.get_solve_base(rig) is not None:
        res = runtime.SOLVE_RESULTS.get(rig.name)
        if res is not None:
            col = layout.column(align=True)
            col.scale_y = 0.8
            for part in res.summary().split(", "):
                col.label(text=part.replace("Solved (", "").replace(")", ""))
            for w in res.warnings[:3]:
                col.label(text=w, icon="ERROR")


def _draw_pins(layout, context, st, rig):
    data = rig.mocap_cleanup
    row = layout.row(align=True)
    row.scale_y = 1.2
    row.operator("mocap.detect_contacts", icon="VIEWZOOM")
    _pin_tool(layout, context, st, rig)
    row = layout.row()
    row.template_list("MOCAP_UL_pins", "", data, "pins", data, "active_pin", rows=4)
    col = row.column(align=True)
    col.operator("mocap.pin_remove", text="", icon="REMOVE")
    col.separator()
    col.operator("mocap.pin_clear", text="", icon="VIEWZOOM").auto_only = True
    col.operator("mocap.pin_clear", text="", icon="TRASH").auto_only = False
    if 0 <= data.active_pin < len(data.pins):
        pin = data.pins[data.active_pin]
        box = layout.box()
        col = box.column(align=True)
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(pin, "name", text="Name")
        col.prop(pin, "owner")
        col.prop(pin, "mode")
        if pin.mode == "TARGET":
            col.prop(pin, "target_object")
            if pin.target_object is not None and pin.target_object.type == "ARMATURE":
                col.prop_search(pin, "target_bone", pin.target_object.data, "bones", text="Bone")
        row = box.row(align=True)
        row.prop(pin, "lock_in")
        row.operator("mocap.pin_edge", text="", icon="TRIA_LEFT_BAR").edge = "IN"
        row.operator("mocap.pin_edge", text="", icon="TRIA_RIGHT_BAR").edge = "OUT"
        row.prop(pin, "lock_out")
        row = box.row(align=True)
        row.prop(pin, "blend_in")
        row.prop(pin, "blend_out")
        row = box.row(align=True)
        row.prop(pin, "lock_location", toggle=True)
        row.prop(pin, "lock_rotation", slider=True)
        if pin.mode != "FLOOR" or pin.use_reference:
            row = box.row(align=True)
            row.prop(pin, "use_reference", text="Pose of Frame")
            sub = row.row(align=True)
            sub.enabled = pin.use_reference
            sub.prop(pin, "reference_frame", text="")
            sub.operator("mocap.pin_edge", text="", icon="TIME").edge = "REF"
        else:
            box.prop(pin, "use_reference", text="Pose of Frame (automatic placement)")
    box = layout.box()
    box.label(text="Floor", icon="TRIA_DOWN_BAR")
    col = box.column(align=True)
    col.use_property_split = True
    col.use_property_decorate = False
    _floor_rows(col, st)
    col.prop(st, "use_floor")
    sub = col.column(align=True)
    sub.enabled = st.use_floor
    sub.prop(st, "clip_toes")
    sub.prop(st, "clip_hands")
    sub.prop(st, "floor_cushion", text="Cushion (s)")
    sub.prop(st, "flat_tolerance")
    _solve_row(layout, st, rig)
    if st.auto_solve:
        note = layout.row()
        note.scale_y = 0.8
        note.label(text="Live Update is on: pins apply as you add or change them", icon="FILE_REFRESH")
    box = layout.box()
    box.prop(st, "show_contact_settings", emboss=False,
             icon="DOWNARROW_HLT" if st.show_contact_settings else "RIGHTARROW")
    if st.show_contact_settings:
        col = box.column(align=True)
        col.use_property_split = True
        col.use_property_decorate = False
        col.prop(st, "contact_height")
        col.prop(st, "contact_speed")
        col.prop(st, "contact_travel")
        col.prop(st, "pin_max_shift")
        col.prop(st, "contact_min_time", text="Min Duration (s)")
        col.prop(st, "contact_hands")


def _is_pin_result(text: str) -> bool:
    """The result line of Pin from Here / Release Here (shown under those buttons, not at the panel's end)."""
    head = text.split(" held ", 1)[0] if " held " in text else text.split(" released at ", 1)[0]
    return head != text and head in props.OWNER_LABELS.values()


def _pin_tool(layout, context, st, rig):
    """Hold a hand, foot or the hips from the current frame on (iClone-style pinning)."""
    box = layout.box()
    box.label(text="Hold a Hand or Foot", icon="PINNED")
    # Laid out like the body seen from the front: the character's right hand on the left.
    col = box.column(align=True)
    row = col.row(align=True)
    for key in ("arm.R", "torso", "arm.L"):
        row.prop_enum(st, "pin_limb", key)
    row = col.row(align=True)
    for key in ("leg.R", "leg.L"):
        row.prop_enum(st, "pin_limb", key)
    box.row(align=True).prop(st, "pin_hold", expand=True)
    if st.pin_hold == "WORLD":
        row = box.row(align=True)
        row.prop(st, "pin_position", toggle=True, icon="ORIENTATION_GLOBAL")
        row.prop(st, "pin_rotation", toggle=True, icon="ORIENTATION_GIMBAL")
    elif st.pin_hold == "TARGET":
        box.label(text="Select the object too (the rig active)", icon="INFO")
    col = box.column(align=True)
    col.prop(st, "pin_length", text="Length (frames)")
    row = col.row(align=True)
    row.prop(st, "pin_blend_in", text="Ease In")
    row.prop(st, "pin_blend_out", text="Ease Out")
    row = box.row(align=True)
    row.scale_y = 1.3
    row.operator("mocap.pin_from_here", icon="PINNED")
    row.operator("mocap.pin_release_here", icon="UNPINNED")
    f = context.scene.frame_current
    hint = box.row()
    hint.scale_y = 0.8
    hint.enabled = False
    hint.label(text=f"Pin from Here holds frames {f}-{f + st.pin_length - 1}")
    text = common.result_for(rig)
    if text and _is_pin_result(text):
        _wrapped(box.box(), context, text)


def _draw_physics(layout, context, st, rig):
    box = layout.box()
    box.prop(st, "use_knee_fix", icon="CON_KINEMATIC")
    col = box.column(align=True)
    col.enabled = st.use_knee_fix
    col.use_property_split = True
    col.use_property_decorate = False
    col.prop(st, "max_extension")
    col.prop(st, "softness")
    col.prop(st, "fix_elbows")
    col.prop(st, "use_pole_fix")
    col.separator()
    col.prop(st, "use_pelvis")
    sub = col.column(align=True)
    sub.enabled = st.use_pelvis
    sub.prop(st, "pelvis_max_drop")
    sub.prop(st, "pelvis_smooth", text="Smoothing (s)")
    sub.prop(st, "pelvis_horizontal")
    col.separator()
    col.prop(st, "use_heel_peel")
    sub = col.column(align=True)
    sub.enabled = st.use_heel_peel
    sub.prop(st, "heel_peel_max")
    box = layout.box()
    box.prop(st, "use_ballistic", icon="FORCE_FORCE")
    col = box.column(align=True)
    col.enabled = st.use_ballistic
    col.use_property_split = True
    col.use_property_decorate = False
    col.prop(st, "ballistic_strength")
    col.prop(st, "gravity_scale")
    box = layout.box()
    box.prop(st, "use_capsules", icon="MESH_CAPSULE")
    col = box.column(align=True)
    col.enabled = st.use_capsules
    col.use_property_split = True
    col.use_property_decorate = False
    col.prop(st, "body_thickness")
    col.prop(st, "show_capsules")
    _solve_row(layout, st, rig)


def _draw_simplify(layout, context, st, rig):
    has = s6.has_residual(rig)
    col = layout.column(align=True)
    col.use_property_split = True
    col.use_property_decorate = False
    col.label(text="Tolerance")
    col.prop(st, "simplify_tol_loc")
    col.prop(st, "simplify_tol_rot", text="Rotation (deg)")
    col.prop(st, "simplify_residual")
    col.prop(st, "simplify_keep_contacts")
    col.prop(st, "only_selected")
    row = layout.row()
    row.scale_y = 1.3
    row.operator("mocap.simplify", icon="KEYFRAME")
    if has:
        box = layout.box()
        on = s6.residual_enabled(rig)
        box.label(text="Residual detail layer is " + ("on" if on else "off"), icon="NLA")
        row = box.row(align=True)
        row.operator("mocap.toggle_residual", text="Hide Detail" if on else "Show Detail",
                     icon="HIDE_OFF" if on else "HIDE_ON")
        row.operator("mocap.merge_residual", icon="NLA_PUSHDOWN")
        box.label(text="Other stages need a merge first", icon="INFO")
    _restore_box(layout, st, rig)


STAGE_DRAW = {
    "DIAGNOSE": _draw_diagnose, "FILTER": _draw_filter, "DRIFT": _draw_drift, "CONTACTS": _draw_pins,
    "PHYSICS": _draw_physics, "SIMPLIFY": _draw_simplify,
}


class _MainPanel:
    bl_label = "MoCap Cleanup"
    bl_region_type = "UI"
    bl_category = "MoCap"

    def draw(self, context):
        layout = self.layout
        st = context.scene.mocap_cleanup
        rig = common.get_rig(context)
        if not _header(layout, st, rig):
            return
        # The steps top to bottom in the order they are done; the open one shows its settings right under
        # it (GUI feedback: a 2 x 3 grid of tabs did not say where to start or what comes next).
        items = st.bl_rna.properties["stage"].enum_items
        for ident in ("DIAGNOSE", "FILTER", "DRIFT", "CONTACTS", "PHYSICS", "SIMPLIFY"):
            is_open = st.stage == ident
            row = layout.row(align=True)
            row.scale_y = 1.15
            row.prop_enum(st, "stage", ident, icon="DOWNARROW_HLT" if is_open else "RIGHTARROW")
            if not is_open:
                continue
            box = layout.box()
            _wrapped(box, context, items[ident].description)
            if ident in {"CONTACTS", "PHYSICS"} and not common.profile_of(rig).is_supported:
                box.label(text="This step needs a control rig or a humanoid skeleton", icon="ERROR")
            else:
                STAGE_DRAW[ident](box, context, st, rig)
        text = common.result_for(rig)
        if text and not (st.stage == "CONTACTS" and _is_pin_result(text)):
            _wrapped(layout.box(), context, text)
        _run_all_footer(layout)


class _OverlayPanel:
    bl_label = "Overlays"
    bl_region_type = "UI"
    bl_category = "MoCap"
    bl_options = {"DEFAULT_CLOSED"}

    def draw(self, context):
        st = context.scene.mocap_cleanup
        col = self.layout.column(align=True)
        col.label(text="Graph Editor / Dope Sheet")
        col.prop(st, "show_heatmap")
        sub = col.row()
        sub.enabled = st.show_heatmap
        sub.prop(st, "marks_only_selected")
        col.prop(st, "show_ghost")
        col.prop(st, "show_pin_tracks")
        sub = col.row()
        sub.enabled = st.show_pin_tracks
        sub.prop(st, "show_pin_tracks_graph")
        col.separator()
        col.label(text="3D Viewport")
        col.prop(st, "show_hud")
        sub = col.column(align=True)
        sub.enabled = st.show_hud
        sub.prop(st, "show_floor")
        sub.prop(st, "show_com")
        sub.prop(st, "show_capsules")
        col.separator()
        col.label(text="[ and ] jump between issues", icon="INFO")
        col.label(text="Drag the pin bars in the Dope Sheet", icon="INFO")


def _make_panels():
    classes = []
    for space, suffix in (("VIEW_3D", "view3d"), ("GRAPH_EDITOR", "graph"), ("DOPESHEET_EDITOR", "dopesheet")):
        main = type(f"MOCAP_PT_main_{suffix}", (_MainPanel, Panel), {"bl_space_type": space, "bl_idname": f"MOCAP_PT_main_{suffix}"})
        over = type(f"MOCAP_PT_overlays_{suffix}", (_OverlayPanel, Panel),
                    {"bl_space_type": space, "bl_idname": f"MOCAP_PT_overlays_{suffix}"})
        classes += [main, over]
    return classes


CLASSES = (MOCAP_UL_issues, MOCAP_UL_pins, *_make_panels())
