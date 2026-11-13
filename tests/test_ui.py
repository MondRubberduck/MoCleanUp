"""UI code that otherwise only runs in an interactive session: panel layouts, the workspace builder, GPU overlays."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np
from mathutils import Matrix

from _common import run_tests
import test_addon as ta

from mocap_cleanup import props
from mocap_cleanup.core import action_io, raw_snapshot
from mocap_cleanup.overlays import draw, pin_tracks
from mocap_cleanup.stages import stage6_simplifier as s6
from mocap_cleanup.ui import panels, workspace

ICONS = set(bpy.types.UILayout.bl_rna.functions["prop"].parameters["icon"].enum_items.keys())
LAYOUT_PROPS = set(bpy.types.UILayout.bl_rna.properties.keys())


# ---------------------------------------------------------------------------
# A stand-in for UILayout that checks every call against Blender's RNA
# ---------------------------------------------------------------------------

class MockOperatorProps:
    def __init__(self, idname):
        mod, name = idname.split(".")
        object.__setattr__(self, "_rna", getattr(getattr(bpy.ops, mod), name).get_rna_type())
        object.__setattr__(self, "_id", idname)

    def __setattr__(self, key, value):
        prop = self._rna.properties.get(key)
        assert prop is not None, f"operator {self._id} has no property '{key}'"
        if prop.type == "ENUM":
            assert value in prop.enum_items.keys(), f"{self._id}.{key} = {value!r}"


class MockLayout:
    def __init__(self, stats):
        object.__setattr__(self, "stats", stats)

    def __setattr__(self, key, value):
        assert key in LAYOUT_PROPS, f"UILayout has no property '{key}'"

    def _sub(self, *args, **kw):
        return MockLayout(self.stats)

    row = column = box = split = grid_flow = column_flow = _sub

    def _icon(self, kw):
        assert kw.get("icon", "NONE") in ICONS, kw.get("icon")

    def separator(self, **kw):
        pass

    def label(self, text="", **kw):
        self._icon(kw)
        assert isinstance(text, str)
        self.stats["labels"].append(text)

    def prop(self, data, name, **kw):
        assert data is not None and name in data.bl_rna.properties, f"{getattr(data, 'bl_rna', data)} . {name}"
        self._icon(kw)
        self.stats["props"].add(name)

    def prop_enum(self, data, name, value, **kw):
        assert value in data.bl_rna.properties[name].enum_items.keys(), (name, value)
        self.stats["props"].add(name)

    def prop_search(self, data, name, search_data, search_name, **kw):
        assert name in data.bl_rna.properties and search_name in search_data.bl_rna.properties, (name, search_name)

    def operator(self, idname, **kw):
        self._icon(kw)
        self.stats["ops"].add(idname)
        return MockOperatorProps(idname)

    def template_list(self, cls_name, list_id, data, name, active_data, active_name, **kw):
        cls = getattr(bpy.types, cls_name)
        assert name in data.bl_rna.properties and active_name in active_data.bl_rna.properties
        for i, item in enumerate(getattr(data, name)):
            cls.draw_item(None, bpy.context, MockLayout(self.stats), data, item, 0, active_data, active_name, i)
            self.stats["rows"] += 1


def draw_panel(cls):
    stats = {"labels": [], "props": set(), "ops": set(), "rows": 0}

    class Holder:
        layout = MockLayout(stats)

    cls.draw(Holder(), bpy.context)
    return stats


# ---------------------------------------------------------------------------

def test_every_panel_layout():
    rig, gt, st = ta.make_scene(n_frames=90)
    data = rig.mocap_cleanup
    assert bpy.ops.mocap.analyze() == {"FINISHED"} and len(data.issues) >= 3
    assert bpy.ops.mocap.detect_contacts() == {"FINISHED"} and len(data.pins) >= 4
    common_ops = {"mocap.run_all", "mocap.workspace_setup", "mocap.apply_preset", "mocap.use_active_rig"}
    expect = {
        "DIAGNOSE": {"mocap.analyze", "mocap.issue_jump", "mocap.issue_fix", "mocap.repair", "mocap.fill_gap"},
        "FILTER": {"mocap.filter", "mocap.estimate_cutoff", "mocap.restore_raw", "mocap.snapshot"},
        "DRIFT": {"mocap.dedrift", "mocap.level_floor", "mocap.close_loop", "mocap.floor_from_feet"},
        "CONTACTS": {"mocap.detect_contacts", "mocap.pin_from_here", "mocap.pin_release_here", "mocap.pin_remove",
                     "mocap.pin_clear", "mocap.pin_edge",
                     "mocap.solve", "mocap.clear_solve"},
        "PHYSICS": {"mocap.solve", "mocap.clear_solve"},
        "SIMPLIFY": {"mocap.simplify", "mocap.restore_raw"},
    }
    data.active_issue = 0
    data.active_pin = 0
    total = 0
    assert st.floor_auto and abs(st.floor_z) < 0.008, st.floor_z      # the floor was taken from the feet
    st.floor_auto = False                                              # (shows the manual floor controls)
    for stage, ops in expect.items():
        st.stage = stage
        stats = draw_panel(panels._MainPanel)
        assert (ops | common_ops) <= stats["ops"], (stage, sorted((ops | common_ops) - stats["ops"]))
        total += len(stats["props"])
        if stage == "DIAGNOSE":
            assert stats["rows"] == len(data.issues)
        if stage == "CONTACTS":
            assert stats["rows"] == len(data.pins)
            assert {"lock_in", "blend_out", "floor_z", "floor_auto", "floor_object", "use_floor", "pin_limb", "pin_hold",
                    "pin_position", "pin_rotation", "pin_length", "show_contact_settings"} <= stats["props"]
    # a pin that follows another object shows its target fields
    with props.suspend_updates():
        pin = data.pins[0]
        pin.mode, pin.target_object, pin.target_bone = "TARGET", rig, "head"
    st.stage = "CONTACTS"
    assert {"target_object", "use_reference", "reference_frame"} <= draw_panel(panels._MainPanel)["props"]
    with props.suspend_updates():
        pin.mode, pin.target_object = "FLOOR", None
    # the Keys stage with a residual layer (opt-in since it is off by default)
    st.simplify_residual = True
    assert bpy.ops.mocap.simplify() == {"FINISHED"} and s6.has_residual(rig)
    st.stage = "SIMPLIFY"
    assert {"mocap.toggle_residual", "mocap.merge_residual"} <= draw_panel(panels._MainPanel)["ops"]
    bpy.ops.mocap.merge_residual()
    overlay = draw_panel(panels._OverlayPanel)
    assert {"show_heatmap", "show_ghost", "show_pin_tracks", "show_pin_tracks_graph", "show_hud", "show_com"} <= overlay["props"]
    # nothing to work on: the panel says so instead of failing
    bpy.ops.object.mode_set(mode="OBJECT")
    empty = bpy.data.objects.new("JustAnEmpty", None)
    bpy.context.scene.collection.objects.link(empty)
    bpy.context.view_layer.objects.active = empty
    st.rig = None
    stats = draw_panel(panels._MainPanel)
    assert any("Pick" in text for text in stats["labels"]) and "mocap.run_all" not in stats["ops"]
    print(f"   6 stage panels + overlay panel: {total} property fields, every icon / property / operator exists")


def test_workspace_layout():
    """Background mode has no real window, so the three-editor layout itself is checked by
    tests/gui/check_workspace.py in an interactive session. Here: the builder must leave the screen
    alone (Blender's screen.area_close does not return in background mode)."""
    ta.ensure_registered()
    window = bpy.context.window_manager.windows[0]
    ws = window.workspace
    before = [a.type for a in window.screen.areas]
    ws[workspace.PHASE_KEY] = 0
    assert workspace.build_now(window) and ws[workspace.PHASE_KEY] == workspace.DONE
    assert [a.type for a in window.screen.areas] == before
    del ws[workspace.PHASE_KEY]
    # stale rectangles (as right after a split) are recognised
    class A:
        def __init__(self, x, y, w, h):
            self.x, self.y, self.width, self.height = x, y, w, h

    class S:
        areas = [A(0, 0, 100, 100), A(0, 0, 100, 40)]

    assert not workspace.areas_measured(S)
    S.areas = [A(0, 42, 100, 58), A(0, 0, 100, 40)]
    assert workspace.areas_measured(S)
    S.areas = [A(0, 42, 100, 58), A(0, 0, 0, 0)]
    assert not workspace.areas_measured(S)


def _ortho(left, right, bottom, top, near=-100.0, far=100.0):
    return Matrix(((2.0 / (right - left), 0, 0, -(right + left) / (right - left)),
                   (0, 2.0 / (top - bottom), 0, -(top + bottom) / (top - bottom)),
                   (0, 0, -2.0 / (far - near), -(far + near) / (far - near)),
                   (0, 0, 0, 1)))


def _render(width, height, projection, fn):
    import gpu
    off = gpu.types.GPUOffScreen(width, height)
    try:
        with off.bind():
            fb = gpu.state.active_framebuffer_get()
            fb.clear(color=(0.0, 0.0, 0.0, 0.0))
            with gpu.matrix.push_pop(), gpu.matrix.push_pop_projection():
                gpu.matrix.load_matrix(Matrix.Identity(4))
                gpu.matrix.load_projection_matrix(projection)
                gpu.state.blend_set("ALPHA")
                fn()
                gpu.state.blend_set("NONE")
            buf = fb.read_color(0, 0, width, height, 4, 0, "UBYTE")
        buf.dimensions = width * height * 4
        return np.array(buf.to_list(), dtype=np.uint8).reshape(height, width, 4)
    finally:
        off.free()


def test_overlays_render():
    import gpu
    try:
        gpu.init()
    except Exception as exc:                                       # no GPU for background processes on this machine
        print(f"   SKIPPED (no GPU in background mode: {exc})")
        return
    rig, gt, st = ta.make_scene(n_frames=70)
    data = rig.mocap_cleanup
    bpy.ops.mocap.analyze()
    bpy.ops.mocap.detect_contacts()
    assert len(data.issues) and len(data.pins) and raw_snapshot.get_raw_snapshot(rig) is not None
    W, H = 900, 300

    class V2D:
        def view_to_region(self, x, y, clip=False):
            return (10.0 + 12.0 * x, 150.0 + 60.0 * y)

        def region_to_view(self, x, y):
            return ((x - 10.0) / 12.0, (y - 150.0) / 60.0)

    class Region:
        width, height, view2d, type = W, H, V2D(), "WINDOW"

    region = Region()
    pixels = _ortho(0, W, 0, H)
    ctx = bpy.context

    # 1. error marks: a tinted column and a solid strip at the bottom, in the colour of the issue
    img = _render(W, H, pixels, lambda: draw._draw_marks(ctx, region, rig, st))
    issue = max(data.issues, key=lambda i: i.frame_end - i.frame_start)
    x = int(10 + 12 * 0.5 * (issue.frame_start + issue.frame_end))
    strip, column = img[2, x].astype(int), img[H // 2, x].astype(int)
    want = np.array(draw.ISSUE_COLORS[issue.kind])
    assert strip[3] > 100 and column[3] > 10 and strip[3] > 3 * column[3], (strip, column)
    assert np.argmax(strip[:3]) == np.argmax(want), (strip, want)
    assert img[H // 2, 5, 3] == 0                                  # nothing before the first frame

    # 2. pin tracks: every bar is drawn in the colour of its mode
    img = _render(W, H, pixels, lambda: draw._draw_tracks(ctx, region, rig))
    lay = pin_tracks.layout(ctx, region, rig)
    seen = 0
    for bar in lay["bars"]:
        cx, cy = int(0.5 * (bar["xi"] + bar["xo"])), int(0.5 * (bar["y0"] + bar["y1"]))
        if 0 <= cx < W and bar["xo"] - bar["xi"] > 6:
            r, g, b, a = img[cy, cx].astype(int)
            assert g > r and g > b and a > 150, (bar["index"], img[cy, cx])           # floor pins are green
            seen += 1
    assert seen >= 3
    assert img[H - 5, W // 2, 3] == 0                              # the rest of the editor stays free

    # 3. ghost of the raw take behind an edited curve
    cb = action_io.get_active_channelbag(rig)
    curve = cb.fcurves.find(action_io.bone_data_path("foot_ik.L", "location"), index=2)
    img = _render(W, H, pixels, lambda: draw._draw_ghosts(ctx, region, rig, curves=[curve]))
    lit = int((img[..., 3] > 0).sum())
    assert lit > 300, lit

    # 4. viewport HUD (top view): floor grid, sole outlines, pins, centre of mass, capsules
    st.show_capsules = True
    bpy.context.scene.frame_set(gt["plants"]["L"][1][0] + 3)
    hud = draw.hud_state(ctx, rig)
    com = hud["com"]
    half = 1.6
    top = _ortho(com.x - half, com.x + half, com.y - half * H / W, com.y + half * H / W)
    img = _render(W, H, top, lambda: draw.draw_hud(ctx, rig, st))
    lit_hud = int((img[..., 3] > 0).sum())
    assert lit_hud > 1500, lit_hud
    px, py = W // 2, H // 2                                        # the centre of mass is in the middle of the image
    assert (img[py - 12:py + 12, px - 12:px + 12, 3] > 0).sum() > 30
    st.show_floor = st.show_com = st.show_capsules = False
    img = _render(W, H, top, lambda: draw.draw_hud(ctx, rig, st))
    assert 0 < int((img[..., 3] > 0).sum()) < lit_hud
    st.show_floor = st.show_com = True
    print(f"   offscreen: marks, {seen} pin bars, ghost curve ({lit} px), HUD ({lit_hud} px)")


def test_editor_track_rows():
    """The MoCap Editor's problem track puts each bone in its body-part row; its operators are registered."""
    from mocap_cleanup.overlays import editor_track
    from mocap_cleanup.ui import common
    rig, _gt, _st = ta.make_scene()
    profile = common.profile_of(rig)
    expect = {"f_index.01.L": "hand.L", "thumb.01.R": "hand.R", "neck": "head", "head": "head", "torso": "torso",
              "foot_ik.R": "leg.R", "thigh_fk.L": "leg.L", "upper_arm_fk.L": "arm.L", "hand_ik.R": "arm.R"}
    for bone, group in expect.items():
        if rig.pose.bones.get(bone) is None:
            continue
        assert editor_track.group_of(profile, bone) == group, (bone, editor_track.group_of(profile, bone))
    assert [g for g, _l in editor_track.GROUPS][0] == "head" and len(editor_track.GROUPS) == 9
    for op in ("editor_open", "editor_track_click"):
        assert hasattr(bpy.ops.mocap, op) and getattr(bpy.ops.mocap, op).get_rna_type() is not None


def test_unused_controls_are_left_out():
    """A problem on a control that does not move the character there (the FK controls of an arm in IK) is not
    counted, not drawn and skipped by [ and ]; with Show Unused Controls it is listed and drawn dimmed (GUI
    feedback: the arm IK poles of a take whose arms are in FK filled the track with flips)."""
    from mocap_cleanup.overlays import editor_track
    rig, gt, st = ta.make_scene(n_frames=90)
    f0, f1 = gt["f0"], gt["f1"]
    ch = action_io.read_bone_local_channels(rig, "upper_arm_fk.L", f0, f1)       # the arm is in IK
    q = ch["quat"].copy()
    q[40] = (0.7, 0.7, 0.0, 0.14)
    action_io.write_bone_local_channels(rig, "upper_arm_fk.L", f0, quat=q / np.linalg.norm(q, axis=1, keepdims=True))
    assert bpy.ops.mocap.analyze() == {"FINISHED"}
    data = rig.mocap_cleanup
    idle = [i for i, it in enumerate(data.issues) if it.idle]
    live = [i for i, it in enumerate(data.issues) if not it.idle]
    assert idle and live
    hit = [data.issues[i] for i in idle if data.issues[i].bone == "upper_arm_fk.L"]
    assert hit and hit[0].frame_start <= f0 + 40 <= hit[0].frame_end, [(it.bone, it.kind) for it in data.issues]
    assert "- not in use here: the arm is in IK" in hit[0].message
    assert all(data.issues[i].bone.endswith("_fk.L") or data.issues[i].bone.endswith("_fk.R") for i in idle)
    assert any(data.issues[i].bone == "hand_ik.L" for i in live)                 # the spike that is seen
    assert data.analysis_summary.endswith(f", {len(idle)} more on unused controls"), data.analysis_summary

    class V2D:
        def region_to_view(self, x, y):
            return (f0 - 2 + x * (gt["T"] + 4) / 1000.0, y)

    class Region:
        view2d, width, height = V2D(), 1000, 400

    def bands():
        return {b["index"]: b for b in editor_track.layout(bpy.context, Region(), rig)["bands"]}

    st.show_idle_issues = False
    shown = bands()
    assert not (set(idle) & set(shown)) and set(live) <= set(shown)
    st.stage = "DIAGNOSE"
    stats = draw_panel(panels._MainPanel)
    assert "show_idle_issues" in stats["props"]
    # [ and ] only visit what is seen
    visited = set()
    bpy.context.scene.frame_set(f0)
    for _ in range(len(live) + 2):
        assert bpy.ops.mocap.issue_jump(direction=1) == {"FINISHED"}
        visited.add(data.active_issue)
    assert visited <= set(live)
    st.show_idle_issues = True
    shown = bands()
    assert set(idle) <= set(shown) and all(shown[i]["idle"] for i in idle)
    st.show_idle_issues = False


def test_view_mapping_is_exact():
    """Tracks and marks sit on their frames at any zoom and pan (GUI feedback: they slid, collapsed onto
    frame 0 or vanished, because the scale came from two neighbouring frames in whole pixels)."""

    class V2D:
        def __init__(self, left, right, bottom, top, w, h):
            self.l, self.r, self.b, self.t, self.w, self.h = left, right, bottom, top, w, h

        def region_to_view(self, x, y):
            return (self.l + x * (self.r - self.l) / self.w, self.b + y * (self.t - self.b) / self.h)

        def view_to_region(self, x, y, clip=True):            # Blender returns whole pixels
            return (int((x - self.l) * self.w / (self.r - self.l)), int((y - self.b) * self.h / (self.t - self.b)))

    class Region:
        def __init__(self, v2d):
            self.view2d, self.width, self.height = v2d, v2d.w, v2d.h

    for left, right in ((-596.3, 3245.1), (952.7, 2534.2), (1400.25, 1700.5)):   # zoomed out, mid, zoomed in
        region = Region(V2D(left, right, -1.3, 2.7, 1012, 300))
        x0, sx, y0, sy = pin_tracks.view_mapping(region)
        for f in (0.0, 1473.0, 1492.5, 3000.0):
            exact = (f - left) * region.width / (right - left)
            assert abs(x0 + sx * f - exact) < 1e-6, (left, right, f, x0 + sx * f, exact)
        assert abs(y0 + sy * 0.5 - (0.5 + 1.3) * 300 / 4.0) < 1e-6
    # what it replaced: two neighbouring frames in whole pixels give 0 px per frame when zoomed out
    region = Region(V2D(-596.3, 3245.1, -1.3, 2.7, 1012, 300))
    old = region.view2d.view_to_region(1.0, 0.0)[0] - region.view2d.view_to_region(0.0, 0.0)[0]
    assert old in (0, 1)


if __name__ == "__main__":
    run_tests([test_every_panel_layout, test_workspace_layout, test_overlays_render, test_view_mapping_is_exact,
               test_editor_track_rows, test_unused_controls_are_left_out])
