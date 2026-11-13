"""The MoCap Editor: a window of its own for finding and fixing problems (GUI feedback: the N-panel alone is
too small to see where the problems are, and the marks did not stay on their frames).

Top: the problem track - one row per body part with a coloured band wherever Analyze found something (click a
band: the playhead goes there, its bone is selected and the Graph Editor frames its curves), the pin rows below
it. Bottom: the Graph Editor with the curves of the selected bones, the problem frames shaded and the raw take
as a dashed ghost; its sidebar holds the cleanup steps in order.

It is built from Blender's own editors (a Dope Sheet that shows no channels is the canvas of the track), so
scrubbing, zoom, pan and the theme are Blender's. The window is made with ``screen.area_dupli`` and laid out
by a timer, like the workspace (see ``workspace`` for why every screen edit runs inside ``temp_override``).
"""

from __future__ import annotations

import bpy
import numpy as np
from bpy.types import Operator

from ..core import action_io
from ..overlays import editor_track, pin_tracks
from . import common
from .workspace import _by_pointer, _ctx, _set_editor, _size, _split, _window_region, areas_measured

EDITOR_KEY = "mocap_editor"           # custom property on the editor window's screen
PHASE_KEY = "mocap_editor_phase"
DONE = 3
NO_CHANNELS = "⁣mocap-editor⁣"   # Dope Sheet filter that matches no channel: an empty canvas
_pending: list[int] = []              # window pointers still being laid out
_mem: dict[int, dict[str, int]] = {}
_waits: dict[int, int] = {}


def is_editor_screen(screen) -> bool:
    return screen is not None and bool(screen.get(EDITOR_KEY, False))


def editor_window(context=None):
    wm = (context or bpy.context).window_manager
    return next((w for w in wm.windows if is_editor_screen(w.screen)), None)


def _window_by_pointer(ptr):
    return next((w for w in bpy.context.window_manager.windows if w.as_pointer() == ptr), None)


def _configure_track(window, area) -> None:
    _set_editor(window, area, "DOPESHEET", "DOPESHEET_EDITOR")
    with _ctx(window, area):
        space = area.spaces.active
        for attr, value in (("show_region_channels", False), ("show_region_ui", False), ("show_region_hud", False),
                            ("show_region_footer", False)):
            try:
                setattr(space, attr, value)
            except Exception:
                pass
        ds = getattr(space, "dopesheet", None)
        if ds is not None:
            ds.filter_text = NO_CHANNELS
            for attr in ("show_summary", "show_only_errors"):
                try:
                    setattr(ds, attr, False)
                except Exception:
                    pass


def _configure_graph(window, area) -> None:
    _set_editor(window, area, "FCURVES", "GRAPH_EDITOR")
    with _ctx(window, area):
        space = area.spaces.active
        for attr, value in (("use_normalization", False), ("use_only_selected", True), ("show_region_ui", True),
                            ("show_handles", False)):          # a key on every frame: handles are only clutter
            try:
                setattr(space, attr, value)
            except Exception:
                pass
        ds = getattr(space, "dopesheet", None)
        if ds is not None:
            # A problem can sit on a bone in a hidden collection (the IK pole of an arm that runs in FK):
            # without this, picking it showed an empty Graph Editor.
            for attr, value in (("show_only_selected", True), ("show_hidden", True)):
                try:
                    setattr(ds, attr, value)
                except Exception:
                    pass


def _show_tab(window, area) -> None:
    for region in area.regions:
        if region.type == "UI":
            try:
                with _ctx(window, area):
                    region.active_panel_category = "MoCap"
            except Exception:
                pass
    area.tag_redraw()


def build_step(window, measured: bool) -> int:
    """One step of laying out the editor window; returns the next phase."""
    screen = window.screen
    ptr = window.as_pointer()
    phase = int(screen.get(PHASE_KEY, 0))
    mem = _mem.setdefault(ptr, {})
    if phase == 0:
        main = max(screen.areas, key=_size)
        new = _split(window, main, 0.58)              # factor > 0.5: the new area is above the line
        mem.update(graph=main.as_pointer(), track=new.as_pointer() if new is not None else 0)
        phase = 1
    elif phase == 1:
        graph, track = _by_pointer(screen, mem.get("graph")), _by_pointer(screen, mem.get("track"))
        areas = [a for a in (graph, track) if a is not None]
        if measured and len(areas) == 2:
            track, graph = sorted(areas, key=lambda a: -a.y)       # the track is on top
        if graph is not None:
            _configure_graph(window, graph)
        if track is not None:
            _configure_track(window, track)
        mem.update(graph=graph.as_pointer() if graph else 0)
        phase = 2
    elif phase == 2:
        graph = _by_pointer(screen, mem.get("graph"))
        if graph is not None:
            _show_tab(window, graph)
        phase = DONE
    screen[PHASE_KEY] = phase
    return phase


def _build_tick():
    again = False
    for ptr in list(_pending):
        window = _window_by_pointer(ptr)
        if window is None or window.screen is None:
            _pending.remove(ptr)
            continue
        screen = window.screen
        measured = areas_measured(screen)
        if not measured and _waits.get(ptr, 0) < 15:
            _waits[ptr] = _waits.get(ptr, 0) + 1
            for a in screen.areas:
                a.tag_redraw()
            again = True
            continue
        _waits[ptr] = 0
        try:
            phase = build_step(window, measured)
        except Exception:
            import traceback
            traceback.print_exc()
            phase = DONE
        if phase >= DONE:
            _pending.remove(ptr)
            _mem.pop(ptr, None)
        else:
            again = True
    return 0.08 if again or _pending else None


def frame_graph_on_issue(context, rig, issue, margin: int = 15) -> None:
    """Frame the Graph Editors of the editor window on the issue's curves around its frames."""
    window = editor_window(context)
    if window is None or not issue.bone:
        return
    lo, hi = issue.frame_start - margin, issue.frame_end + margin
    prefix = f'pose.bones["{issue.bone}"]'
    # The curves that jump at the issue decide the value range: a 5 mm spike on a foot is invisible
    # next to its 2 m height and its rotation (score: largest second difference around the issue).
    mine = []
    for fc in action_io.iter_fcurves(rig):
        kps = fc.keyframe_points
        n = len(kps)
        if not n:
            continue
        off = [False] * n
        kps.foreach_set("select_left_handle", off)        # framing counts selected handles too
        kps.foreach_set("select_right_handle", off)
        kps.foreach_set("select_control_point", off)
        if fc.data_path.startswith(prefix):
            co = np.empty(2 * n)
            kps.foreach_get("co", co)
            x, v = co[0::2], co[1::2]
            inside = (x >= lo) & (x <= hi)
            if inside.sum() >= 3:
                near = (x >= issue.frame_start - 3) & (x <= issue.frame_end + 3)
                d2 = np.zeros(n)
                d2[1:-1] = np.abs(v[2:] - 2.0 * v[1:-1] + v[:-2])
                score = float(d2[near].max()) if near.any() else 0.0
                mine.append((score, fc, inside, v))
    if not mine:
        return
    top = max(s for s, *_ in mine)
    picked = [m for m in mine if m[0] >= 0.3 * top] or mine
    vmin = min(float(v[inside].min()) for _s, _fc, inside, v in picked)
    vmax = max(float(v[inside].max()) for _s, _fc, inside, v in picked)
    for _s, fc, inside, _v in picked:
        fc.keyframe_points.foreach_set("select_control_point", inside.tolist())   # highlighted in orange
    pad = max(0.12 * (vmax - vmin), 1e-3)
    vmin, vmax = vmin - pad, vmax + pad
    # The view is set explicitly: graph.view_selected kept the whole take in view here.
    for area in window.screen.areas:
        if area.type != "GRAPH_EDITOR":
            continue
        region = _window_region(area)
        if region is None:
            continue
        x0, sx, y0, sy = pin_tracks.view_mapping(region)
        rect = dict(xmin=int(x0 + sx * lo), xmax=int(x0 + sx * hi), ymin=int(y0 + sy * vmin), ymax=int(y0 + sy * vmax))
        try:
            with context.temp_override(window=window, screen=window.screen, area=area, region=region):
                bpy.ops.view2d.zoom_border(wait_for_input=False, zoom_out=False, **rect)
        except Exception:
            try:
                with context.temp_override(window=window, screen=window.screen, area=area, region=region):
                    bpy.ops.graph.view_selected()
            except Exception:
                pass
        area.tag_redraw()


class MOCAP_OT_editor_open(Operator):
    bl_idname = "mocap.editor_open"
    bl_label = "Open MoCap Editor"
    bl_description = ("Open a window with the problems over time (one row per body part), the curves of what you "
                      "pick, the pins and the cleanup steps in order")
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return context.window is not None and not bpy.app.background

    def execute(self, context):
        if editor_window(context) is not None:
            self.report({"INFO"}, "The MoCap Editor is already open")
            return {"FINISHED"}
        window = context.window
        area = context.area if context.area is not None else max(window.screen.areas, key=_size)
        wm = context.window_manager
        before = {w.as_pointer() for w in wm.windows}
        try:
            with context.temp_override(window=window, screen=window.screen, area=area):
                bpy.ops.screen.area_dupli("INVOKE_DEFAULT")
        except Exception as exc:
            self.report({"ERROR"}, f"Could not open a window: {exc}")
            return {"CANCELLED"}
        new = [w for w in wm.windows if w.as_pointer() not in before]
        if not new:
            self.report({"ERROR"}, "Could not open a window")
            return {"CANCELLED"}
        win = new[0]
        win.screen[EDITOR_KEY] = True
        win.screen[PHASE_KEY] = 0
        _pending.append(win.as_pointer())
        if not bpy.app.timers.is_registered(_build_tick):
            bpy.app.timers.register(_build_tick, first_interval=0.1)
        rig = common.get_rig(context)
        if rig is not None and not len(rig.mocap_cleanup.issues) and not rig.mocap_cleanup.analysis_summary:
            try:
                bpy.ops.mocap.analyze()
            except Exception:
                pass
        return {"FINISHED"}


class MOCAP_OT_editor_track_click(Operator):
    bl_idname = "mocap.editor_track_click"
    bl_label = "Pick Problem"
    bl_description = "Click a band in the problem track: go there, select its bone and frame its curves"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    @classmethod
    def poll(cls, context):
        return (context.area is not None and context.area.type == "DOPESHEET_EDITOR"
                and is_editor_screen(context.screen) and common.get_rig(context) is not None)

    def invoke(self, context, event):
        region = context.region
        if region is None or region.type != "WINDOW":
            return {"PASS_THROUGH"}
        rig = common.get_rig(context)
        lay = editor_track.layout(context, region, rig)
        idx = editor_track.hit_test(lay, event.mouse_region_x, event.mouse_region_y, 3.0 * pin_tracks.ui_scale(context))
        if idx < 0:
            return {"PASS_THROUGH"}
        data = rig.mocap_cleanup
        data.active_issue = idx                        # goes there and selects the bone (common.focus_issue)
        issue = data.issues[idx]
        frame_graph_on_issue(context, rig, issue)
        self.report({"INFO"}, f"{editor_track.KIND_LABELS.get(issue.kind, issue.kind)}, frames "
                              f"{issue.frame_start}-{issue.frame_end}: {issue.message}")
        common.tag_redraw()
        return {"FINISHED"}


def unregister() -> None:
    if bpy.app.timers.is_registered(_build_tick):
        bpy.app.timers.unregister(_build_tick)
    _pending.clear()
    _mem.clear()
    _waits.clear()


CLASSES = (MOCAP_OT_editor_open, MOCAP_OT_editor_track_click)
