"""Pin & contact tracks in the Dope Sheet / Graph Editor: layout, hit testing and the drag operator.

Each limb that has pins gets a row at the bottom of the editor, above the marker line (the top rows of
a Dope Sheet hold the summary keys, which stay visible this way). A pin is drawn as a trapezoid::

        lock_in ______________ lock_out
               /              \\
        start /                \\ end        (ramps = ease in / ease out)

Drag the body to move a pin, its edges to change the locked span and the ramp feet to change the
blend lengths (Shift + edge works as well when a ramp has no length yet).
"""

from __future__ import annotations

import bpy
from bpy.types import Operator

from .. import props
from ..ui import autosolve, common

OWNER_ORDER = ("arm.L", "arm.R", "leg.L", "leg.R", "torso")
MODE_COLORS = {
    "FLOOR": (0.25, 0.80, 0.45),
    "WORLD": (0.30, 0.60, 1.00),
    "TARGET": (1.00, 0.62, 0.20),
}


def view_mapping(region) -> tuple[float, float, float, float]:
    """Exact view -> region mapping of a 2D editor: ``(x0, sx, y0, sy)``, region x = ``x0 + sx * frame``.

    ``View2D.view_to_region`` returns whole pixels, so a scale taken from two neighbouring frames came
    out as 0, 1 or 2 px depending on zoom and pan: tracks and marks slid along the time line, collapsed
    onto frame 0 or vanished (GUI feedback). ``region_to_view`` is exact; it is taken across the whole
    region.
    """
    v2d = region.view2d
    w, h = max(int(region.width), 1), max(int(region.height), 1)
    left, bottom = v2d.region_to_view(0, 0)
    right, top = v2d.region_to_view(w, h)
    sx = w / (right - left) if right != left else 1.0
    sy = h / (top - bottom) if top != bottom else 1.0
    return -left * sx, sx, -bottom * sy, sy


def ui_scale(context) -> float:
    s = float(context.preferences.system.ui_scale)
    return s if s > 0.01 else 1.0          # (0 in background mode)


def tracks_enabled(context, st) -> bool:
    """Are pin tracks shown (and editable) in the editor of ``context``?"""
    area = context.area
    if area is None or st is None:
        return False
    screen = getattr(context, "screen", None)
    if screen is not None and screen.get("mocap_editor", False):
        return area.type == "DOPESHEET_EDITOR"       # the MoCap Editor shows its pins in the track, always
    if not st.show_pin_tracks:
        return False
    if area.type == "GRAPH_EDITOR":
        return bool(st.show_pin_tracks_graph)
    if area.type == "DOPESHEET_EDITOR":
        return getattr(area, "ui_type", "") != "TIMELINE"
    return False


def layout(context, region, rig) -> dict:
    """Pixel layout of the tracks in ``region``: rows and pin bars.

    Normally the rows sit at the bottom of the editor. In the MoCap Editor they follow the problem rows
    (``editor_track.pins_top``), so the two never cover each other."""
    data = rig.mocap_cleanup
    s = ui_scale(context)
    row_h = 17.0 * s
    gap = 2.0 * s
    owners = [o for o in OWNER_ORDER if any(p.owner == o for p in data.pins)]
    scene = getattr(context, "scene", None)
    has_markers = scene is not None and len(scene.timeline_markers) > 0
    base = (44.0 if has_markers else 16.0) * s                 # scrollbar (+ marker line)
    top = base + len(owners) * (row_h + gap) - gap
    screen = getattr(context, "screen", None)
    if screen is not None and screen.get("mocap_editor", False):
        from . import editor_track
        top = editor_track.pins_top(context, region, rig)
    x0, per_frame, _y0, _sy = view_mapping(region)

    def fx(frame: float) -> float:
        return x0 + per_frame * frame

    rows, bars = [], []
    for i, owner in enumerate(owners):
        y1 = top - i * (row_h + gap)
        y0 = y1 - row_h
        rows.append((owner, y0, y1))
        for idx, p in enumerate(data.pins):
            if p.owner != owner:
                continue
            bars.append({
                "index": idx, "owner": owner, "y0": y0, "y1": y1,
                "xs": fx(p.lock_in - p.blend_in - 0.5), "xi": fx(p.lock_in - 0.5),
                "xo": fx(p.lock_out + 0.5), "xe": fx(p.lock_out + p.blend_out + 0.5),
                "mode": p.mode, "enabled": p.enabled, "auto": p.auto,
            })
    return {"rows": rows, "bars": bars, "per_frame": per_frame, "label_w": 62.0 * s, "x0": x0}


def hit_test(lay: dict, mx: float, my: float, grab: float = 6.0):
    """``(bar, zone)`` under the mouse; zone in MOVE / IN / OUT / BLEND_IN / BLEND_OUT."""
    best = None
    for bar in lay["bars"]:
        if not (bar["y0"] <= my <= bar["y1"]):
            continue
        if not (bar["xs"] - grab <= mx <= bar["xe"] + grab):
            continue
        if bar["xi"] - bar["xs"] > 2 * grab and abs(mx - bar["xs"]) <= grab:
            zone = "BLEND_IN"
        elif bar["xe"] - bar["xo"] > 2 * grab and abs(mx - bar["xe"]) <= grab:
            zone = "BLEND_OUT"
        elif abs(mx - bar["xi"]) <= grab:
            zone = "IN"
        elif abs(mx - bar["xo"]) <= grab:
            zone = "OUT"
        elif mx < bar["xi"]:
            zone = "BLEND_IN"
        elif mx > bar["xo"]:
            zone = "BLEND_OUT"
        else:
            zone = "MOVE"
        # prefer edges of narrow bars over the body of a neighbour
        rank = 0 if zone != "MOVE" else 1
        if best is None or rank < best[2]:
            best = (bar, zone, rank)
    return (best[0], best[1]) if best else (None, "")


class MOCAP_OT_pin_track_edit(Operator):
    bl_idname = "mocap.pin_track_edit"
    bl_label = "Edit Pin Track"
    bl_description = "Drag pins in the track overlay: body = move, edges = locked span, ramps = ease in / out"
    bl_options = {"REGISTER", "UNDO", "INTERNAL"}

    @classmethod
    def poll(cls, context):
        st = getattr(context.scene, "mocap_cleanup", None)
        if not tracks_enabled(context, st):
            return False
        rig = common.get_rig(context)
        return rig is not None and len(rig.mocap_cleanup.pins) > 0

    def invoke(self, context, event):
        region = context.region
        if region is None or region.type != "WINDOW":
            return {"PASS_THROUGH"}
        rig = common.get_rig(context)
        lay = layout(context, region, rig)
        bar, zone = hit_test(lay, event.mouse_region_x, event.mouse_region_y, 6.0 * ui_scale(context))
        if bar is None:
            return {"PASS_THROUGH"}
        data = rig.mocap_cleanup
        data.active_pin = bar["index"]
        pin = data.pins[bar["index"]]
        if event.shift and zone in {"IN", "OUT"}:
            zone = "BLEND_IN" if zone == "IN" else "BLEND_OUT"
        self._rig_name = rig.name
        self._index = bar["index"]
        self._zone = zone
        self._start = (pin.lock_in, pin.lock_out, pin.blend_in, pin.blend_out)
        self._mouse_x = event.mouse_region_x
        self._per_frame = max(lay["per_frame"], 1e-6)
        self._moved = False
        context.window_manager.modal_handler_add(self)
        context.window.cursor_modal_set("SCROLL_X")
        return {"RUNNING_MODAL"}

    def _apply(self, pin, delta: int) -> None:
        li, lo, bi, bo = self._start
        with props.suspend_updates():
            if self._zone == "MOVE":
                pin.lock_in, pin.lock_out = li + delta, lo + delta
            elif self._zone == "IN":
                pin.lock_in = min(li + delta, lo)
                pin.blend_in = max(bi - (pin.lock_in - li), 0) if bi else 0
            elif self._zone == "OUT":
                pin.lock_out = max(lo + delta, li)
                pin.blend_out = max(bo + (lo - pin.lock_out), 0) if bo else 0
            elif self._zone == "BLEND_IN":
                pin.blend_in = max(bi - delta, 0)
            elif self._zone == "BLEND_OUT":
                pin.blend_out = max(bo + delta, 0)

    def modal(self, context, event):
        rig = bpy.data.objects.get(self._rig_name)
        if rig is None or self._index >= len(rig.mocap_cleanup.pins):
            context.window.cursor_modal_restore()
            return {"CANCELLED"}
        pin = rig.mocap_cleanup.pins[self._index]
        if event.type == "MOUSEMOVE":
            delta = int(round((event.mouse_region_x - self._mouse_x) / self._per_frame))
            if delta != 0:
                self._moved = True
            self._apply(pin, delta)
            context.area.header_text_set(
                f"Pin {pin.lock_in - pin.blend_in} / {pin.lock_in} - {pin.lock_out} / {pin.lock_out + pin.blend_out}")
            context.area.tag_redraw()
            return {"RUNNING_MODAL"}
        if event.type in {"RIGHTMOUSE", "ESC"}:
            with props.suspend_updates():
                pin.lock_in, pin.lock_out, pin.blend_in, pin.blend_out = self._start
            self._finish(context)
            return {"CANCELLED"}
        if event.type == "LEFTMOUSE" and event.value == "RELEASE":
            self._finish(context)
            if self._moved:
                autosolve.schedule(rig)
            return {"FINISHED"}
        return {"RUNNING_MODAL"}

    def _finish(self, context) -> None:
        context.window.cursor_modal_restore()
        if context.area is not None:
            context.area.header_text_set(None)
        common.tag_redraw()
