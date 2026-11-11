"""Draw handlers: error marks, raw ghost curves and pin tracks in the animation editors, HUD in the viewport.

Handlers never raise: an overlay must not be able to break the editor it is drawn in.
"""

from __future__ import annotations

import math

import blf
import bpy
import gpu
import numpy as np
from gpu_extras.batch import batch_for_shader
from mathutils import Vector

from ..core import action_io, raw_snapshot, runtime
from ..core.rig_inspector import sole_contacts
from ..ui import common
from . import pin_tracks

ISSUE_COLORS = {
    "FLIP": (0.95, 0.22, 0.20),
    "SPIKE": (1.00, 0.33, 0.10),
    "KNEE_POP": (0.95, 0.20, 0.60),
    "JITTER": (1.00, 0.76, 0.12),
    "HYPEREXT": (1.00, 0.52, 0.12),
    "FROZEN": (0.20, 0.80, 0.95),
    "PENETRATION": (0.62, 0.45, 1.00),
    "SKATE": (0.62, 0.90, 0.22),
}
_handles: list[tuple] = []
_reported: set[str] = set()


def _report(where: str) -> None:
    """Print a draw error once per kind (a draw handler must never raise)."""
    import traceback
    text = traceback.format_exc()
    key = where + text[-200:]
    if key not in _reported:
        _reported.add(key)
        print(f"MoCap Cleanup: {where} overlay failed\n{text}")


# ---------------------------------------------------------------------------
# Tiny GPU helpers
# ---------------------------------------------------------------------------

def _tris(verts, color) -> None:
    """Filled triangles from a flat vertex list (3 per triangle)."""
    if not verts:
        return
    sh = gpu.shader.from_builtin("UNIFORM_COLOR")
    batch = batch_for_shader(sh, "TRIS", {"pos": verts})
    sh.bind()
    sh.uniform_float("color", color)
    batch.draw(sh)


def _rects(rects, color) -> None:
    verts = []
    for x0, y0, x1, y1 in rects:
        verts += [(x0, y0), (x1, y0), (x1, y1), (x0, y0), (x1, y1), (x0, y1)]
    _tris(verts, color)


def _lines(points, color, width: float = 1.5) -> None:
    """Line segments (pairs of points, 2-D or 3-D)."""
    if len(points) < 2:
        return
    pts = [(p[0], p[1], p[2] if len(p) > 2 else 0.0) for p in points]
    sh = gpu.shader.from_builtin("POLYLINE_UNIFORM_COLOR")
    batch = batch_for_shader(sh, "LINES", {"pos": pts})
    sh.bind()
    sh.uniform_float("viewportSize", gpu.state.viewport_get()[2:])
    sh.uniform_float("lineWidth", width)
    sh.uniform_float("color", color)
    batch.draw(sh)


def _strip_to_lines(points):
    out = []
    for a, b in zip(points[:-1], points[1:]):
        out += [a, b]
    return out


def _text(x: float, y: float, text: str, size: float, color) -> None:
    blf.size(0, size)
    blf.color(0, *color)
    blf.position(0, x, y, 0)
    blf.draw(0, text)


# ---------------------------------------------------------------------------
# Animation editors
# ---------------------------------------------------------------------------

def _visible_issue(issue, selected: set[str] | None) -> bool:
    return selected is None or not issue.bone or issue.bone in selected


def _draw_marks(context, region, rig, st) -> None:
    data = rig.mocap_cleanup
    if not len(data.issues):
        return
    v2d = region.view2d
    x0 = v2d.view_to_region(0.0, 0.0, clip=False)[0]
    per_frame = v2d.view_to_region(1.0, 0.0, clip=False)[0] - x0
    selected = set(action_io.selected_pose_bone_names(rig)) if st.marks_only_selected else None
    s = pin_tracks.ui_scale(context)
    h = region.height
    by_kind: dict[str, list] = {}
    strong: dict[str, list] = {}
    active = data.active_issue
    outline = None
    for i, issue in enumerate(data.issues):
        if not _visible_issue(issue, selected):
            continue
        xa = x0 + per_frame * (issue.frame_start - 0.5)
        xb = x0 + per_frame * (issue.frame_end + 0.5)
        if xb < 0 or xa > region.width:
            continue
        xb = max(xb, xa + 2.0)
        by_kind.setdefault(issue.kind, []).append((xa, 0.0, xb, h))
        strong.setdefault(issue.kind, []).append((xa, 0.0, xb, 5.0 * s))
        if i == active:
            outline = (xa, xb)
    stale = 0.55 if data.analysis_stale else 1.0
    for kind, rects in by_kind.items():
        c = ISSUE_COLORS.get(kind, (1.0, 1.0, 1.0))
        _rects(rects, (*c, 0.13 * stale))
        _rects(strong[kind], (*c, 0.95 * stale))
    if outline is not None:
        xa, xb = outline
        _lines([(xa, 0), (xa, h), (xb, 0), (xb, h)], (1.0, 1.0, 1.0, 0.55), 1.0)


def _draw_ghosts(context, region, rig, curves=None) -> None:
    """Dashed raw-take curves behind the visible F-Curves (``curves``: default = what the editor shows)."""
    space = context.space_data
    if getattr(space, "use_normalization", False):
        return
    snap = raw_snapshot.get_raw_snapshot(rig)
    act = action_io.get_active_action(rig)
    if snap is None or act is None or snap == act:
        return
    cb_raw = action_io.get_active_channelbag(rig, snap)
    if cb_raw is None:
        return
    if curves is None:
        curves = list(getattr(context, "selected_visible_fcurves", None) or [])
        if not curves:
            curves = list(getattr(context, "visible_fcurves", None) or [])
    curves = [fc for fc in curves if fc.id_data == act][:16]
    if not curves:
        return
    v2d = region.view2d
    ox, oy = v2d.view_to_region(0.0, 0.0, clip=False)
    px, py = v2d.view_to_region(1.0, 1.0, clip=False)
    sx, sy = px - ox, py - oy
    left = v2d.region_to_view(0, 0)[0]
    right = v2d.region_to_view(region.width, 0)[0]
    degrees = context.scene.unit_settings.system_rotation == "DEGREES"
    for fc in curves:
        key = (snap.name, fc.data_path, fc.array_index)
        cached = runtime.GHOSTS.get(key)
        if cached is None:
            raw_fc = cb_raw.fcurves.find(fc.data_path, index=fc.array_index)
            if raw_fc is None or len(raw_fc.keyframe_points) == 0:
                runtime.GHOSTS[key] = (0, None)
                continue
            f0, f1 = action_io.get_action_frame_range(rig, snap)
            cached = (f0, action_io.sample_fcurve_range(raw_fc, f0, f1))
            if len(runtime.GHOSTS) > 4000:
                runtime.GHOSTS.clear()
            runtime.GHOSTS[key] = cached
        f0, vals = cached
        if vals is None:
            continue
        a = max(int(math.floor(left)) - f0, 0)
        b = min(int(math.ceil(right)) - f0, len(vals) - 1)
        if b - a < 1:
            continue
        step = max(1, int((b - a) / max(region.width, 1)))
        idx = np.arange(a, b + 1, step)
        fac = 57.29577951308232 if (degrees and fc.data_path.endswith("rotation_euler")) else 1.0
        xs = ox + sx * (idx + f0)
        ys = oy + sy * vals[idx] * fac
        pts = list(zip(xs.tolist(), ys.tolist()))
        c = tuple(fc.color) if fc.color_mode != "AUTO_RAINBOW" or max(fc.color) > 0 else (0.8, 0.8, 0.8)
        # dashed: every other segment
        segs = []
        for i in range(0, len(pts) - 1, 2):
            segs += [pts[i], pts[i + 1]]
        _lines(segs, (0.55 * c[0] + 0.3, 0.55 * c[1] + 0.3, 0.55 * c[2] + 0.3, 0.75), 1.6)


def _draw_tracks(context, region, rig) -> None:
    data = rig.mocap_cleanup
    if not len(data.pins):
        return
    lay = pin_tracks.layout(context, region, rig)
    s = pin_tracks.ui_scale(context)
    if not lay["rows"]:
        return
    _rects([(0.0, y0, region.width, y1) for _o, y0, y1 in lay["rows"]], (0.08, 0.08, 0.09, 0.62))
    groups: dict[tuple, list] = {}
    outlines = []
    for bar in lay["bars"]:
        if bar["xe"] < 0 or bar["xs"] > region.width:
            continue
        base = pin_tracks.MODE_COLORS.get(bar["mode"], (0.8, 0.8, 0.8))
        if not bar["enabled"]:
            col = (0.45, 0.45, 0.45, 0.55)
        elif bar["auto"]:
            col = (*base, 0.62)
        else:
            col = (*base, 0.95)
        y0, y1 = bar["y0"] + 2.0 * s, bar["y1"] - 2.0 * s
        xs, xi, xo, xe = bar["xs"], bar["xi"], bar["xo"], bar["xe"]
        xo = max(xo, xi + 2.0)
        verts = groups.setdefault(col, [])
        verts += [(xi, y0), (xo, y0), (xo, y1), (xi, y0), (xo, y1), (xi, y1)]
        if xi - xs > 0.5:
            verts += [(xs, y0), (xi, y0), (xi, y1)]
        if xe - xo > 0.5:
            verts += [(xo, y0), (xe, y0), (xo, y1)]
        if bar["index"] == data.active_pin:
            outlines += _strip_to_lines([(xs, y0), (xi, y1), (xo, y1), (xe, y0), (xs, y0)])
    for col, verts in groups.items():
        _tris(verts, col)
    if outlines:
        _lines(outlines, (1.0, 1.0, 1.0, 0.95), 1.5)
    # labels on a small plate at the left edge
    _rects([(0.0, y0, lay["label_w"], y1) for _o, y0, y1 in lay["rows"]], (0.08, 0.08, 0.09, 0.9))
    from ..props import OWNER_LABELS
    for owner, y0, y1 in lay["rows"]:
        _text(6.0 * s, y0 + 4.5 * s, OWNER_LABELS.get(owner, owner), 11.0 * s, (0.92, 0.92, 0.92, 1.0))


def draw_anim_overlays(context, region, rig, st, graph: bool) -> None:
    """Error marks, ghost curves (Graph Editor only) and pin tracks in pixel space of ``region``."""
    if st.show_heatmap:
        _draw_marks(context, region, rig, st)
    if st.show_ghost and graph:
        _draw_ghosts(context, region, rig)
    if pin_tracks.tracks_enabled(context, st):
        _draw_tracks(context, region, rig)


def draw_anim_editor() -> None:
    try:
        context = bpy.context
        st = getattr(context.scene, "mocap_cleanup", None)
        region = context.region
        if st is None or region is None or region.type != "WINDOW":
            return
        if not (st.show_heatmap or st.show_ghost or st.show_pin_tracks):
            return
        rig = common.get_rig(context)
        if rig is None:
            return
        gpu.state.blend_set("ALPHA")
        draw_anim_overlays(context, region, rig, st, context.area.type == "GRAPH_EDITOR")
        gpu.state.blend_set("NONE")
    except Exception:
        _report("animation editor")
        try:
            gpu.state.blend_set("NONE")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 3D viewport
# ---------------------------------------------------------------------------

def _circle(center, radius: float, normal=(0.0, 0.0, 1.0), n: int = 24):
    nrm = Vector(normal).normalized()
    a = nrm.orthogonal().normalized()
    b = nrm.cross(a)
    c = Vector(center)
    pts = [c + radius * (math.cos(2 * math.pi * i / n) * a + math.sin(2 * math.pi * i / n) * b) for i in range(n + 1)]
    return _strip_to_lines([tuple(p) for p in pts])


def _convex_hull_2d(points):
    pts = sorted(set((round(p[0], 6), round(p[1], 6)) for p in points))
    if len(pts) < 3:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _inside(poly, p) -> bool:
    if len(poly) < 3:
        return False
    sign = None
    for a, b in zip(poly, poly[1:] + poly[:1]):
        c = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(c) < 1e-12:
            continue
        if sign is None:
            sign = c > 0
        elif (c > 0) != sign:
            return False
    return True


def hud_state(context, rig) -> dict | None:
    """World-space HUD data for the current frame, read from the evaluated rig."""
    profile = common.profile_of(rig)
    st = context.scene.mocap_cleanup
    deps = context.evaluated_depsgraph_get()
    rig_eval = rig.evaluated_get(deps)
    mw = rig_eval.matrix_world
    bones = rig_eval.pose.bones
    frame = context.scene.frame_current
    unit = profile.unit
    out = {"unit": unit, "floor": st.floor_z, "soles": {}, "planted": {}, "pins": [], "com": None, "capsules": []}

    com = Vector((0.0, 0.0, 0.0))
    total = 0.0
    for name, mass, c in profile.com_segments:
        pb = bones.get(name)
        if pb is not None:
            com += mass * (mw @ (pb.head + c * (pb.tail - pb.head)))
            total += mass
    if total > 0.0:
        out["com"] = com / total

    data = rig.mocap_cleanup
    for leg in profile.legs.values():
        pb = bones.get(leg.org_end)
        if pb is None or not len(leg.sole_points):
            continue
        M = mw @ pb.matrix
        pts = [Vector(p) for p in sole_contacts(leg, np.array(M, dtype=np.float64)[None])[0]]
        out["soles"][leg.key] = pts
        pinned = any(p.enabled and p.owner == leg.key and p.mode in {"FLOOR", "WORLD"} and p.lock_in <= frame <= p.lock_out
                     for p in data.pins)
        touching = st.floor_object is None and min(p.z for p in pts) - st.floor_z < 0.015 * unit
        out["planted"][leg.key] = "PIN" if pinned else ("TOUCH" if touching else "")

    ends = {limb.key: limb.org_end for limb in profile.limbs()}
    ends["torso"] = profile.torso_ctrl
    for p in data.pins:
        if not p.enabled or not (p.lock_in - p.blend_in <= frame <= p.lock_out + p.blend_out):
            continue
        pb = bones.get(ends.get(p.owner, ""))
        if pb is None:
            continue
        pos = mw @ pb.head
        target = None
        if p.mode == "TARGET" and p.target_object is not None:
            tob = p.target_object.evaluated_get(deps)
            if tob.type == "ARMATURE" and p.target_bone in tob.pose.bones:
                target = tob.matrix_world @ tob.pose.bones[p.target_bone].head
            else:
                target = tob.matrix_world.translation.copy()
        locked = p.lock_in <= frame <= p.lock_out
        out["pins"].append((pos, p.mode, locked, target))

    if st.show_capsules and profile.is_supported:
        hips = [mw @ bones[l.org_upper].head for l in profile.legs.values() if l.org_upper in bones]
        hip_w = (hips[0] - hips[1]).length if len(hips) == 2 else 0.2 * unit
        n = max(len(profile.spine_bones), 1)
        for i, name in enumerate(profile.spine_bones):
            pb = bones.get(name)
            if pb is not None:
                out["capsules"].append((mw @ pb.head, mw @ pb.tail, hip_w * (0.66 + 0.12 * i / max(n - 1, 1)) * st.body_thickness))
        pb = bones.get(profile.head_bone)
        if pb is not None:
            c = mw @ (pb.head + 0.5 * (pb.tail - pb.head))
            r = max(0.5 * (pb.tail - pb.head).length * profile.scale, 0.45 * hip_w) * st.body_thickness
            out["capsules"].append((c, c, r))
        for leg in profile.legs.values():
            if leg.org_upper in bones and leg.org_lower in bones:
                out["capsules"].append((mw @ bones[leg.org_upper].head, mw @ bones[leg.org_lower].head,
                                        0.36 * hip_w * st.body_thickness))
    return out


def draw_view3d() -> None:
    try:
        context = bpy.context
        st = getattr(context.scene, "mocap_cleanup", None)
        if st is None or not st.show_hud:
            return
        space = context.space_data
        if space is None or not space.overlay.show_overlays:
            return
        rig = common.get_rig(context)
        if rig is None or rig.type != "ARMATURE" or not rig.visible_get():
            return
        gpu.state.blend_set("ALPHA")
        gpu.state.depth_test_set("NONE")
        draw_hud(context, rig, st)
        gpu.state.depth_test_set("LESS_EQUAL")
        gpu.state.blend_set("NONE")
    except Exception:
        _report("viewport")
        try:
            gpu.state.blend_set("NONE")
        except Exception:
            pass


def draw_hud(context, rig, st) -> None:
    """Draw the HUD of ``rig`` with the current view / projection matrices."""
    hud = hud_state(context, rig)
    if hud is None:
        return
    unit, fz = hud["unit"], hud["floor"]
    lift = 0.002 * unit
    com = hud["com"]

    if st.show_floor and st.floor_object is None:
        c = com if com is not None else rig.matrix_world.translation
        half, stepn = 1.2 * unit, 8
        cx, cy = round(c.x / (0.3 * unit)) * 0.3 * unit, round(c.y / (0.3 * unit)) * 0.3 * unit
        grid = []
        for i in range(stepn + 1):
            o = -half + 2 * half * i / stepn
            grid += [(cx + o, cy - half, fz), (cx + o, cy + half, fz), (cx - half, cy + o, fz), (cx + half, cy + o, fz)]
        _lines(grid, (0.55, 0.75, 1.0, 0.22), 1.0)

    planted_pts = []
    for key, pts in hud["soles"].items():
        state = hud["planted"].get(key, "")
        col = (0.25, 0.9, 0.45, 0.95) if state == "PIN" else ((0.95, 0.85, 0.3, 0.8) if state == "TOUCH" else (0.7, 0.7, 0.7, 0.35))
        order = [pts[0], pts[1], pts[3], pts[2], pts[0]] if len(pts) == 4 else pts + pts[:1]
        _lines(_strip_to_lines([(p.x, p.y, max(p.z, fz) + lift) for p in order]), col, 2.0 if state else 1.0)
        if state:
            planted_pts += [(p.x, p.y) for p in pts]

    if st.show_com and com is not None:
        hull = _convex_hull_2d(planted_pts)
        balanced = _inside(hull, (com.x, com.y))
        if len(hull) >= 3:
            tri = []
            for a, b in zip(hull[1:-1], hull[2:]):
                tri += [(hull[0][0], hull[0][1], fz + lift), (a[0], a[1], fz + lift), (b[0], b[1], fz + lift)]
            _tris(tri, (0.25, 0.9, 0.45, 0.16) if balanced else (1.0, 0.6, 0.15, 0.16))
            _lines(_strip_to_lines([(p[0], p[1], fz + lift) for p in hull + hull[:1]]),
                   (0.25, 0.9, 0.45, 0.7) if balanced else (1.0, 0.6, 0.15, 0.7), 1.5)
        col = (0.25, 0.9, 0.45, 0.95) if (balanced or not hull) else (1.0, 0.55, 0.15, 0.95)
        r = 0.03 * unit
        cross = []
        for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            d = Vector(axis) * r
            cross += [tuple(com - d), tuple(com + d)]
        _lines(cross, (1.0, 1.0, 1.0, 0.95), 2.0)
        _lines(_circle(com, r, (0, 0, 1)) + _circle(com, r, (1, 0, 0)) + _circle(com, r, (0, 1, 0)), col, 1.5)
        _lines([tuple(com), (com.x, com.y, fz + lift)], (*col[:3], 0.5), 1.0)
        _lines(_circle((com.x, com.y, fz + lift), 0.5 * r), col, 2.0)

    for pos, mode, locked, target in hud["pins"]:
        c = pin_tracks.MODE_COLORS.get(mode, (1, 1, 1))
        col = (*c, 0.95 if locked else 0.45)
        r = 0.035 * unit
        dia = []
        top, bot = pos + Vector((0, 0, r)), pos - Vector((0, 0, r))
        ring = [pos + Vector((r, 0, 0)), pos + Vector((0, r, 0)), pos - Vector((r, 0, 0)), pos - Vector((0, r, 0))]
        for i in range(4):
            dia += [tuple(ring[i]), tuple(ring[(i + 1) % 4]), tuple(ring[i]), tuple(top), tuple(ring[i]), tuple(bot)]
        _lines(dia, col, 2.0 if locked else 1.2)
        if target is not None:
            _lines([tuple(pos), tuple(target)], (*c, 0.6), 1.2)
            _lines(_circle(target, 0.6 * r, (0, 0, 1)) + _circle(target, 0.6 * r, (1, 0, 0)), (*c, 0.8), 1.2)

    for a, b, radius in hud["capsules"]:
        axis = (b - a)
        nrm = axis.normalized() if axis.length > 1e-6 else Vector((0, 0, 1))
        segs = _circle(a, radius, nrm) + _circle(b, radius, nrm)
        if axis.length > 1e-6:
            side = nrm.orthogonal().normalized()
            other = nrm.cross(side)
            for d in (side, -side, other, -other):
                segs += [tuple(a + radius * d), tuple(b + radius * d)]
        else:
            segs += _circle(a, radius, (1, 0, 0)) + _circle(a, radius, (0, 1, 0))
        _lines(segs, (0.4, 0.8, 1.0, 0.45), 1.0)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def register() -> None:
    unregister()
    if bpy.app.background:
        return
    _handles.append((bpy.types.SpaceGraphEditor, bpy.types.SpaceGraphEditor.draw_handler_add(draw_anim_editor, (), "WINDOW", "POST_PIXEL")))
    _handles.append((bpy.types.SpaceDopeSheetEditor, bpy.types.SpaceDopeSheetEditor.draw_handler_add(draw_anim_editor, (), "WINDOW", "POST_PIXEL")))
    _handles.append((bpy.types.SpaceView3D, bpy.types.SpaceView3D.draw_handler_add(draw_view3d, (), "WINDOW", "POST_VIEW")))


def unregister() -> None:
    while _handles:
        space, handle = _handles.pop()
        try:
            space.draw_handler_remove(handle, "WINDOW")
        except Exception:
            pass
