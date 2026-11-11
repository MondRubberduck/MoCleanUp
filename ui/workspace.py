"""The "MoCap Cleanup" workspace: 3D viewport on top, Dope Sheet (pin tracks) and Graph Editor below.

The layout is built by a timer after the window has switched to the new workspace. Timers run without
a window in the context, and Blender dereferences that window when an editor changes its type or a
region is shown or hidden, so every screen edit below happens inside a ``temp_override``.
"""

from __future__ import annotations

import bpy
from bpy.types import Operator

WORKSPACE_NAME = "MoCap Cleanup"
PHASE_KEY = "mc_layout_phase"          # 0 = untouched, 1 = split once, 2 = split twice, 3 = editors set, 4 = finished
DONE = 4

_memory: dict[str, dict[str, int]] = {}   # workspace name -> area pointers remembered between timer ticks
_waits: dict[str, int] = {}
_ticks = 0


def _window_region(area):
    return next((r for r in area.regions if r.type == "WINDOW"), None)


def _ctx(window, area):
    return bpy.context.temp_override(window=window, screen=window.screen, area=area, region=_window_region(area))


def _size(area) -> int:
    return area.width * area.height


def _by_pointer(screen, pointer):
    return next((a for a in screen.areas if a.as_pointer() == pointer), None)


def areas_measured(screen) -> bool:
    """True when the area rectangles are up to date (they lag one redraw behind a split)."""
    areas = list(screen.areas)
    if any(a.width <= 1 or a.height <= 1 for a in areas):
        return False
    for i, a in enumerate(areas):
        for b in areas[i + 1:]:
            if a.x < b.x + b.width and b.x < a.x + a.width and a.y < b.y + b.height and b.y < a.y + a.height:
                return False
    return True


def _split(window, area, factor: float):
    """Split ``area`` with a horizontal line; returns the newly created area (or None).

    Blender puts the new area below the line for factors up to 0.5 and above it otherwise.
    """
    screen = window.screen
    before = {a.as_pointer() for a in screen.areas}
    try:
        with _ctx(window, area):
            bpy.ops.screen.area_split(direction="HORIZONTAL", factor=factor)
    except Exception:
        return None
    new = [a for a in screen.areas if a.as_pointer() not in before]
    return new[0] if new else None


def _collapse(window) -> None:
    """Close areas until one is left (or nothing more can be closed)."""
    screen = window.screen
    for _ in range(32):
        areas = sorted(screen.areas, key=_size)
        if len(areas) <= 1:
            return
        closed = False
        for area in areas[:-1]:
            try:
                with _ctx(window, area):
                    if "FINISHED" in bpy.ops.screen.area_close():
                        closed = True
                        break
            except Exception:
                continue
        if not closed:
            return


def _set_editor(window, area, ui_type: str, fallback: str) -> None:
    with _ctx(window, area):
        try:
            area.ui_type = ui_type
        except Exception:
            area.type = fallback


def _configure(window, view, dope, graph) -> None:
    if view is not None:
        _set_editor(window, view, "VIEW_3D", "VIEW_3D")
        with _ctx(window, view):
            view.spaces.active.show_region_ui = True
    if dope is not None:
        _set_editor(window, dope, "DOPESHEET", "DOPESHEET_EDITOR")
        with _ctx(window, dope):
            space = dope.spaces.active
            try:
                space.ui_mode = "ACTION"
            except Exception:
                pass
            space.show_region_ui = False
    if graph is not None:
        _set_editor(window, graph, "FCURVES", "GRAPH_EDITOR")
        with _ctx(window, graph):
            space = graph.spaces.active
            for attr, value in (("use_normalization", False), ("use_only_selected", True), ("show_region_ui", False)):
                try:
                    setattr(space, attr, value)
                except Exception:
                    pass
    for area in window.screen.areas:
        area.tag_redraw()


def _show_sidebar_tab(window) -> None:
    """Bring the add-on's tab to the front (the sidebar only knows its tabs after it has been drawn once)."""
    for area in window.screen.areas:
        if area.type != "VIEW_3D":
            continue
        for region in area.regions:
            if region.type == "UI":
                try:
                    with _ctx(window, area):
                        region.active_panel_category = "MoCap"
                except Exception:
                    pass
        area.tag_redraw()


def build_step(window, measured: bool) -> int:
    """One step of the layout build for the workspace shown in ``window``; returns the next phase.

    ``measured`` says whether the area rectangles can be trusted; otherwise the areas are told apart
    by Blender's split rule (see ``_split``).
    """
    ws = window.workspace
    screen = window.screen
    phase = int(ws.get(PHASE_KEY, 0))
    mem = _memory.setdefault(ws.name, {})

    if phase == 0 and (bpy.app.background or not areas_measured(screen)):
        # Without a real window (background mode, or one that was never drawn) Blender's
        # screen.area_close does not return, so the layout is left as it is.
        phase = DONE

    elif phase == 0:
        _collapse(window)
        main = max(screen.areas, key=_size)
        _set_editor(window, main, "VIEW_3D", "VIEW_3D")
        new = _split(window, main, 0.5)
        if new is None:
            _configure(window, main, None, None)
            phase = 3
        else:
            mem.clear()
            mem.update(top=main.as_pointer(), bottom=new.as_pointer())
            phase = 1

    elif phase == 1:
        mine = [a for a in (_by_pointer(screen, mem.get("top")), _by_pointer(screen, mem.get("bottom"))) if a]
        if len(mine) < 2:
            mine = sorted(screen.areas, key=_size)[-2:]
            measured = measured and len(mine) == 2
        if len(mine) < 2:
            phase = 3
        else:
            bottom = min(mine, key=lambda a: a.y) if measured else mine[1]
            top = mine[0] if mine[1] == bottom else mine[1]
            new = _split(window, bottom, 0.58)
            mem.clear()
            mem.update(view=top.as_pointer(), graph=bottom.as_pointer())
            if new is not None:
                mem["dope"] = new.as_pointer()
            phase = 2

    elif phase == 2:
        view, dope, graph = (_by_pointer(screen, mem.get(k)) for k in ("view", "dope", "graph"))
        mine = [a for a in (view, dope, graph) if a is not None]
        if measured and len(mine) == 3:
            view, dope, graph = sorted(mine, key=lambda a: -a.y)   # top to bottom
        elif measured and len(mine) == 2:
            (view, graph), dope = sorted(mine, key=lambda a: -a.y), None
        _configure(window, view, dope, graph)
        phase = 3

    elif phase == 3:
        _show_sidebar_tab(window)
        phase = DONE

    ws[PHASE_KEY] = phase
    if phase >= DONE:
        _memory.pop(ws.name, None)
    return phase


def build_now(window) -> bool:
    """Build the whole layout at once (no redraw in between, so by the split rule)."""
    for _ in range(5):
        if build_step(window, measured=False) >= DONE:
            return True
    return False


def _deferred_build():
    global _ticks
    again = False
    for window in bpy.context.window_manager.windows:
        ws = window.workspace
        if ws is None or ws.name != WORKSPACE_NAME or int(ws.get(PHASE_KEY, 0)) >= DONE:
            continue
        screen = window.screen
        if screen is None or screen.name not in {s.name for s in ws.screens}:
            again = True                                           # the window has not switched yet
            continue
        phase = int(ws.get(PHASE_KEY, 0))
        measured = areas_measured(screen)
        if phase == 3 and _waits.get(ws.name, 0) < 3:
            measured = False                                       # let the new editors draw first
        if not measured and _waits.get(ws.name, 0) < 12:
            _waits[ws.name] = _waits.get(ws.name, 0) + 1           # wait for the redraw after the split
            for area in screen.areas:
                area.tag_redraw()
            again = True
            continue
        _waits[ws.name] = 0
        try:
            phase = build_step(window, measured)
        except Exception:
            ws[PHASE_KEY] = phase = DONE                           # leave the workspace as it is
        again = again or phase < DONE
    _ticks += 1
    if again and _ticks < 120:
        return 0.08
    _ticks = 0
    return None


def unregister() -> None:
    if bpy.app.timers.is_registered(_deferred_build):
        bpy.app.timers.unregister(_deferred_build)
    _memory.clear()
    _waits.clear()


class MOCAP_OT_workspace_setup(Operator):
    bl_idname = "mocap.workspace_setup"
    bl_label = "Open MoCap Cleanup Workspace"
    bl_description = "Create (or switch to) a workspace laid out for cleanup: viewport, pin tracks and Graph Editor"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return context.window is not None

    def execute(self, context):
        global _ticks
        ws = bpy.data.workspaces.get(WORKSPACE_NAME)
        if ws is None:
            before = set(bpy.data.workspaces.keys())
            bpy.ops.workspace.duplicate()
            created = [w for w in bpy.data.workspaces if w.name not in before]
            if not created:
                self.report({"ERROR"}, "Could not create a workspace")
                return {"CANCELLED"}
            ws = created[0]
            ws.name = WORKSPACE_NAME
            ws[PHASE_KEY] = 0
        context.window.workspace = ws
        if int(ws.get(PHASE_KEY, DONE)) < DONE and not bpy.app.timers.is_registered(_deferred_build):
            # the window switches workspaces after this operator returns: build the layout then
            _ticks = 0
            bpy.app.timers.register(_deferred_build, first_interval=0.1)
        return {"FINISHED"}
