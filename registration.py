"""Class registration, draw handlers and key maps."""

from __future__ import annotations

import bpy
from bpy.app.handlers import persistent

from . import props
from .core import runtime, sampler
from .overlays import draw, pin_tracks
from .ui import autosolve, operators, panels, workspace

_keymaps: list[tuple] = []
CLASSES = (*operators.CLASSES, pin_tracks.MOCAP_OT_pin_track_edit, workspace.MOCAP_OT_workspace_setup, *panels.CLASSES)


@persistent
def _on_load(_dummy=None):
    runtime.clear_all()
    sampler.reset_cache()


def _topbar_button(self, context):
    self.layout.operator("mocap.workspace_setup", text="MoCap Cleanup", icon="WORKSPACE")


def _register_keymaps() -> None:
    wm = bpy.context.window_manager
    kc = wm.keyconfigs.addon if wm is not None else None
    if kc is None:
        return
    for name, space in (("Dopesheet", "DOPESHEET_EDITOR"), ("Graph Editor", "GRAPH_EDITOR")):
        km = kc.keymaps.new(name=name, space_type=space)
        for shift in (False, True):
            kmi = km.keymap_items.new("mocap.pin_track_edit", "LEFTMOUSE", "PRESS", shift=shift)
            _keymaps.append((km, kmi))
    for name, space in (("Dopesheet", "DOPESHEET_EDITOR"), ("Graph Editor", "GRAPH_EDITOR"), ("3D View", "VIEW_3D")):
        km = kc.keymaps.new(name=name, space_type=space)
        for key, direction in (("LEFT_BRACKET", -1), ("RIGHT_BRACKET", 1)):
            kmi = km.keymap_items.new("mocap.issue_jump", key, "PRESS")
            kmi.properties.direction = direction
            _keymaps.append((km, kmi))


def register() -> None:
    props.register()
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    draw.register()
    _register_keymaps()
    bpy.app.handlers.load_post.append(_on_load)
    bpy.types.TOPBAR_MT_window.append(_topbar_button)


def unregister() -> None:
    bpy.types.TOPBAR_MT_window.remove(_topbar_button)
    if _on_load in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_on_load)
    for km, kmi in _keymaps:
        try:
            km.keymap_items.remove(kmi)
        except Exception:
            pass
    _keymaps.clear()
    draw.unregister()
    autosolve.unregister()
    workspace.unregister()
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
    props.unregister()
