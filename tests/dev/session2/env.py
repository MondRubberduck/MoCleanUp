"""Enable the three add-ons from the isolated profile (source junctions)."""
import bpy, addon_utils, importlib
PK = {"rb": "bl_ext.user_default.rig_bridge", "bpr": "bl_ext.user_default.bulletproof_retarget", "mc": "bl_ext.user_default.mocap_cleanup"}
def enable(*which):
    out = {}
    for k in (which or PK):
        m = PK[k]
        if m not in bpy.context.preferences.addons:
            bpy.ops.preferences.addon_enable(module=m)
        mod = importlib.import_module(m)
        out[k] = mod.__file__
    return out
