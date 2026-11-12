"""Build test fixtures from the QA result file (opened read-only, copies written to scratch).

blender -b --factory-startup -y --python fx1.py -- <QA_Result.blend> <out dir> [a-b ...]

full.blend      both characters with the UNCLEANED Rig Bridge bake on the control rigs (taken from the raw snapshots),
                no MoCap Cleanup state, default settings
f<a>_<b>.blend  the same, every action cropped to frames a..b (frame numbers kept)
"""
import os
import sys

sys.path.insert(0, r"C:\Users\kondi\Documents\GIT_Folder\Mocap_Blender\tests")
import bpy
import numpy as np

import _common
import mocap_cleanup
from mocap_cleanup.core import raw_snapshot

args = sys.argv[sys.argv.index("--") + 1:]
src, out_dir = args[0], args[1]
windows = [tuple(int(v) for v in a.split("-")) for a in args[2:]]
os.makedirs(out_dir, exist_ok=True)
mocap_cleanup.register()
full = os.path.join(out_dir, "full.blend")


def all_fcurves(act):
    for layer in act.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                yield from bag.fcurves


def crop(act, a, b):
    kept = 0
    for fc in all_fcurves(act):
        n = len(fc.keyframe_points)
        if n == 0:
            continue
        co = np.empty(2 * n)
        fc.keyframe_points.foreach_get("co", co)
        co = co.reshape(-1, 2)
        ipo = np.empty(n, dtype=np.int32)
        fc.keyframe_points.foreach_get("interpolation", ipo)
        keep = (co[:, 0] >= a - 0.5) & (co[:, 0] <= b + 0.5)
        if keep.all():
            kept += n
            continue
        if not keep.any():                      # a constant channel keyed outside the window (Rig Bridge keys IK_FK
            j = int(np.argmin(np.abs(co[:, 0] - a)))        # once, at frame 1): keep its value, at the start of the window.
            keep[j] = True                      # Left at frame 1 it made the take 1..b: mostly a held pose.
            co[j, 0] = a
        c, i = co[keep], ipo[keep]
        fc.keyframe_points.clear()
        fc.keyframe_points.add(len(c))
        fc.keyframe_points.foreach_set("co", c.ravel())
        fc.keyframe_points.foreach_set("interpolation", i)
        fc.keyframe_points.foreach_set("handle_left_type", np.full(len(c), 3, dtype=np.int32))    # AUTO
        fc.keyframe_points.foreach_set("handle_right_type", np.full(len(c), 3, dtype=np.int32))
        fc.update()
        kept += len(c)
    return kept


if not os.path.exists(full):
    bpy.ops.wm.open_mainfile(filepath=src, load_ui=False)
    scene = bpy.context.scene
    for ob in scene.objects:
        if ob.type != "ARMATURE" or ob.animation_data is None or ob.animation_data.action is None:
            continue
        snap = raw_snapshot.get_raw_snapshot(ob)
        if snap is None:
            continue
        name = ob.animation_data.action.name
        old = ob.animation_data.action
        new = snap.copy()
        for key in list(new.keys()):
            if key.startswith("mc_") or key.startswith("mocap"):
                del new[key]
        for tr in list(ob.animation_data.nla_tracks):
            ob.animation_data.nla_tracks.remove(tr)
        ob.animation_data.action = new
        ob.animation_data.action_slot = next((s for s in new.slots if s.identifier == "OB" + ob.name), new.slots[0])
        bpy.data.actions.remove(old)
        new.name = name
        new.use_fake_user = False
        d = ob.mocap_cleanup
        for p in d.bl_rna.properties:
            if not p.is_readonly and p.identifier != "rna_type":
                d.property_unset(p.identifier)
        d.pins.clear()
        d.issues.clear()
        print("FX restored the uncleaned bake on", ob.name, "->", new.name, "| custom keys left:", list(new.keys()))
    for act in list(bpy.data.actions):
        if raw_snapshot.is_helper_action(act) or act.name.startswith("."):
            bpy.data.actions.remove(act)
    st = scene.mocap_cleanup
    for p in st.bl_rna.properties:
        if not p.is_readonly and p.identifier != "rna_type":
            st.property_unset(p.identifier)
    for ob in scene.objects:
        if ob.type == "ARMATURE":
            ob.data.pose_position = "POSE"
    bpy.ops.wm.save_as_mainfile(filepath=full, copy=True, compress=True)
    print("FX wrote", full, os.path.getsize(full) // 2 ** 20, "MB | actions", [(a.name, a.users) for a in bpy.data.actions])

for a, b in windows:
    out = os.path.join(out_dir, f"f{a}_{b}.blend")
    if os.path.exists(out):
        continue
    bpy.ops.wm.open_mainfile(filepath=full, load_ui=False)
    scene = bpy.context.scene
    total = sum(crop(act, a, b) for act in bpy.data.actions)
    scene.frame_start, scene.frame_end = a, b
    scene.frame_set(a)
    bpy.ops.wm.save_as_mainfile(filepath=out, copy=True, compress=True)
    print("FX wrote", out, os.path.getsize(out) // 2 ** 20, "MB | keys", total)
sys.stdout.flush()
