"""Bake a window with Rig Bridge's Bake to Control Rig and measure it against the skeleton's own action.

argv: <skeleton> <a> <b> <label>
Prints: bake time, then per IK/FK mode the worst position / rotation error of the bound bones and the number
of (bone, frame) pairs off by more than 90 deg (roll flips). Saves the IK channel samples to cmp_<label>.npz.
"""
import sys, os, time, math, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import numpy as np
from mathutils import Matrix
import env

argv = sys.argv[sys.argv.index("--") + 1:]
name, a, b, label = argv[0], int(argv[1]), int(argv[2]), argv[3]
env.enable("rb")
builder = importlib.import_module(env.PK["rb"] + ".builder")
C = bpy.context
sk = bpy.data.objects[name]
rig = sk.rig_bridge.rig
for o in bpy.data.objects:          # meshes do not matter for the numbers, keep runs fast
    if o.type == "MESH":
        o.hide_viewport = True
act = sk.animation_data.action
ref = sk.copy(); ref.name = name + "_REF"
for pb in ref.pose.bones:
    for c in list(pb.constraints):
        pb.constraints.remove(c)
C.scene.collection.objects.link(ref)
ref.animation_data.action = act
ref.animation_data.action_slot = sk.animation_data.action_slot
C.scene.frame_start, C.scene.frame_end = a, b
for o in C.view_layer.objects:
    o.select_set(False)
C.view_layer.objects.active = sk
sk.hide_set(False)
t0 = time.perf_counter()
builder.transfer_to_rig(C, sk, rig, sk, a, b, mode="IK_LEGS")
dt = time.perf_counter() - t0
print(f"CMP {label} bake {b - a + 1} frames in {dt:.1f}s = {dt * 1000 / (b - a + 1):.1f} ms/frame", flush=True)
if len(argv) > 4 and argv[4] == "timeonly":
    raise SystemExit(0)
sk.animation_data.action = None
bound = [pb.name for pb in sk.pose.bones if any(c.name.startswith("RB_") for c in pb.constraints) and not pb.name.startswith("ik_")]
frames = list(range(a, b + 1))
for mode in ("FK", "IK_LEGS", "IK"):
    builder.set_ik_mode(rig, mode, a)
    pe = re_ = 0.0
    flips = {}
    wpe = None
    for f in frames:
        C.scene.frame_set(f)
        for n in bound:
            ma = sk.matrix_world @ sk.pose.bones[n].matrix
            mb = ref.matrix_world @ ref.pose.bones[n].matrix
            d = (ma.translation - mb.translation).length
            ang = math.degrees(ma.to_quaternion().rotation_difference(mb.to_quaternion()).angle)
            ang = min(ang, 360 - ang)
            if d > pe:
                pe, wpe = d, (n, f)
            re_ = max(re_, ang)
            if ang > 90:
                flips.setdefault(n, []).append(f)
    nflip = sum(len(v) for v in flips.values())
    print(f"CMP {label} {mode:7s} pos {pe * 100:.3f} cm ({wpe}) rot {re_:.2f} deg  flips>90deg: {nflip} "
          + " ".join(f"{k}:{v[0]}-{v[-1]}({len(v)})" for k, v in list(flips.items())[:6]), flush=True)
builder.set_ik_mode(rig, "IK_LEGS", a)
# IK channel samples for old-vs-new comparison
data = {}
ad = rig.animation_data
for layer in ad.action.layers:
    for strip in layer.strips:
        for bag in strip.channelbags:
            for fc in bag.fcurves:
                if any(k in fc.data_path for k in ("_ik", "ik_target", "torso", "IK_FK")):
                    data[f"{fc.data_path}|{fc.array_index}"] = np.array([fc.evaluate(f) for f in frames])
np.savez(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"cmp_{label}.npz"), **data)
print(f"CMP {label} saved {len(data)} IK channels", flush=True)
