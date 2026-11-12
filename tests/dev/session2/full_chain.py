"""Row 12, part 1: the chain on the full take with the fixed add-ons. Opens the user's file read-only.

Rig Bridge Compile (both characters) -> Bulletproof Retarget of the whole take onto each skeleton with
Bake to Control Rig -> saves a copy (scratch) for the cleanup runs. Prints timings.
argv: <out.blend>
"""
import sys, os, time, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import env

out = sys.argv[sys.argv.index("--") + 1]
env.enable()
builder = importlib.import_module(env.PK["rb"] + ".builder")
C = bpy.context
S = C.scene.bp_retarget
mocap = bpy.data.objects["Ayana_Root"]
take = mocap.animation_data.action


def activate(ob):
    for o in C.view_layer.objects:
        o.select_set(False)
    ob.hide_set(False); ob.select_set(True); C.view_layer.objects.active = ob


for name in ("Armature", "Armature.001"):
    sk = bpy.data.objects[name]
    activate(sk)
    t = time.perf_counter(); r = bpy.ops.rig_bridge.build(); dt = time.perf_counter() - t
    print(f"CHAIN compile {name}: {r} {sk.rig_bridge.status} in {dt:.1f}s", flush=True)
for name in ("Armature", "Armature.001"):
    sk = bpy.data.objects[name]
    S.source, S.target, S.action = mocap, sk.rig_bridge.rig, take       # pick the control rig, as a user would
    S.use_batch = False; S.range_mode = "ACTION"; S.rig_bridge_bake = True
    t = time.perf_counter(); r = bpy.ops.bpr.retarget(); dt = time.perf_counter() - t
    rig = sk.rig_bridge.rig
    act = rig.animation_data.action
    n = int(act.frame_range[1] - act.frame_range[0] + 1)
    print(f"CHAIN retarget + bake onto {rig.name}: {r} {n} frames in {dt:.1f}s ({dt * 1000 / n:.1f} ms/frame) -> "
          f"{act.name} | {S.last_report.splitlines()[-1][:120]}", flush=True)
C.scene.frame_end = int(take.frame_range[1] + 0.999)
bpy.ops.wm.save_as_mainfile(filepath=out, copy=True)
print("CHAIN saved", out, flush=True)
