"""Row 12, part 3: one click (Optical preset, defaults) on both full-length control rigs, saved as a new file.
argv: <out.blend>"""
import sys, os, time, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import env

out = sys.argv[sys.argv.index("--") + 1]
env.enable()
C = bpy.context
st = C.scene.mocap_cleanup
for name in ("Armature_rig", "Armature.001_rig"):
    rig = bpy.data.objects[name]
    for o in C.view_layer.objects:
        o.select_set(False)
    rig.hide_set(False); rig.select_set(True); C.view_layer.objects.active = rig
    st.rig = rig
    st.preset = "OPTICAL"; bpy.ops.mocap.apply_preset()
    t = time.perf_counter(); r = bpy.ops.mocap.run_all(); dt = time.perf_counter() - t
    print(f"FINAL {name}: {r} in {dt:.1f}s: {rig.mocap_cleanup.last_result}", flush=True)
st.rig = None
C.scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=out, copy=True)
print("FINAL saved", out, round(os.path.getsize(out) / 1e6, 1), "MB", flush=True)
