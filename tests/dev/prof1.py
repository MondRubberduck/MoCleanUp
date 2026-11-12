import sys, cProfile, pstats, io
sys.path.insert(0, r"C:\Users\kondi\Documents\GIT_Folder\Mocap_Blender\tests")
import bpy
import _common, mocap_cleanup
mocap_cleanup.register()
bpy.ops.wm.open_mainfile(filepath=sys.argv[sys.argv.index("--") + 1], load_ui=False)
rig = bpy.data.objects[sys.argv[sys.argv.index("--") + 2]]
st = bpy.context.scene.mocap_cleanup
st.rig = rig
bpy.context.view_layer.objects.active = rig
st.preset = "OPTICAL"; bpy.ops.mocap.apply_preset()
pr = cProfile.Profile(); pr.enable()
bpy.ops.mocap.run_all()
pr.disable()
out = io.StringIO(); pstats.Stats(pr, stream=out).sort_stats("cumulative").print_stats(28)
print("PROF", rig.mocap_cleanup.last_result)
for line in out.getvalue().splitlines():
    if "seconds" in line or "{" in line or ".py" in line:
        print("PROF", line[:200])
