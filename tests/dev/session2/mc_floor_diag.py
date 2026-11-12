"""After Clean Up Take: rig soles against the solver floor, mesh against the visible floor. argv: <rig>"""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r"C:\Users\kondi\Documents\GIT_Folder\Mocap_Blender\tests")
import bpy
import numpy as np
import env
env.enable("mc")
M = env.PK["mc"]
action_io = importlib.import_module(M + ".core.action_io"); sampler = importlib.import_module(M + ".core.sampler")
floor_fit = importlib.import_module(M + ".core.floor_fit")
solver_bone_names = importlib.import_module(M + ".core.rig_inspector").solver_bone_names
s3 = importlib.import_module(M + ".stages.stage3_drift"); s4 = importlib.import_module(M + ".stages.stage4_contacts_pins")
common = importlib.import_module(M + ".ui.common")

rig_name = sys.argv[sys.argv.index("--") + 1]
C = bpy.context
rig = bpy.data.objects[rig_name]
st = C.scene.mocap_cleanup
st.rig = rig
C.view_layer.objects.active = rig
st.preset = "OPTICAL"; bpy.ops.mocap.apply_preset()


def soles_now():
    prof = common.profile_of(rig)
    f0, f1 = action_io.get_action_frame_range(rig)
    smp = sampler.sample_objects([(rig, solver_bone_names(prof, rig.data.bones))], f0, f1)[rig.name]
    so = s3._sole_points(prof, smp)
    return f0, f1, so, np.min([p[..., 2].min(axis=1) for p in so.values()], axis=0)


f0, f1, so0, low0 = soles_now()
rest0 = s4.estimate_floor_height(so0, 30.0, common.profile_of(rig).unit)
print(f"FD raw: rig sole rest {rest0 * 100:.2f} cm", flush=True)
r = bpy.ops.mocap.run_all()
print("FD", rig.mocap_cleanup.last_result[:200], flush=True)
f0, f1, so, low = soles_now()
k = rig.mocap_cleanup.sole_offset
fr = common.floor_for(C, rig)
rest1 = s4.estimate_floor_height(so, 30.0, common.profile_of(rig).unit)
frames = list(range(f0, f1 + 1, 6))
meshes = floor_fit.character_meshes(C.scene, rig)
ml = floor_fit.mesh_lowest(C.scene, meshes, frames)
li = np.array([f - f0 for f in frames])
print(f"FD after: k {k * 100:.2f} cm, solver floor {fr * 100:.2f} cm, rig sole rest {rest1 * 100:.2f} cm "
      f"(min {low.min() * 100:.2f}, p10 {np.percentile(low, 10) * 100:.2f}, median {np.median(low) * 100:.2f})", flush=True)
gap = low[li] - ml
print(f"FD mesh lowest: median {np.median(ml) * 100:.2f} cm, p10 {np.percentile(ml, 10) * 100:.2f}, min {ml.min() * 100:.2f};"
      f" rig sole minus mesh: median {np.median(gap) * 100:.2f} cm, p10 {np.percentile(gap, 10) * 100:.2f}, p90 {np.percentile(gap, 90) * 100:.2f}", flush=True)
rest = np.abs(low[li] - rest1) < 0.005
print(f"FD at rest frames ({rest.sum()} of {len(li)}): mesh {np.median(ml[rest]) * 100:.2f} cm, rig sole {np.median(low[li][rest]) * 100:.2f} cm", flush=True)
