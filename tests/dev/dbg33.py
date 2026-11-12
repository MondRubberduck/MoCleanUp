"""Around given frames: what moved a knee? raw vs cleaned reach, foot height, lift (source tree, read-only).

blender -b --factory-startup -y --python dbg33.py -- <file.blend> <rig> <L|R> <a> <b> [step]
"""
import sys

sys.path.insert(0, r"C:\Users\kondi\Documents\GIT_Folder\Mocap_Blender\tests")
import bpy
import numpy as np

import _common
import mocap_cleanup
from mocap_cleanup.core import action_io, runtime, sampler
from mocap_cleanup.core.rig_inspector import commanded_end, sole_contacts, solver_bone_names
from mocap_cleanup.ui import common

args = sys.argv[sys.argv.index("--") + 1:]
mocap_cleanup.register()
bpy.ops.wm.open_mainfile(filepath=args[0], load_ui=False)
np.set_printoptions(precision=1, suppress=True, linewidth=250)
scene = bpy.context.scene
rig = bpy.data.objects[args[1]]
S, a, b = args[2], int(args[3]), int(args[4])
step = int(args[5]) if len(args) > 5 else 1
st = scene.mocap_cleanup
st.rig = rig
bpy.context.view_layer.objects.active = rig
st.preset = "OPTICAL"
bpy.ops.mocap.apply_preset()
profile = common.profile_of(rig)
unit = profile.unit
f0, f1 = action_io.get_action_frame_range(rig)
leg = profile.legs[S]


def state():
    smp = sampler.sample_objects([(rig, solver_bone_names(profile, rig.data.bones))], f0, f1)[rig.name]
    root, mid, end = smp.head_w(leg.org_upper), smp.head_w(leg.org_lower), smp.head_w(leg.org_end)
    aim = commanded_end(leg, smp)
    sole = sole_contacts(leg, smp.world_m(leg.org_end))
    return dict(root=root, mid=mid, end=end, aim=aim if aim is not None else end, sole=sole,
                pole=smp.head_w(leg.pole_ctrl) if leg.pole_ctrl and smp.has(leg.pole_ctrl) else None)


for label, setup in (("filter only", dict(pipe_contacts=False, pipe_solve=False)), ("all", {})):
    bpy.ops.wm.open_mainfile(filepath=args[0], load_ui=False)
    scene = bpy.context.scene
    rig = bpy.data.objects[args[1]]
    st = scene.mocap_cleanup
    st.rig = rig
    bpy.context.view_layer.objects.active = rig
    st.preset = "OPTICAL"
    bpy.ops.mocap.apply_preset()
    if label == "filter only":
        raw = state()
    for k, v in setup.items():
        setattr(st, k, v)
    bpy.ops.mocap.run_all()
    if label == "filter only":
        filt = state()
    else:
        out = state()
        res = runtime.SOLVE_RESULTS.get(rig.name)
        print(rig.mocap_cleanup.last_result)
        print(res.summary() if res else None, "|", (res.warnings if res else None))
        print("pins", [(p.owner, p.lock_in, p.lock_out, p.blend_in, p.blend_out) for p in rig.mocap_cleanup.pins if p.owner == "leg." + S])
i = np.arange(a - f0, b - f0 + 1, step)
cm = 100.0 / unit
print(f"leg {S}, frames", i + f0, "| floor", round(st.floor_z * cm, 2))
for name, d in (("raw", raw), ("filtered", filt), ("cleaned", out)):
    reach = np.linalg.norm(d["aim"] - d["root"], axis=1) / leg.max_reach
    print(f"  {name:9s} reach x1000 ", reach[i] * 1000)
    print(f"  {name:9s} sole low cm ", (d["sole"][i][..., 2].min(axis=1) - st.floor_z) * cm)
print("  foot moved cm: flat", np.linalg.norm((out["end"] - filt["end"])[i, :2], axis=1) * cm, "up", (out["end"] - filt["end"])[i, 2] * cm)
print("  knee moved cm (vs filtered)", np.linalg.norm(out["mid"] - filt["mid"], axis=1)[i] * cm)
print("  knee moved cm (filter vs raw)", np.linalg.norm(filt["mid"] - raw["mid"], axis=1)[i] * cm)
print("  hip moved cm", np.linalg.norm(out["root"] - filt["root"], axis=1)[i] * cm)
if out["pole"] is not None:
    print("  pole moved cm", np.linalg.norm(out["pole"] - filt["pole"], axis=1)[i] * cm)
sys.stdout.flush()
