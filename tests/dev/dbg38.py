import sys
sys.path.insert(0, r"C:\Users\kondi\Documents\GIT_Folder\Mocap_Blender\tests")
import bpy, numpy as np
import _common, mocap_cleanup
from mocap_cleanup.core import action_io, sampler
from mocap_cleanup.core.math_quat import quat_from_matrix, quat_angle_between
from mocap_cleanup.ui import common
mocap_cleanup.register()
bpy.ops.wm.open_mainfile(filepath=sys.argv[sys.argv.index("--") + 1], load_ui=False)
rig = bpy.data.objects[sys.argv[sys.argv.index("--") + 2]]
a, b = 1102, 1114
np.set_printoptions(precision=0, suppress=True, linewidth=220)
def show(tag):
    f0, f1 = action_io.get_action_frame_range(rig)
    smp = sampler.sample_objects([(rig, ["ORG-thigh.R", "thigh_fk.R", "ORG-shin.R", "shin_fk.R", "ORG-foot.R", "MCH-thigh_ik_target.R", "thigh_ik_target.R"])], f0, f1)[rig.name]
    i = slice(a - f0, b - f0 + 1)
    qo, qf = quat_from_matrix(smp.pose_m("ORG-thigh.R")), quat_from_matrix(smp.pose_m("thigh_fk.R"))
    print(tag, "ORG-thigh vs thigh_fk deg     ", np.degrees(quat_angle_between(qo, qf))[i])
    print(tag, "ORG-thigh frame-to-frame deg ", np.degrees(quat_angle_between(qo[1:], qo[:-1]))[a - f0 - 1: b - f0])
    root, mid, end = smp.pose_m("ORG-thigh.R")[:, :3, 3], smp.pose_m("ORG-shin.R")[:, :3, 3], smp.pose_m("ORG-foot.R")[:, :3, 3]
    reach = np.linalg.norm(end - root, axis=1) / (np.linalg.norm(mid - root, axis=1) + np.linalg.norm(end - mid, axis=1))
    print(tag, "reach x10000                 ", (reach * 10000)[i])
show("RAW    ")
st = bpy.context.scene.mocap_cleanup
st.rig = rig
bpy.context.view_layer.objects.active = rig
st.preset = "OPTICAL"; bpy.ops.mocap.apply_preset()
bpy.ops.mocap.run_all()
show("CLEANED")
