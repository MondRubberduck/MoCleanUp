"""Inside one detected pin: what does the planted-foot target do around given frames? (source tree, read-only)

blender -b --factory-startup -y --python dbg36.py -- <file.blend> <rig> <L|R> <a> <b>
"""
import sys

sys.path.insert(0, r"C:\Users\kondi\Documents\GIT_Folder\Mocap_Blender\tests")
import bpy
import numpy as np

import _common
import mocap_cleanup
from mocap_cleanup.core import action_io, sampler
from mocap_cleanup.core.math_quat import quat_from_matrix
from mocap_cleanup.core.rig_inspector import solver_bone_names
from mocap_cleanup.stages import stage4_contacts_pins as s4
from mocap_cleanup.ui import common

args = sys.argv[sys.argv.index("--") + 1:]
mocap_cleanup.register()
bpy.ops.wm.open_mainfile(filepath=args[0], load_ui=False)
np.set_printoptions(precision=2, suppress=True, linewidth=250)
scene = bpy.context.scene
rig = bpy.data.objects[args[1]]
S, a, b = args[2], int(args[3]), int(args[4])
st = scene.mocap_cleanup
st.rig = rig
bpy.context.view_layer.objects.active = rig
st.preset = "OPTICAL"
bpy.ops.mocap.apply_preset()
st.pipe_solve = False
bpy.ops.mocap.run_all()                        # repair, filter, contacts - no solve
profile = common.profile_of(rig)
unit = profile.unit
f0, f1 = action_io.get_action_frame_range(rig)
fps = scene.render.fps / scene.render.fps_base
leg = profile.legs[S]
smp = sampler.sample_objects([(rig, solver_bone_names(profile, rig.data.bones))], f0, f1)[rig.name]
W = smp.world_m(leg.org_end)
pos, quat = W[:, :3, 3].copy(), quat_from_matrix(W)
scale = float(np.linalg.norm(smp.obj_world[0][:3, 0]))
sole = leg.sole_points * scale
fwd = leg.ball_local - leg.heel_local
fwd /= np.linalg.norm(fwd)
rest_q = quat_from_matrix(smp.obj_world[0] @ np.array(rig.data.bones[leg.org_end].matrix_local))
floor = s4.Floor(st.floor_z)
pins = [p for p in rig.mocap_cleanup.pin_data() if p.owner == "leg." + S]
pin = next(p for p in pins if p.lock_in <= a and p.lock_out >= a)
li, lo = pin.lock_in - f0, pin.lock_out - f0
p_t, q_t = s4._planted_foot(pos, quat, sole, fwd, rest_q, floor, li, lo, -1, fps, unit, pin.lock_rotation, np.radians(4.0),
                            leg.sole_radius * scale, leg.up_local, max_yaw=np.radians(12.0))
i = np.arange(a - f0, b - f0 + 1)
cm = 100.0 / unit
print("pin", pin.lock_in, pin.lock_out, "frames", i + f0)
print("  raw foot x cm    ", (pos[i, 0] - pos[i[0], 0]) * cm)
print("  raw foot y cm    ", (pos[i, 1] - pos[i[0], 1]) * cm)
print("  target x cm      ", (p_t[i, 0] - pos[i[0], 0]) * cm)
print("  target y cm      ", (p_t[i, 1] - pos[i[0], 1]) * cm)
print("  target - raw flat", np.linalg.norm((p_t - pos)[i, :2], axis=1) * cm)
print("  target - raw up  ", (p_t - pos)[i, 2] * cm)
from mocap_cleanup.core.math_quat import quat_angle_between
print("  turned deg       ", np.degrees(quat_angle_between(q_t[i], quat[i])))
r_raw = s4.sole_offsets(quat[:, None, :], sole[None], leg.sole_radius * scale, leg.up_local)
pts = pos[:, None, :] + r_raw
h = pts[..., 2] - floor.z
print("  sole heights cm (heel in, heel out, ball in, ball out):")
print(np.round(h[i].T * cm, 2))
sp = np.linalg.norm(s4.windowed_velocity(pts, fps), axis=2)
print("  sole speeds cm/s:")
print(np.round(sp[i].T * cm, 1))
sys.stdout.flush()
