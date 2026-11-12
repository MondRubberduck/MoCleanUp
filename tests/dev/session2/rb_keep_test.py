"""Row 8 (QA I2) check on f940_1140: a re-bake replaces only an unedited own take."""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import env

env.enable()
builder = importlib.import_module(env.PK["rb"] + ".builder")
C = bpy.context
R = []
def check(name, ok, info=""):
    R.append(ok); print(f"KEEP [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)

sk = bpy.data.objects["Armature"]; rig = sk.rig_bridge.rig
for o in bpy.data.objects:
    if o.type == "MESH": o.hide_viewport = True
C.scene.frame_start, C.scene.frame_end = 940, 1000
def bake(notes=None):
    for o in C.view_layer.objects: o.select_set(False)
    C.view_layer.objects.active = sk
    act, n = builder.transfer_to_rig(C, sk, rig, sk, 940, 1000, mode="IK_LEGS", notes=notes)
    return act
name = f"{rig.name}|{sk.animation_data.action.name}"
legacy = bpy.data.actions.get(name)
check("fixture take is a legacy bake (tag True)", legacy is not None and legacy.get("rig_bridge_bake") in (True, 1))
notes = []
a1 = bake(notes)
check("legacy take kept, new bake gets _2", a1.name == name + "_2" and bpy.data.actions.get(name) is not None
      and bpy.data.actions[name].use_fake_user, f"{a1.name} | {notes}")
check("new bake is tagged with a checksum", isinstance(a1.get("rig_bridge_bake"), float) and builder.is_own_bake(a1))
notes = []
a2 = bake(notes)
check("re-bake replaces the unedited own take (no _3)", a2.name == name + "_2" and bpy.data.actions.get(name + "_3") is None,
      f"{a2.name} | {notes}")
# edit one key, as MoCap Cleanup or an animator would
fc = next(builder._action_fcurves(a2))
fc.keyframe_points[3].co.y += 0.01
check("an edited take is no longer 'own, unedited'", not builder.is_own_bake(a2))
notes = []
a3 = bake(notes)
kept = bpy.data.actions.get(name + "_2")
check("edited take kept, with fake user", kept is not None and kept != a3 and kept.use_fake_user, f"kept={kept and kept.name}")
check("new bake named _3 and reported", a3.name == name + "_3" and notes and notes[0].startswith("Kept '"), f"{a3.name} | {notes}")

# the real QA I2 path: MoCap Cleanup cleans the rig's take, then BP Retarget runs again onto the character
mc = C.scene.mocap_cleanup
mc.rig = rig
C.view_layer.objects.active = rig
r = bpy.ops.mocap.run_all()
cleaned = rig.animation_data.action
check("MoCap Cleanup ran on the rig's take", r == {"FINISHED"} and not builder.is_own_bake(cleaned), rig.mocap_cleanup.last_result[:120])
S = C.scene.bp_retarget
S.source = bpy.data.objects["Ayana_Root"]; S.target = rig; S.action = bpy.data.actions["Fight_Ayana_FBX"]
S.range_mode = "CUSTOM"; S.frame_start = 940; S.frame_end = 1000; S.overwrite = True; S.rig_bridge_bake = True
before = cleaned.name
r = bpy.ops.bpr.retarget()
check("re-running BP Retarget keeps the cleaned take", bpy.data.actions.get(before) is cleaned and rig.animation_data.action is not cleaned
      and cleaned.use_fake_user, f"cleaned={before} now={rig.animation_data.action.name} | {S.last_report.splitlines()[-1][:120]}")
print(f"KEEP SUMMARY {sum(R)} / {len(R)}", flush=True)
