"""Row 11 checks on f940_1140: BPR with Rig Bridge copies (I3), Batch helpers (I7), Ignore NLA note (I6)."""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import env

env.enable("rb", "bpr")
builder = importlib.import_module(env.PK["rb"] + ".builder")
C = bpy.context
S = C.scene.bp_retarget
R = []
def check(name, ok, info=""):
    R.append(ok); print(f"R11 [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)

def select(*objs):
    for o in C.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.hide_set(False); o.select_set(True)
    C.view_layer.objects.active = objs[-1]

def keys(act):
    return sum(len(fc.keyframe_points) for fc in builder._action_fcurves(act))

for o in bpy.data.objects:
    if o.type == "MESH":
        o.hide_viewport = True
mocap = bpy.data.objects["Ayana_Root"]; take = bpy.data.actions["Fight_Ayana_FBX"]
orig = bpy.data.objects["Armature"]; rig = orig.rig_bridge.rig
rig_act = rig.animation_data.action; rig_keys = keys(rig_act)

def retarget(target):
    S.source, S.target, S.action = mocap, target, take
    S.use_batch = False; S.range_mode = "CUSTOM"; S.frame_start = 940; S.frame_end = 960
    S.rig_bridge_bake = True; S.overwrite = True
    return bpy.ops.bpr.retarget()

# I3: a Shift+D copy whose name sorts before the original
select(orig)
bpy.ops.object.duplicate(linked=False)
dup = next(o for o in C.selected_objects if o.type == "ARMATURE")
dup.name = "AAA_copy"
r = retarget(dup)
check("retarget onto the copy leaves the original's rig take alone",
      rig.animation_data.action == rig_act and keys(rig_act) == rig_keys, rig.animation_data.action.name)
check("the report says the copy follows another rig", "is a copy of 'Armature'" in S.last_report
      and "Baked onto" not in S.last_report, S.last_report.replace("\n", " | ")[:220])
S.target = rig
check("picking the control rig resolves to its owner, not the copy", S.target == orig, S.target and S.target.name)

# normal path still bakes onto the owner's rig
r = retarget(rig)
check("retarget onto the owner bakes onto its rig", "Baked onto the Rig Bridge control rig 'Armature_rig'" in S.last_report
      and rig.animation_data.action != rig_act, rig.animation_data.action.name)

# I7: hidden helper actions are not listed in Batch
helper = take.copy(); helper.name = ".Fight_Ayana_FBX.raw"
S.source = mocap
bpy.ops.bpr.batch_refresh()
listed = [b.action.name for b in S.batch if b.action]
check("Batch skips dot-prefixed helper actions", ".Fight_Ayana_FBX.raw" not in listed and "Fight_Ayana_FBX" in listed, str(listed))

# I6: Ignore NLA names the source tracks it leaves out
tr = mocap.animation_data.nla_tracks.new(); tr.name = "MoCap Residual Detail"
tr.strips.new("detail", 940, helper)
plain = bpy.data.objects["Armature.001"].copy(); plain.name = "UE5_plain"
for pb in plain.pose.bones:
    for c in list(pb.constraints):
        pb.constraints.remove(c)
plain.rig_bridge.rig = None
C.scene.collection.objects.link(plain)
S.isolate_action = True
r = retarget(plain)
check("Ignore NLA reports the source's NLA track", "'MoCap Residual Detail'" in S.last_report and "Ignore NLA" in S.last_report,
      S.last_report.replace("\n", " | ")[:220])
print(f"R11 SUMMARY {sum(R)} / {len(R)}", flush=True)
