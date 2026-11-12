"""Row 10 checks on f940_1140: duplicates (I3), NLA over a rebuild (I5), Bake to Skeleton naming (R2)."""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import env

env.enable("rb")
builder = importlib.import_module(env.PK["rb"] + ".builder")
C = bpy.context
R = []
def check(name, ok, info=""):
    R.append(ok); print(f"R10 [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)

def select(*objs):
    for o in C.view_layer.objects:
        o.select_set(False)
    for o in objs:
        o.hide_set(False); o.select_set(True)
    C.view_layer.objects.active = objs[-1]

def keys(act):
    return sum(len(fc.keyframe_points) for fc in builder._action_fcurves(act))

orig = bpy.data.objects["Armature"]; rig = orig.rig_bridge.rig
rig_act = rig.animation_data.action; rig_keys = keys(rig_act); rig_bones = len(rig.data.bones)
C.scene.frame_start, C.scene.frame_end = 940, 980
check("original owns its rig", builder.owns_rig(orig) and rig.rig_bridge.source == orig)

# --- I3: Shift+D of skeleton + mesh only ---
select(bpy.data.objects["Beta_Surface"], orig)
bpy.ops.object.duplicate(linked=False)
dup = next(o for o in C.selected_objects if o.type == "ARMATURE")
check("copy still points at the original's rig", dup.rig_bridge.rig == rig, dup.name)
check("copy does not own it", not builder.owns_rig(dup))
select(dup)
check("rig tools are off for the copy", not bpy.ops.rig_bridge.transfer.poll() and not bpy.ops.rig_bridge.bake.poll()
      and not bpy.ops.rig_bridge.mute.poll())
try:
    builder.transfer_to_rig(C, dup, rig, dup, 940, 980, mode="IK_LEGS")
    refused = False
except builder.BuildError as exc:
    refused = "copy" in str(exc)
check("the bake refuses the copy", refused)
check("original take untouched", rig.animation_data.action == rig_act and keys(rig_act) == rig_keys)
select(dup)
r = bpy.ops.rig_bridge.build()
nrig = dup.rig_bridge.rig
check("Compile gives the copy its own rig", r == {"FINISHED"} and nrig is not None and nrig != rig and builder.owns_rig(dup),
      nrig and nrig.name)
bound_to = {c.target.name for pb in dup.pose.bones for c in pb.constraints if c.name.startswith("RB_") and getattr(c, "target", None)}
check("copy is bound to its own rig only", rig.name not in bound_to and nrig.name in bound_to, str(sorted(bound_to))[:120])
check("original rig and take untouched", len(rig.data.bones) == rig_bones and rig.animation_data.action == rig_act
      and keys(rig_act) == rig_keys and builder.owns_rig(orig) and orig.rig_bridge.metarig != dup.rig_bridge.metarig)
obound = {c.target.name for pb in orig.pose.bones for c in pb.constraints if c.name.startswith("RB_") and getattr(c, "target", None)}
check("original still bound to its rig", rig.name in obound and nrig.name not in obound)

# --- skeleton + rig duplicated together: an independent copy ---
select(orig, rig)
bpy.ops.object.duplicate(linked=False)
dup2 = next(o for o in C.selected_objects if o.type == "ARMATURE" and o.rig_bridge.rig is not None and o.name.startswith("Armature.") and not o.name.endswith("rig"))
check("copy made together with its rig owns the copied rig", builder.owns_rig(dup2) and dup2.rig_bridge.rig != rig,
      f"{dup2.name} -> {dup2.rig_bridge.rig.name}")

# --- I5: NLA tracks survive a rebuild ---
tr = rig.animation_data.nla_tracks.new(); tr.name = "MoCap Residual Detail"
detail = rig_act.copy(); detail.name = ".detail_test"
st = tr.strips.new("detail", 940, detail); st.blend_type = "REPLACE"; st.influence = 0.7
select(orig)
r = bpy.ops.rig_bridge.build()
ad = rig.animation_data
tracks = [(t.name, [(s.action.name, s.blend_type, round(s.influence, 2)) for s in t.strips]) for t in ad.nla_tracks]
check("rebuild keeps the active action", r == {"FINISHED"} and ad.action == rig_act, ad.action and ad.action.name)
check("rebuild keeps NLA tracks and strips", tracks == [("MoCap Residual Detail", [(".detail_test", "REPLACE", 0.7)])], str(tracks))

# --- R2: Bake to Skeleton names its take ---
prev = orig.animation_data.action
select(orig)
r = bpy.ops.rig_bridge.bake()
act = orig.animation_data.action
check("Bake to Skeleton gives a readable name", r == {"FINISHED"} and act.name.startswith("Armature|") and "Action" not in act.name, act.name)
check("skeleton's previous take kept", prev.name in bpy.data.actions and bpy.data.actions[prev.name].use_fake_user, prev.name)
print(f"R10 SUMMARY {sum(R)} / {len(R)}", flush=True)
