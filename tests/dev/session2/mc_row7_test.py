"""Row 7 checks on f940_1140 (Armature_rig)."""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import env

env.enable("mc")
M = env.PK["mc"]
common = importlib.import_module(M + ".ui.common")
raw_snapshot = importlib.import_module(M + ".core.raw_snapshot")
registration = importlib.import_module(M + ".registration")
C = bpy.context
st = C.scene.mocap_cleanup
R = []
def check(name, ok, info=""):
    R.append(ok); print(f"R7 [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)

def helpers():
    return sorted(a.name for a in bpy.data.actions if raw_snapshot.is_helper_action(a))

rig = bpy.data.objects["Armature_rig"]
for o in C.view_layer.objects:
    o.select_set(False)
rig.hide_set(False); C.view_layer.objects.active = rig; st.rig = rig
st.preset = "OPTICAL"; bpy.ops.mocap.apply_preset()
take = rig.animation_data.action
check("residual layer is off by default", not st.simplify_residual)
bpy.ops.mocap.run_all()
check("after Clean Up Take: a snapshot and a result line", raw_snapshot.get_raw_snapshot(rig) is not None
      and common.result_for(rig).startswith("Done"), str(helpers()))
bpy.ops.mocap.simplify()
check("Simplify adds no residual layer by default", not rig.animation_data.nla_tracks, str([t.name for t in rig.animation_data.nla_tracks]))
bpy.ops.mocap.restore_raw()
check("full Restore Raw: no snapshot, no solve base, no helpers left",
      raw_snapshot.get_raw_snapshot(rig) is None and raw_snapshot.get_solve_base(rig) is None and not helpers(), str(helpers()))
# another tool replaces the take: the old one is deleted, its helpers must not stay behind
bpy.ops.mocap.run_all()
before = helpers()
new = take.copy(); new.name = take.name + "_rebake"
for k in list(new.keys()):
    del new[k]
rig.animation_data.action = new
rig.animation_data.action_slot = new.slots[0]
check("the result line is not shown for another take", common.result_for(rig) == "", common.result_for(rig)[:60])
bpy.data.actions.remove(take)
registration._on_load()
check("helpers of a deleted take are purged (on load)", before and not helpers(), f"before {before} after {helpers()}")
print(f"R7 SUMMARY {sum(R)} / {len(R)}", flush=True)
