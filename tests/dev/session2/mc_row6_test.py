"""Row 6 checks. argv: <part: rig|flip>"""
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import env

part = sys.argv[sys.argv.index("--") + 1]
env.enable("mc")
M = env.PK["mc"]
common = importlib.import_module(M + ".ui.common")
C = bpy.context
st = C.scene.mocap_cleanup
R = []
def check(name, ok, info=""):
    R.append(ok); print(f"R6 [{'PASS' if ok else 'FAIL'}] {name} {info}", flush=True)

def activate(ob):
    for o in C.view_layer.objects:
        o.select_set(False)
    ob.hide_set(False); ob.select_set(True); C.view_layer.objects.active = ob

if part == "rig":
    rig = bpy.data.objects["Armature_rig"]; sk = bpy.data.objects["Armature"]; mesh = bpy.data.objects["Beta_Surface"]
    st.rig = None
    activate(mesh)
    check("character mesh selected -> its control rig", common.get_rig(C) == rig, str(common.get_rig(C)))
    activate(sk)
    check("skeleton selected -> its control rig", common.get_rig(C) == rig, str(common.get_rig(C)))
    activate(mesh)
    r = bpy.ops.mocap.use_active_rig()
    check("Use Active on the mesh picks the control rig", st.rig == rig and "control rig" in rig.mocap_cleanup.last_result,
          rig.mocap_cleanup.last_result)
    activate(bpy.data.objects["Ayana_Root"]); st.rig = None
    check("a plain skeleton stays itself", common.get_rig(C) == bpy.data.objects["Ayana_Root"])
    # M5: nothing flagged on controls of a chain that is never active
    st.rig = rig; activate(rig)
    st.preset = "OPTICAL"; bpy.ops.mocap.apply_preset()
    bpy.ops.mocap.run_all()                      # the issues left after Clean Up Take (QA M5)
    bones = {i.bone for i in rig.mocap_cleanup.issues}
    idle = {b for b in bones if b.startswith(("upper_arm_ik_target", "hand_ik", "thigh_fk", "shin_fk", "foot_fk"))}
    check("no issues on the inactive chains", not idle, f"{sorted(bones)[:8]} | idle {sorted(idle)}")
    # M7: Pin Here on a simplified take refuses before storing anything
    bpy.ops.mocap.simplify()
    n = len(rig.mocap_cleanup.pins)
    for pb in rig.pose.bones:
        pb.select = False
    bpy.ops.object.mode_set(mode="POSE")
    rig.pose.bones["hand_ik.L"].select = True
    try:
        r = bpy.ops.mocap.pin_add(mode="WORLD")
        err = ""
    except RuntimeError as e:
        r, err = None, str(e)
    check("Pin Here refuses on a simplified take and stores no pin", r != {"FINISHED"} and len(rig.mocap_cleanup.pins) == n
          and "residual" in err, err.strip()[:100])
else:
    mocap = bpy.data.objects["Ayana_Root"]
    st.rig = mocap; activate(mocap)
    bpy.ops.mocap.analyze()
    data = mocap.mocap_cleanup
    idx = next((i for i, it in enumerate(data.issues) if it.kind == "FLIP" and it.frame_start == 2475), None)
    check("the persistent 116 deg hand step is flagged", idx is not None)
    data.active_issue = idx
    try:
        r = bpy.ops.mocap.issue_fix()
        err = ""
    except RuntimeError as e:
        r, err = None, str(e)
    check("Fix This Issue says it could not repair it", r != {"FINISHED"} and "Could not repair" in err, err.strip()[:140])
print(f"R6 SUMMARY {sum(R)} / {len(R)}", flush=True)
