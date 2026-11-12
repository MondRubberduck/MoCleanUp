"""Where does one click spend its time? argv: <rig>"""
import sys, os, time, importlib, functools
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy
import env

rig_name = sys.argv[sys.argv.index("--") + 1]
env.enable("mc")
M = env.PK["mc"]
C = bpy.context
st = C.scene.mocap_cleanup
T = {}
N = {}


def wrap(mod, name, label=None):
    m = importlib.import_module(M + mod)
    fn = getattr(m, name)
    lab = label or name

    @functools.wraps(fn)
    def w(*a, **k):
        t0 = time.perf_counter()
        try:
            return fn(*a, **k)
        finally:
            T[lab] = T.get(lab, 0.0) + time.perf_counter() - t0
            N[lab] = N.get(lab, 0) + 1
    setattr(m, name, w)
    return w


wrap(".core.sampler", "sample_objects")
wrap(".core.sampler", "sample_frames")
wrap(".core.floor_fit", "mesh_lowest")
ops = importlib.import_module(M + ".ui.operators")
for fn in ("put_on_floor", "settle_on_floor", "detect_contacts"):
    wrap(".ui.operators", fn)
common = importlib.import_module(M + ".ui.common")
for fn in ("run_analyze", "run_solve", "sync_chains", "begin_edit", "end_edit"):
    wrap(".ui.common", fn)
cs = importlib.import_module(M + ".core.chain_sync")
wrap(".core.chain_sync", "_ik_follows_fk")
wrap(".core.chain_sync", "remember")
s1 = importlib.import_module(M + ".stages.stage1_diagnostics"); wrap(".stages.stage1_diagnostics", "repair")
s2 = importlib.import_module(M + ".stages.stage2_filters"); wrap(".stages.stage2_filters", "filter_action")
rig = bpy.data.objects[rig_name]
for o in C.view_layer.objects:
    o.select_set(False)
rig.hide_set(False); C.view_layer.objects.active = rig; st.rig = rig
st.preset = "OPTICAL"; bpy.ops.mocap.apply_preset()
t0 = time.perf_counter()
bpy.ops.mocap.run_all()
tot = time.perf_counter() - t0
print(f"PROF total {tot:.1f}s", flush=True)
for k, v in sorted(T.items(), key=lambda kv: -kv[1]):
    print(f"PROF   {k:22s} {v:7.1f}s  calls={N[k]}", flush=True)
