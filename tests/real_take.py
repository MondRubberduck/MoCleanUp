"""Run the add-on on a real take, headless, and check that it did no harm.

    blender -b -y --python tests/real_take.py -- <file.blend> <rig name> [options]

    source=1            use this source tree instead of the installed extension (add --factory-startup)
    expect=<text>       fail unless the extension was loaded from a path that contains <text>
    preset=<name>       apply a Source preset first: OPTICAL, IMU or VIDEO
    set=<prop>:<value>,...   settings of the MoCap panel to change first (scene.mocap_cleanup)
    render=<dir>        contact sheets, raw on top and cleaned below (Workbench): feet close-up and whole body
    body=<frame,...>    frames of the body sheet (default: four spread over the take)
    feet=<frame,...>    frames of the feet sheet (default: around the deepest floor penetration)
    save=<copy.blend>   save the cleaned take as a COPY (plus <copy.blend>.npz with raw and cleaned curves)
    keys=0              skip the key reduction at the end

The file is opened and changed in memory only, it is never saved in place. Without ``source=1`` do
not pass ``--factory-startup``: the extension is loaded from the Blender profile (point
``BLENDER_USER_RESOURCES`` at a clean one) and nothing is imported from this repository but
``harm.py``, which only measures.

Prints what Clean Up Take reports, then what it did to the body (``harm.py``): how far feet were
carried, how far the hips were lowered, knee acceleration, joint pops, whether the IK and FK chains
still agree, where the character mesh rests. Ends with ``RESULT: ok`` only if every limit holds
and Restore Raw is exact.
"""
import importlib
import math
import os
import re
import sys
import time

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harm  # noqa: E402

PKG = "bl_ext.user_default.mocap_cleanup"
args = sys.argv[sys.argv.index("--") + 1:]
path, rig_name = args[0], args[1]
opts = dict(a.split("=", 1) for a in args[2:])

if opts.get("source", "0") not in ("", "0"):
    import _common  # noqa: E402,F401  (loads the source tree as ``mocap_cleanup``)
    PKG = "mocap_cleanup"
    if not hasattr(bpy.types.Scene, "mocap_cleanup"):
        sys.modules[PKG].register()
elif PKG not in bpy.context.preferences.addons:
    print("extension was not enabled in this profile; enabling:", bpy.ops.preferences.addon_enable(module=PKG))
pkg = importlib.import_module(PKG)
where = pkg.__file__.replace("\\", "/")
print("blender", bpy.app.version_string, "| package:", where)
if "expect" in opts:
    assert opts["expect"] in where, f"the extension was expected under {opts['expect']!r}"
action_io = importlib.import_module(PKG + ".core.action_io")
sampler = importlib.import_module(PKG + ".core.sampler")
ri = importlib.import_module(PKG + ".core.rig_inspector")
s1 = importlib.import_module(PKG + ".stages.stage1_diagnostics")


class mods:
    pass


mods.action_io, mods.sampler, mods.ri, mods.s1 = action_io, sampler, ri, s1

bpy.ops.wm.open_mainfile(filepath=path, load_ui=False)
scene = bpy.context.scene
rig = bpy.data.objects[rig_name]
st = scene.mocap_cleanup
st.rig = rig
bpy.context.view_layer.objects.active = rig
data = rig.mocap_cleanup
if "preset" in opts:
    st.preset = opts["preset"]
    print("  preset:", bpy.ops.mocap.apply_preset(), opts["preset"])
for item in filter(None, opts.get("set", "").split(",")):
    key, value = item.split(":", 1)
    cur = getattr(st, key)
    setattr(st, key, (value.lower() in ("1", "true", "on")) if isinstance(cur, bool) else type(cur)(value))
    print(f"  set {key} = {getattr(st, key)!r}")
f0, f1 = action_io.get_action_frame_range(rig)
T = f1 - f0 + 1


def curves():
    return {f"{fc.data_path}|{fc.array_index}": action_io.sample_fcurve_range(fc, f0, f1) for fc in action_io.iter_fcurves(rig)}


def sample():
    profile = ri.inspect_rig(rig)
    smp = sampler.sample_objects([(rig, ri.solver_bone_names(profile, rig.data.bones))], f0, f1)[rig.name]
    return profile, smp


def soles(profile, smp):
    return {S: ri.sole_contacts(leg, smp.world_m(leg.org_end)) for S, leg in profile.legs.items()}


def measure(profile, smp, pins, floor_z):
    """Slide of the pinned feet (cm/s: 90th percentile, maximum) and depth of the soles below the floor (mm)."""
    sl = soles(profile, smp)
    u = profile.unit
    p90, worst = harm.plant_slide(sl, pins, f0, scene.render.fps / scene.render.fps_base)
    depth = max((float(floor_z - pts[..., 2].min()) for pts in sl.values()), default=0.0)
    return p90 / u * 100, worst / u * 100, depth / u * 1000


# ---- rendering ------------------------------------------------------------------------------------------------------
def checker_floor(center, z, size=6.0, cell=0.05):
    n = int(size / cell)
    xs = np.arange(n + 1) * cell - 0.5 * size
    verts = [(float(x), float(y), 0.0) for y in xs for x in xs]
    faces = [(j * (n + 1) + i, j * (n + 1) + i + 1, (j + 1) * (n + 1) + i + 1, (j + 1) * (n + 1) + i) for j in range(n) for i in range(n)]
    me = bpy.data.meshes.new("mc_check_floor")
    me.from_pydata(verts, [], faces)
    col = me.color_attributes.new("Col", "FLOAT_COLOR", "CORNER")
    shade = np.array([(0.10 if (i + j) % 2 else 0.26) for j in range(n) for i in range(n)], dtype=np.float32)
    rgba = np.repeat(np.stack([shade, shade, shade * 1.1, np.ones_like(shade)], axis=1), 4, axis=0)
    col.data.foreach_set("color", rgba.ravel())
    ob = bpy.data.objects.new("mc_check_floor", me)
    scene.collection.objects.link(ob)
    ob.location = (float(center[0]), float(center[1]), float(z))
    return ob


def make_camera(name, ortho_scale):
    cam = bpy.data.objects.new(name, bpy.data.cameras.new(name))
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = ortho_scale
    cam.data.clip_start, cam.data.clip_end = 0.1, 100.0
    scene.collection.objects.link(cam)
    return cam


def aim(cam, target, direction, dist=10.0):
    d = Vector(direction).normalized()
    cam.location = Vector(target) + d * dist
    cam.rotation_euler = (-d).to_track_quat("-Z", "Y").to_euler()


def render(cam, frame, res, out):
    scene.camera = cam
    scene.render.resolution_x, scene.render.resolution_y = res
    scene.render.resolution_percentage = 100
    scene.frame_set(frame)
    scene.render.filepath = out
    bpy.ops.render.render(write_still=True)


def load(pathname):
    img = bpy.data.images.load(pathname)
    w, h = img.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    img.pixels.foreach_get(buf)
    bpy.data.images.remove(img)
    return buf.reshape(h, w, 4)


def sheet(rows, out):
    """``rows``: lists of tile paths, first row on top. A white line separates the rows."""
    lines = []
    for r in rows:
        tiles = [load(p) for p in r]
        lines.append(np.concatenate(tiles, axis=1))
    sep = np.ones((4, lines[0].shape[1], 4), dtype=np.float32)
    stack = []
    for k, line in enumerate(reversed(lines)):              # image rows are stored bottom-up
        if k:
            stack.append(sep)
        stack.append(line)
    arr = np.concatenate(stack, axis=0)
    img = bpy.data.images.new("mc_sheet", arr.shape[1], arr.shape[0], alpha=False)
    img.pixels.foreach_set(arr.ravel())
    img.filepath_raw = out
    img.file_format = "PNG"
    img.save()
    bpy.data.images.remove(img)
    print("  sheet", os.path.basename(out), arr.shape[1], "x", arr.shape[0])


raw = curves()
assert bpy.ops.mocap.analyze() == {"FINISHED"}
kinds = {}
for it in data.issues:
    kinds[it.kind] = kinds.get(it.kind, 0) + 1
print(f"rig {rig.name!r} frames {f0}-{f1} @ {scene.render.fps} fps, {len(raw)} curves | floor {st.floor_z:.3f} (auto {st.floor_auto})")
print("  analyze:", data.analysis_summary)
profile, smp0 = sample()
unit = profile.unit
skin = harm.character_meshes(scene, rig)
before = harm.take(mods, scene, rig, f0, f1, skin)

shots = None
if "render" in opts:
    out_dir = opts["render"]
    os.makedirs(out_dir, exist_ok=True)
    tag = rig.name.replace(".", "_")
    meshes = skin
    print("  character meshes:", [m.name for m in meshes])
    for ob in scene.objects:
        if ob.type in {"MESH", "CURVE", "SURFACE", "FONT", "META", "VOLUME", "POINTCLOUD", "CURVES"} and ob not in meshes:
            ob.hide_render = True
    for ob in meshes:
        ob.hide_render = False
    sl = soles(profile, smp0)
    depth = {S: (st.floor_z - p[..., 2].min(axis=1)) for S, p in sl.items()}
    S = max(depth, key=lambda k: depth[k].max())
    other = "L" if S == "R" else "R"
    worst = int(np.argmax(depth[S]))
    leg = profile.legs[S]
    ball = sl[S][:, 2:].mean(axis=1)
    heel = sl[S][:, :2].mean(axis=1)
    fwd = ball[worst] - heel[worst]
    fwd[2] = 0.0
    fwd /= max(np.linalg.norm(fwd), 1e-9)
    side = ball[worst] - sl[other][worst, 2:].mean(axis=0)
    side[2] = 0.0
    side -= fwd * float(side @ fwd)
    side = side / np.linalg.norm(side) if np.linalg.norm(side) > 1e-3 else np.array([fwd[1], -fwd[0], 0.0])
    torso = smp0.head_w(profile.torso_ctrl)
    floor = checker_floor(torso[:, :2].mean(axis=0), st.floor_z)
    sh = scene.display.shading
    scene.render.engine = "BLENDER_WORKBENCH"
    sh.light, sh.color_type = "STUDIO", "VERTEX"
    sh.show_cavity, sh.show_object_outline, sh.show_shadows = True, True, False
    try:
        scene.view_settings.view_transform = "Standard"
        scene.render.image_settings.file_format = "PNG"
        scene.display.render_aa = "8"
        if scene.world is None:
            scene.world = bpy.data.worlds.new("mc_world")
        scene.world.color = (0.03, 0.03, 0.04)
    except Exception as exc:
        print("  (render setup detail skipped:", exc, ")")
    feet_cam = make_camera("mc_feet_cam", 0.75 * unit)
    look = fwd * 0.75 + side * 0.66
    look = look / np.linalg.norm(look) * math.cos(math.radians(22.0))
    body_cam = make_camera("mc_body_cam", 2.5 * unit)
    feet_frames = [int(np.clip(worst + d, 0, T - 1)) + f0 for d in (-4, -2, 0, 2)]
    if "feet" in opts:
        feet_frames = [int(v) for v in opts["feet"].split(",")]
    body_frames = [f0 + int(round(q * (T - 1))) for q in (0.12, 0.37, 0.62, 0.87)]
    if "body" in opts:
        body_frames = [int(v) for v in opts["body"].split(",")]
    print(f"  close-up: foot {S}, deepest raw penetration {depth[S].max() / unit * 1000:.1f} mm at frame {worst + f0}; frames {feet_frames}; body frames {body_frames}")

    def shoot(label):
        files = {"feet": [], "body": []}
        for f in feet_frames:
            # same aim for raw and cleaned: where the raw foot is at this frame, at floor height
            at = ball[f - f0]
            aim(feet_cam, (at[0], at[1], st.floor_z + 0.10 * unit), (look[0], look[1], math.sin(math.radians(22.0))))
            p = os.path.join(out_dir, f"{tag}_feet_{label}_{f:04d}.png")
            render(feet_cam, f, (480, 360), p)
            files["feet"].append(p)
        for f in body_frames:
            c = torso[f - f0]
            body_dir = fwd * 0.8 + side * 0.6
            aim(body_cam, (c[0], c[1], st.floor_z + 0.95 * unit), (body_dir[0], body_dir[1], 0.18))
            p = os.path.join(out_dir, f"{tag}_body_{label}_{f:04d}.png")
            render(body_cam, f, (400, 480), p)
            files["body"].append(p)
        return files

    t = time.time()
    shots = {"raw": shoot("raw")}
    print(f"  rendered raw tiles in {time.time() - t:.1f}s")

t = time.time()
assert bpy.ops.mocap.run_all() == {"FINISHED"}
print(f"  one click ({time.time() - t:.1f}s): {data.last_result}")
profile, smp1 = sample()
b, a = measure(profile, smp0, data.pins, st.floor_z), measure(profile, smp1, data.pins, st.floor_z)
print(f"  pinned feet slide at (p90 / max) {b[0]:.1f} / {b[1]:.1f} -> {a[0]:.2f} / {a[1]:.2f} cm/s | soles below the floor {b[2]:.1f} -> {a[2]:.2f} mm")
spans = sorted(((p.lock_out - p.lock_in + 1, p.owner, p.lock_in, p.lock_out) for p in data.pins), reverse=True)
print(f"  {len(spans)} pins" + (f", the longest: {[(o, i, j) for _n, o, i, j in spans[:4]]}" if spans else ""))
cleaned = curves()
m = re.search(r"Largest change: (.+?) ([0-9.]+) cm at frame (\d+)", data.last_result)
if m:
    name, frame = m.group(1), int(m.group(3))
    j0, j1 = s1.body_joints(profile, smp0).get(name), s1.body_joints(profile, smp1).get(name)
    i = frame - f0
    if j0 is not None and 2 <= i < T - 2:
        def flick(x):
            """Distance of each frame from the middle of its two neighbours: large for one frame = a flick."""
            return np.linalg.norm(x[i - 2 : i + 3] - 0.5 * (x[i - 3 : i + 2] + x[i - 1 : i + 4]), axis=1) / unit * 100
        with np.printoptions(precision=1, suppress=True):
            print(f"  largest change, {name} at frame {frame}: cm off the line between the neighbouring frames, frames {frame - 2}..{frame + 2}: "
                  f"raw {flick(j0)} -> cleaned {flick(j1)}")
after = harm.take(mods, scene, rig, f0, f1, skin)
rows, failed = harm.compare(before, after, st.floor_z, pins=[p.to_data() for p in data.pins])
print(f"  what it did to the body (floor at {st.floor_z / unit * 100:.1f} cm" + (f", mesh {[m.name for m in skin]}" if skin else ", no character mesh") + "):")
harm.print_rows(rows)

if shots is not None:
    shots["clean"] = shoot("clean")
    for kind in ("feet", "body"):
        sheet([shots["raw"][kind], shots["clean"][kind]], os.path.join(out_dir, f"{tag}_{kind}_raw_top_cleaned_bottom.png"))
    for files in shots.values():
        for lst in files.values():
            for p in lst:
                os.remove(p)

if "save" in opts:
    bpy.ops.wm.save_as_mainfile(filepath=opts["save"], copy=True)
    np.savez(opts["save"] + ".npz", f0=f0, f1=f1, **{"raw:" + k: v for k, v in raw.items()}, **{"clean:" + k: v for k, v in cleaned.items()})
    print("  saved a COPY:", opts["save"], "| original still", bpy.data.filepath)

if opts.get("keys", "1") != "0":
    keys0 = sum(len(fc.keyframe_points) for fc in action_io.iter_fcurves(rig))
    r = bpy.ops.mocap.simplify()
    keys1 = sum(len(fc.keyframe_points) for fc in action_io.iter_fcurves(rig))
    print(f"  reduce keys: {r} {keys0} -> {keys1} keys | {data.last_result}")
    if bpy.ops.mocap.merge_residual.poll():
        print("  merge detail layer:", bpy.ops.mocap.merge_residual())
assert bpy.ops.mocap.restore_raw() == {"FINISHED"}
now = curves()
worst_diff = max((float(np.abs(now[k] - v).max()) for k, v in raw.items() if k in now), default=0.0)
print(f"  restore raw: max curve difference {worst_diff:.2e}, missing {len([k for k in raw if k not in now])}, extra {len([k for k in now if k not in raw])}")
if worst_diff >= 1e-6:
    failed.append("Restore Raw is not exact")
print("RESULT: ok" if not failed else "RESULT: FAILED - " + "; ".join(failed))
sys.stdout.flush()
