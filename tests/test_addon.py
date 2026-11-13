"""End-to-end test through the registered add-on: properties, operators, pin tracks, HUD data, save / reload."""

import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np

from _common import FPS, ROOT, fresh_rig, load_package, run_tests, sample_rig, set_limb_props, synth_walk
from test_contacts import corrupt_feet, plant_slide

import mocap_cleanup
from mocap_cleanup import props
from mocap_cleanup.core import action_io, raw_snapshot, runtime
from mocap_cleanup.core.world_edit import write_world_groups
from mocap_cleanup.overlays import draw, pin_tracks
from mocap_cleanup.stages import stage6_simplifier as s6
from mocap_cleanup.ui import autosolve, common, operators

_registered = False


def ensure_registered():
    global _registered
    if not _registered:
        mocap_cleanup.register()
        _registered = True


def make_scene(n_frames=110):
    ensure_registered()
    rig = fresh_rig()
    set_limb_props(rig, ik_stretch=0.0, pole_vector=True)
    gt = synth_walk(rig, n_frames=n_frames)
    corrupt_feet(rig, gt)
    f0, f1 = gt["f0"], gt["f1"]
    ch = action_io.read_bone_local_channels(rig, "hand_ik.L", f0, f1)
    loc = ch["loc"].copy()
    loc[50] += (0.06, 0.0, 0.05)                                   # a tracking spike
    action_io.write_bone_local_channels(rig, "hand_ik.L", f0, loc=loc)
    bpy.context.view_layer.objects.active = rig
    rig.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    st = bpy.context.scene.mocap_cleanup
    st.rig = rig
    return rig, gt, st


def test_registration_and_ui_definitions():
    ensure_registered()
    for cls in operators.CLASSES:
        mod, name = cls.bl_idname.split(".")
        assert hasattr(getattr(bpy.ops, mod), name)
        getattr(getattr(bpy.ops, mod), name).get_rna_type()
    assert hasattr(bpy.types.Scene, "mocap_cleanup") and hasattr(bpy.types.Object, "mocap_cleanup")
    for name in ("MOCAP_PT_main_view3d", "MOCAP_PT_main_graph", "MOCAP_PT_main_dopesheet", "MOCAP_UL_pins"):
        assert hasattr(bpy.types, name), name
    # every icon used anywhere exists in this Blender version
    icons = set(bpy.types.UILayout.bl_rna.functions["prop"].parameters["icon"].enum_items.keys())
    used = set()
    for folder, _dirs, files in os.walk(ROOT):
        if "tests" in folder or "__pycache__" in folder:
            continue
        for fn in files:
            if fn.endswith(".py"):
                src = open(os.path.join(folder, fn), encoding="utf-8").read()
                used |= set(re.findall(r'icon="([A-Z0-9_]+)"', src))
                used |= set(re.findall(r'"([A-Z0-9_]+)", \d\),?\n', src))          # enum item icons
    for table in (panels_icons(),):
        used |= table
    missing = sorted(used - icons)
    assert not missing, f"unknown icons: {missing}"
    # manifest
    import tomllib
    with open(os.path.join(ROOT, "blender_manifest.toml"), "rb") as fh:
        man = tomllib.load(fh)
    assert man["id"] == "mocap_cleanup" and len(man["tagline"]) <= 64 and man["type"] == "add-on"
    print(f"   {len(operators.CLASSES) + 2} operators, {len(used)} icons checked")


def panels_icons():
    from mocap_cleanup.ui import panels
    out = set(panels.ISSUE_ICONS.values()) | set(panels.MODE_ICONS.values())
    out |= {item[3] for item in props.MODE_ITEMS} | {item[3] for item in props.STAGE_ITEMS}
    return out


def test_pipeline_through_operators():
    rig, gt, st = make_scene()
    data = rig.mocap_cleanup
    f0, f1 = gt["f0"], gt["f1"]
    # bend the right arm (the synthetic walk keeps it almost straight) so that a pinned hand has slack
    profile, smp = sample_rig(rig, ["hand_ik.R"])
    W = smp.world_m("hand_ik.R").copy()
    W[:, :3, 3] += 0.3 * (smp.head_w("ORG-upper_arm.R") - W[:, :3, 3])
    assert write_world_groups(rig, [{"hand_ik.R": W}], f0, f1, first_sample=smp) == ["hand_ik.R"]
    raw_hand = action_io.read_bone_local_channels(rig, "hand_ik.L", f0, f1)["loc"].copy()

    assert bpy.ops.mocap.analyze() == {"FINISHED"}
    kinds = {i.kind for i in data.issues}
    assert {"SPIKE", "SKATE", "PENETRATION"} <= kinds, kinds
    assert data.analysis_summary and not data.analysis_stale
    spike_idx = next(i for i, it in enumerate(data.issues) if it.kind == "SPIKE")
    bpy.context.scene.frame_set(f0)
    data.active_issue = spike_idx
    assert bpy.context.scene.frame_current == f0 + 50
    assert action_io.selected_pose_bone_names(rig) == ["hand_ik.L"]
    bpy.context.scene.frame_set(f0)
    assert bpy.ops.mocap.issue_jump(direction=1) == {"FINISHED"} and bpy.context.scene.frame_current > f0

    # fix just this issue, then everything
    data.active_issue = spike_idx
    assert bpy.ops.mocap.issue_fix() == {"FINISHED"}
    hand = action_io.read_bone_local_channels(rig, "hand_ik.L", f0, f1)["loc"]
    assert np.linalg.norm(hand[50] - raw_hand[50]) > 0.05 and np.abs(hand[40] - raw_hand[40]).max() < 1e-7
    assert raw_snapshot.get_raw_snapshot(rig) is not None
    assert not any(i.kind == "SPIKE" for i in data.issues)
    assert bpy.ops.mocap.repair() == {"FINISHED"}

    st.filter_method, st.cutoff_hz = "WHITTAKER", 9.0
    assert bpy.ops.mocap.estimate_cutoff() == {"FINISHED"} and 1.0 <= st.cutoff_hz <= 15.0
    st.cutoff_hz = 9.0
    assert bpy.ops.mocap.filter() == {"FINISHED"}

    profile, before = sample_rig(rig)
    slide_before = plant_slide(profile, before, gt)
    assert bpy.ops.mocap.detect_contacts() == {"FINISHED"}
    n_auto = sum(1 for p in data.pins if p.auto)
    assert n_auto >= 6 and raw_snapshot.get_solve_base(rig) is not None
    profile, after = sample_rig(rig)
    slide_after = plant_slide(profile, after, gt)
    print(f"   operators: {len(data.issues)} issues, {n_auto} plants, slide {slide_before * 1000:.1f} -> {slide_after * 1000:.2f} mm")
    assert slide_after < 0.002 < slide_before

    # live update: editing a pin re-solves (the timer is flushed by hand in background mode)
    pin = next(p for p in data.pins if p.owner == "leg.L" and p.lock_out - p.lock_in > 10)
    crc = action_io.channels_checksum(rig, [action_io.bone_data_path("foot_ik.L", "location")])
    pin.lock_out -= 6
    assert autosolve.flush() == 1
    assert action_io.channels_checksum(rig, [action_io.bone_data_path("foot_ik.L", "location")]) != crc
    pin.lock_out += 6
    autosolve.flush()
    assert action_io.channels_checksum(rig, [action_io.bone_data_path("foot_ik.L", "location")]) == crc

    # "Pin Here": world pin on the selected hand control
    common.select_only_bone(rig, "hand_ik.R")
    bpy.context.scene.frame_set(f0 + 30)
    st.pin_length = 15
    assert bpy.ops.mocap.pin_add(mode="WORLD") == {"FINISHED"}
    new = data.pins[data.active_pin]
    assert (new.owner, new.mode, new.lock_in, new.lock_out, new.use_reference) == ("arm.R", "WORLD", f0 + 30, f0 + 44, True)
    _, held = sample_rig(rig)
    hand_w = held.head_w("ORG-hand.R")
    # the body walks on: the hand holds for as long as the arm can reach, then soft IK lets it go smoothly
    dev = np.linalg.norm(hand_w[30:45] - hand_w[30], axis=1)
    print("   world pin on a walking body, hand deviation per frame (mm):", np.round(dev * 1000, 1).tolist())
    assert dev[:4].max() < 0.003
    hold = dev.max()
    bpy.context.scene.frame_set(f0 + 36)
    assert bpy.ops.mocap.pin_edge(edge="OUT") == {"FINISHED"} and new.lock_out == f0 + 36
    n = len(data.pins)
    assert bpy.ops.mocap.pin_remove() == {"FINISHED"} and len(data.pins) == n - 1
    _, freed = sample_rig(rig)
    free = np.linalg.norm(freed.head_w("ORG-hand.R")[30:45] - freed.head_w("ORG-hand.R")[30], axis=1).max()
    assert free > 0.1 and hold < 0.5 * free, (hold, free)

    # HUD data for the current frame
    bpy.context.scene.frame_set(gt["plants"]["L"][1][0] + 4)
    hud = draw.hud_state(bpy.context, rig)
    assert hud["com"] is not None and 0.7 < hud["com"].z < 1.3
    assert hud["planted"]["leg.L"] == "PIN" and len(hud["soles"]["leg.L"]) == 4
    assert any(mode == "FLOOR" and locked for _pos, mode, locked, _t in hud["pins"])

    # solve can be taken off again, pins stay
    assert bpy.ops.mocap.clear_solve() == {"FINISHED"} and raw_snapshot.get_solve_base(rig) is None
    profile, off = sample_rig(rig)
    assert plant_slide(profile, off, gt) > 0.01 and len(data.pins) == n - 1
    assert bpy.ops.mocap.solve() == {"FINISHED"}
    assert bpy.ops.mocap.pin_clear(auto_only=True) == {"FINISHED"} and len(data.pins) == 0


def test_run_all_simplify_restore_reload():
    rig, gt, st = make_scene()
    data = rig.mocap_cleanup
    f0, f1 = gt["f0"], gt["f1"]
    raw = {b: action_io.read_bone_local_channels(rig, b, f0, f1) for b in ("hand_ik.L", "foot_ik.L", "torso")}
    st.preset = "IMU"
    assert bpy.ops.mocap.apply_preset() == {"FINISHED"} and st.pipe_dedrift and st.filter_method == "WHITTAKER"
    st.preset = "OPTICAL"
    bpy.ops.mocap.apply_preset()
    assert st.filter_method == "MODIFIED_SINC" and not st.pipe_dedrift
    assert bpy.ops.mocap.run_all() == {"FINISHED"}
    left = {i.kind for i in data.issues}
    print(f"   one click: {data.last_result}")
    assert not left, left
    assert len(data.pins) >= 6

    # keys stage: residual layer blocks the other stages until it is merged
    keys_before = sum(len(fc.keyframe_points) for fc in action_io.iter_fcurves(rig))
    profile, dense = sample_rig(rig)
    # without a layer, Merge / Show Detail are greyed out instead of failing when pressed (QA N3)
    assert not bpy.ops.mocap.merge_residual.poll() and not bpy.ops.mocap.toggle_residual.poll()
    bpy.context.scene.mocap_cleanup.simplify_residual = True      # opt-in since the layer is off by default
    assert bpy.ops.mocap.simplify() == {"FINISHED"} and s6.has_residual(rig)
    keys_after = sum(len(fc.keyframe_points) for fc in action_io.iter_fcurves(rig))
    assert keys_after < 0.5 * keys_before
    profile, sparse = sample_rig(rig)
    assert np.abs(sparse.head_w("ORG-foot.L") - dense.head_w("ORG-foot.L")).max() < 5e-4
    try:
        bpy.ops.mocap.filter()
        raise AssertionError("filter must refuse while a residual layer exists")
    except RuntimeError as exc:
        assert "residual" in str(exc).lower()
    assert bpy.ops.mocap.merge_residual.poll() and bpy.ops.mocap.toggle_residual.poll()
    assert bpy.ops.mocap.toggle_residual() == {"FINISHED"} and not s6.residual_enabled(rig)
    assert bpy.ops.mocap.merge_residual() == {"FINISHED"} and not s6.has_residual(rig)
    assert not bpy.ops.mocap.merge_residual.poll()

    # save, reload: pins, issues and snapshots survive, runtime caches do not
    path = os.path.join(tempfile.gettempdir(), "mc_test_addon_reload.blend")
    n_pins = len(data.pins)
    bpy.ops.wm.save_as_mainfile(filepath=path)
    bpy.ops.wm.open_mainfile(filepath=path)
    rig = bpy.data.objects["rig"]
    data = rig.mocap_cleanup
    st = bpy.context.scene.mocap_cleanup
    assert st.rig == rig and len(data.pins) == n_pins and data.analysis_summary == "No problems found"
    assert raw_snapshot.get_raw_snapshot(rig) is not None
    bpy.context.view_layer.objects.active = rig
    assert bpy.ops.mocap.solve() == {"FINISHED"}

    # restore raw: the take is the untouched capture again
    assert bpy.ops.mocap.restore_raw() == {"FINISHED"}
    for b, ch in raw.items():
        now = action_io.read_bone_local_channels(rig, b, f0, f1)
        assert np.abs(now["loc"] - ch["loc"]).max() < 1e-6, b
    assert raw_snapshot.get_solve_base(rig) is None
    os.remove(path)


def test_pin_track_layout_and_drag():
    rig, gt, st = make_scene(n_frames=60)
    data = rig.mocap_cleanup
    st.auto_solve = False
    with props.suspend_updates():
        for owner, a, b in (("leg.L", 10, 30), ("leg.L", 40, 50), ("arm.R", 20, 25)):
            p = data.pins.add()
            p.owner, p.lock_in, p.lock_out, p.blend_in, p.blend_out = owner, a, b, 4, 5

    class V2D:
        def view_to_region(self, x, y, clip=False):
            return (100.0 + 10.0 * x, 50.0 + y)

        def region_to_view(self, x, y):
            return ((x - 100.0) / 10.0, y - 50.0)

    class Region:
        width, height, view2d, type = 1200, 400, V2D(), "WINDOW"

    lay = pin_tracks.layout(bpy.context, Region(), rig)
    assert [r[0] for r in lay["rows"]] == ["arm.R", "leg.L"] and len(lay["bars"]) == 3
    bar = next(b for b in lay["bars"] if b["index"] == 0)
    assert abs(bar["xi"] - (100 + 10 * 9.5)) < 1e-6 and abs(bar["xe"] - (100 + 10 * 35.5)) < 1e-6
    ymid = 0.5 * (bar["y0"] + bar["y1"])
    assert pin_tracks.hit_test(lay, 300.0, ymid)[1] == "MOVE"
    assert pin_tracks.hit_test(lay, bar["xi"] + 1, ymid)[1] == "IN"
    assert pin_tracks.hit_test(lay, bar["xo"] - 1, ymid)[1] == "OUT"
    assert pin_tracks.hit_test(lay, bar["xs"] + 2, ymid)[1] == "BLEND_IN"
    assert pin_tracks.hit_test(lay, bar["xe"] - 2, ymid)[1] == "BLEND_OUT"
    assert pin_tracks.hit_test(lay, 300.0, 5.0)[0] is None
    hit, _ = pin_tracks.hit_test(lay, 100 + 10 * 45, ymid)
    assert hit["index"] == 1

    class Drag:
        pass

    pin = data.pins[0]
    apply = pin_tracks.MOCAP_OT_pin_track_edit._apply
    for zone, delta, expect in (("MOVE", 3, (13, 33, 4, 5)), ("IN", 2, (12, 30, 2, 5)), ("IN", 40, (30, 30, 0, 5)),
                                ("OUT", -5, (10, 25, 4, 10)), ("BLEND_IN", -3, (10, 30, 7, 5)), ("BLEND_OUT", 4, (10, 30, 4, 9))):
        d = Drag()
        d._zone, d._start = zone, (10, 30, 4, 5)
        with props.suspend_updates():
            pin.lock_in, pin.lock_out, pin.blend_in, pin.blend_out = d._start
        apply(d, pin, delta)
        assert (pin.lock_in, pin.lock_out, pin.blend_in, pin.blend_out) == expect, (zone, delta, (pin.lock_in, pin.lock_out, pin.blend_in, pin.blend_out))
    st.auto_solve = True


def test_unregister_and_register_again():
    global _registered
    ensure_registered()
    mocap_cleanup.unregister()
    assert not hasattr(bpy.types.Scene, "mocap_cleanup") and not hasattr(bpy.types, "MOCAP_PT_main_view3d")
    assert not runtime.REPORTS and not runtime.SOLVE_SAMPLES
    mocap_cleanup.register()
    assert hasattr(bpy.types.Scene, "mocap_cleanup")
    mocap_cleanup.unregister()
    _registered = False


if __name__ == "__main__":
    run_tests([test_registration_and_ui_definitions, test_pipeline_through_operators, test_run_all_simplify_restore_reload,
               test_pin_track_layout_and_drag, test_unregister_and_register_again])
