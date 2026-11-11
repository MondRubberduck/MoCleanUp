"""What real takes taught: in-place cycles, pose-to-pose snaps, twisted toe bones, floating takes, unexpected errors.

Every test here reproduces a situation that was found by running the add-on on retargeted takes
(a Mixamo clip and a 60 fps pose-to-pose take on Rig-Bridge rigs).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import bpy
import numpy as np

from _common import FPS, fresh_rig, run_tests, sample_rig, set_limb_props, sole_world, synth_walk
from test_contacts import corrupt_feet, plant_slide
import test_addon as ta

from mocap_cleanup.core import action_io
from mocap_cleanup.core.rig_inspector import commanded_end
from mocap_cleanup.stages import solver
from mocap_cleanup.stages import stage1_diagnostics as s1
from mocap_cleanup.stages import stage4_contacts_pins as s4
from mocap_cleanup.ui import operators

SOLE = np.array([[-0.045, 0.06, 0.0], [0.045, 0.06, 0.0], [-0.05, -0.16, 0.0], [0.05, -0.16, 0.0]])   # heel, heel, ball, ball


def _feet(T, fps, stance_velocity, swing_height=0.10, stance_height=(0.0, 0.0), cycle=30, stride=0.7):
    """Sole points ``{owner: (T, 4, 3)}`` of a two-footed cycle. During stance the foot moves with
    ``stance_velocity`` (m/s along Y; 0 = a true plant), during swing it arcs through the air."""
    t = np.arange(T)
    out = {}
    for k, (owner, phase) in enumerate((("leg.L", 0.0), ("leg.R", 0.5))):
        s = ((t / cycle) + phase) % 1.0
        stance = s < 0.5
        u = np.where(stance, s / 0.5, (s - 0.5) / 0.5)
        step = stance_velocity * (0.5 * cycle / fps)                       # travel during one stance
        y_stance = -0.5 * step + step * u
        y_swing = 0.5 * step - step * (0.5 - 0.5 * np.cos(np.pi * u))     # back to the start of the next stance
        walk = stride * np.floor((t / cycle) + phase) if stance_velocity == 0.0 else 0.0
        y = np.where(stance, y_stance, y_swing) - walk - np.where(stance | (stance_velocity != 0.0), 0.0,
                                                                 stride * (0.5 - 0.5 * np.cos(np.pi * u)))
        z = np.where(stance, stance_height[k], stance_height[k] + swing_height * np.sin(np.pi * u) ** 2)
        centre = np.stack([np.full(T, 0.1 * (1 if k else -1)), y, z], axis=1)
        out[owner] = centre[:, None, :] + SOLE[None]
    return out


def _covered(pins, owner, f0, T):
    m = np.zeros(T, dtype=bool)
    for p in pins:
        if p.owner == owner:
            m[max(p.lock_in - f0, 0): p.lock_out - f0 + 1] = True
    return m


def test_only_feet_that_hold_still_are_plants():
    fps, T, floor = 30.0, 120, s4.Floor(0.0)
    # a walk: every stance is found
    pins, _ = s4.detect_contacts(_feet(T, fps, 0.0), floor, 1, fps)
    assert sum(p.owner == "leg.L" for p in pins) >= 3 and sum(p.owner == "leg.R" for p in pins) >= 3, len(pins)
    # retarget skate: the stance foot creeps at 0.2 m/s (10 cm per step). Still a plant - the one to fix.
    pins, _ = s4.detect_contacts(_feet(T, fps, 0.2), floor, 1, fps)
    assert sum(p.owner == "leg.L" for p in pins) >= 3, len(pins)
    # an in-place run: the foot on the floor travels backwards at 2 m/s. Nothing may be planted.
    for speed in (0.9, 2.0, 3.5):
        pins, _ = s4.detect_contacts(_feet(T, fps, -speed), floor, 1, fps)
        assert not pins, (speed, [(p.owner, p.lock_in, p.lock_out) for p in pins])
    # a stance foot that floats 3.5 cm above the floor while the other one is in the air is still a plant
    pins, _ = s4.detect_contacts(_feet(T, fps, 0.0, stance_height=(0.035, 0.0)), floor, 1, fps)
    assert sum(p.owner == "leg.L" for p in pins) >= 3, [(p.owner, p.lock_in, p.lock_out) for p in pins]

    # a shuffle: planted, dragged 40 cm along the floor at 1.5 m/s (1 cm up), planted again
    T = 70
    y = np.zeros(T)
    y[30:38] = np.linspace(0.0, 0.4, 10)[1:-1]
    y[38:] = 0.4
    z = np.zeros(T)
    z[30:38] = 0.01
    left = np.stack([np.zeros(T), y, z], axis=1)[:, None, :] + SOLE[None]
    right = np.stack([np.full(T, 0.2), np.zeros(T), np.full(T, 0.3)], axis=1)[:, None, :] + SOLE[None]   # held in the air
    pins, _ = s4.detect_contacts({"leg.L": left, "leg.R": right}, floor, 1, fps)
    cover = _covered(pins, "leg.L", 1, T)
    assert sum(p.owner == "leg.L" for p in pins) == 2, [(p.lock_in, p.lock_out) for p in pins]
    assert not cover[31:37].any() and cover[5:25].all() and cover[45:65].all()


def test_fast_moves_are_not_spikes():
    floor = 0.0004
    # pose-to-pose: hold 10 frames, move to the next pose within 4 frames (ease out), hold again ...
    rng = np.random.default_rng(3)
    poses = rng.normal(0.0, 0.25, (8, 3))
    x = []
    for a, b in zip(poses[:-1], poses[1:]):
        x += [a] * 10
        x += [a + (b - a) * w for w in (0.45, 0.75, 0.92, 0.98)]
    x = np.array(x + [poses[-1]] * 10)
    spike, *_ = s1.outlier_scores(x, floor)
    assert not spike.any(), np.flatnonzero(spike)
    # ... arriving with an overshoot that settles (still a move, not an excursion)
    x2 = []
    for a, b in zip(poses[:-1], poses[1:]):
        x2 += [a] * 8
        x2 += [a + (b - a) * w for w in (0.45, 0.80, 1.02, 1.08, 1.06, 1.03, 1.01)]
    x2 = np.array(x2 + [poses[-1]] * 10)
    spike, *_ = s1.outlier_scores(x2, floor)
    assert not spike.any(), np.flatnonzero(spike)

    # excursions come back: a 2 frame jump on a slow drift, a 1 frame jump inside a fast move
    t = np.arange(120.0)
    slow = np.stack([0.002 * t, 0.05 * np.sin(t / 14.0), 0.03 * np.cos(t / 9.0)], axis=1)
    slow[60:62] += (0.05, -0.03, 0.04)
    spike, *_ = s1.outlier_scores(slow, floor)
    assert set(np.flatnonzero(spike)) >= {60, 61} and spike.sum() <= 4, np.flatnonzero(spike)
    fast = np.stack([0.02 * t, 0.3 * np.sin(t / 10.0), np.zeros(len(t))], axis=1)       # 0.6 m/s at 30 fps
    fast[45] += (0.0, 0.0, 0.06)
    spike, *_ = s1.outlier_scores(fast, floor)
    assert spike[45] and spike.sum() <= 3, np.flatnonzero(spike)


def _pitched_toe(T, max_pitch, depth=0.005, toe=0.06):
    """A toe joint whose foot tips forward from flat to ``max_pitch`` while the toe tip stays ``depth`` below z = 0."""
    th = np.linspace(0.0, max_pitch, T)
    fwd = np.stack([np.zeros(T), -np.cos(th), -np.sin(th)], axis=1)       # toe direction, nose down
    up = np.stack([np.zeros(T), -np.sin(th), np.cos(th)], axis=1)         # sole normal
    tip = np.stack([np.zeros(T), -0.1 - 0.01 * np.arange(T), np.full(T, -depth)], axis=1)
    joint = tip - toe * fwd
    axis = np.tile([1.0, 0.0, 0.0], (T, 1))
    return joint, tip, axis, up


def _rotate_about(joint, tip, axis, gamma):
    d = tip - joint
    c, s = np.cos(gamma)[:, None], np.sin(gamma)[:, None]
    return joint + d * c + np.cross(axis, d) * s + axis * np.sum(axis * d, axis=1, keepdims=True) * (1.0 - c)


def test_toe_bend_is_stable():
    floor = s4.Floor(0.0)
    # heel rising until the foot stands on its toes and tips over (110 degrees): the bend never changes sign
    joint, tip, axis, up = _pitched_toe(60, np.radians(110.0))
    g = solver.toe_lift_angle(joint, tip, axis, floor, 3, up=up)
    assert np.all(g >= -1e-9) or np.all(g <= 1e-9), np.degrees(g).round(1)
    assert np.abs(np.diff(g)).max() < np.radians(15.0), np.degrees(np.abs(np.diff(g)).max())
    assert np.abs(g).max() <= np.radians(60.0) + 1e-9
    lifted = _rotate_about(joint, tip, axis, g)
    shallow = np.linspace(0.0, 110.0, 60) < 45.0
    assert lifted[shallow, 2].min() > -1e-4, lifted[shallow, 2].min()         # reachable: the tip is on the floor
    assert np.all(lifted[:, 2] >= tip[:, 2] - 1e-9)                           # never pushed further down

    # a toe bone that points along the hinge axis (twisted by a retarget) cannot be lifted by it: no correction
    T = 40
    rng = np.random.default_rng(1)
    joint = np.stack([np.zeros(T), np.zeros(T), np.full(T, 0.001)], axis=1)
    tip = joint + np.array([-0.058, 0.0, -0.004]) + rng.normal(0.0, 0.003, (T, 3)) * (0.0, 1.0, 0.3)
    assert tip[:, 2].max() < 0.0
    g = solver.toe_lift_angle(joint, tip, np.tile([1.0, 0.0, 0.0], (T, 1)), floor, 3, up=np.tile([0.0, 0.0, 1.0], (T, 1)))
    assert np.abs(g).max() < 1e-9, np.degrees(np.abs(g).max())
    # a hair of penetration gives a hair of bend, not the maximum
    joint, tip, axis, up = _pitched_toe(30, np.radians(20.0), depth=0.0003)
    g = solver.toe_lift_angle(joint, tip, axis, floor, 3, up=up)
    assert np.abs(g).max() < np.radians(2.0), np.degrees(np.abs(g).max())


def test_floating_take_is_cleaned_without_setting_the_floor():
    """A retargeted take whose feet rest 6 cm above the grid: one click still plants them."""
    rig, gt, st = ta.make_scene()
    bpy.ops.object.mode_set(mode="OBJECT")
    rig.location.z = 0.06
    bpy.context.view_layer.update()
    data = rig.mocap_cleanup
    assert st.floor_auto and st.floor_z == 0.0 and st.floor_object is None
    assert bpy.ops.mocap.analyze() == {"FINISHED"}
    assert "SKATE" in {i.kind for i in data.issues}, data.analysis_summary
    assert abs(st.floor_z - 0.06) < 0.008, st.floor_z
    assert bpy.ops.mocap.run_all() == {"FINISHED"}
    profile, smp = sample_rig(rig)
    slide = plant_slide(profile, smp, gt)
    rest = min(float(sole_world(profile, smp, S)[..., 2].min()) for S in ("L", "R"))
    print(f"   feet at {st.floor_z * 100:.1f} cm: {sum(p.auto for p in data.pins)} plants, slide {slide * 1000:.2f} mm, lowest sole {rest * 100:.2f} cm")
    assert sum(p.auto for p in data.pins) >= 6 and slide < 0.002 and abs(rest - st.floor_z) < 0.001
    assert not {"SKATE", "PENETRATION"} & {i.kind for i in data.issues}, data.analysis_summary

    # by hand: the floor stays where it is put, and the panel can tell where the feet are
    bpy.ops.mocap.restore_raw()
    st.floor_auto = False
    st.floor_z = 0.0
    bpy.ops.mocap.analyze()
    assert st.floor_z == 0.0 and data.has_floor_estimate and abs(data.floor_estimate - 0.06) < 0.008
    assert bpy.ops.mocap.detect_contacts() == {"FINISHED"} and not any(p.auto for p in data.pins)   # 6 cm up: no contact
    assert bpy.ops.mocap.floor_from_feet() == {"FINISHED"} and abs(st.floor_z - 0.06) < 0.008
    assert bpy.ops.mocap.detect_contacts() == {"FINISHED"} and sum(p.auto for p in data.pins) >= 6
    st.floor_auto = True


def test_toes_of_a_planted_foot_rest_on_the_floor():
    floor = s4.Floor(0.0)
    T = 40
    # a foot on its ball (joint 2 cm up), toes pointing 20 degrees into the air
    th = np.radians(20.0)
    joint = np.tile([0.0, 0.0, 0.02], (T, 1))
    tip = joint + 0.06 * np.array([0.0, -np.cos(th), np.sin(th)])
    axis = np.tile([1.0, 0.0, 0.0], (T, 1))
    up = np.tile([0.0, 0.0, 1.0], (T, 1))
    weight = np.clip(np.minimum(np.arange(T) - 4, 35 - np.arange(T)) / 4.0, 0.0, 1.0)     # a pin: blends in, holds, blends out
    g = solver.toe_rest_angle(joint, tip, axis, floor, weight, 3, up=up)
    lifted = solver.toe_lift_angle(joint, tip - (0.0, 0.0, 0.06), axis, floor, 2, up=up)      # the same toe, under the floor
    assert np.all(g[:5] == 0.0) and np.all(g[35:] == 0.0)                 # not planted: left alone
    print(f"   toe laid down by {np.degrees(np.abs(g).max()):.1f} deg, steepest step {np.degrees(np.abs(np.diff(g)).max()):.1f} deg per frame")
    assert np.all(g * np.sign(lifted[15]) <= 1e-12)                       # the other way round than lifting
    laid = _rotate_about(joint, tip, axis, g)
    assert np.abs(laid[15:25, 2]).max() < 1e-6, laid[15:25, 2]            # on the floor
    assert laid[:, 2].min() > -1e-9                                       # never in it
    assert np.abs(np.diff(g)).max() < 0.6 * np.abs(g).max()               # eased in and out
    # next to a frame where the toe touches already, easing must not push it through the floor
    mixed = tip.copy()
    mixed[:15] = joint[:15] + (0.0, -np.sqrt(0.06 ** 2 - 0.02 ** 2), -0.02)
    g2 = solver.toe_rest_angle(joint, mixed, axis, floor, weight, 3, up=up)
    assert _rotate_about(joint, mixed, axis, g2)[:, 2].min() > -1e-9 and np.abs(g2[22:26]).min() > np.radians(39.0)
    # a toe that hovers for two frames while the foot comes down is not worth a correction
    brief = np.zeros(T)
    brief[10:12] = 1.0
    assert np.all(solver.toe_rest_angle(joint, tip, axis, floor, brief, 3, up=up) == 0.0)
    # a toe that touches the floor already, and one the hinge cannot bring down: nothing / as far as it goes
    flat = joint + (0.0, -0.06, -0.02)
    assert np.all(solver.toe_rest_angle(joint, flat, axis, floor, weight, 2, up=up) == 0.0)
    high = solver.toe_rest_angle(joint + (0.0, 0.0, 0.2), tip + (0.0, 0.0, 0.2), axis, floor, weight, 2, up=up)
    assert np.abs(high).max() <= np.radians(45.0) + 1e-9


def test_a_foot_is_planted_where_it_stays():
    # one contact, two holds: 5 frames here, 12 frames 6 cm further on. The plant is the second place.
    from mocap_cleanup.core.dsp import weighted_geometric_median
    a, b = np.array([0.0, 0.0, 0.0]), np.array([0.06, 0.01, 0.0])
    x = np.concatenate([np.tile(a, (5, 1)), [a + (b - a) * u for u in (0.25, 0.5, 0.75)], np.tile(b, (12, 1))])
    m = weighted_geometric_median(x, np.ones(len(x)))
    assert np.linalg.norm(m - b) < 1e-4, m
    assert np.linalg.norm(x.mean(axis=0) - b) > 0.015                     # the mean is where the foot never was
    # a steady creep: the middle of the path, as the mean
    line = np.linspace(0.0, 1.0, 21)[:, None] * np.array([0.03, 0.0, 0.0])
    assert np.linalg.norm(weighted_geometric_median(line, np.ones(21)) - line.mean(axis=0)) < 1e-6
    # weights decide, a single point is itself
    assert np.allclose(weighted_geometric_median(np.stack([a, b, b + 1.0]), np.array([0.0, 1.0, 0.0])), b)


def _roll_onto_the_ball(rig, gt, angle):
    """Key a heel roll on the left foot: flat for the first half of every stance, then onto the ball."""
    f0, T = gt["f0"], gt["T"]
    roll = np.zeros(T)
    for a, b in gt["plants"]["L"]:
        n = b - a + 1
        u = np.clip((np.arange(n) / max(n - 1, 1) - 0.4) / 0.4, 0.0, 1.0)
        roll[a - f0 : b - f0 + 1] = angle * u * u * (3.0 - 2.0 * u)
    quat = np.stack([np.cos(roll / 2), np.sin(roll / 2), np.zeros(T), np.zeros(T)], axis=1)
    action_io.write_bone_local_channels(rig, "foot_heel_ik.L", f0, quat=quat)
    return roll


def test_foot_roll_is_part_of_the_pose():
    """A foot rolled onto its ball has its ankle a hand's width from the IK control. The control may
    then be out of reach while the leg is bent: that is neither over-extension nor a sliding plant."""
    rig = fresh_rig()
    set_limb_props(rig, ik_stretch=0.0, pole_vector=True)
    gt = synth_walk(rig, n_frames=100)
    f0, f1, T = gt["f0"], gt["f1"], gt["T"]
    profile, flat = sample_rig(rig)
    leg = profile.legs["L"]
    assert leg.ik_target == "MCH-thigh_ik_target.L" and profile.arms["L"].ik_target, (leg.ik_target, profile.arms["L"].ik_target)
    assert leg.ik_aim_bone == "MCH-foot_roll.L", leg.ik_aim_bone       # Rigify clamps the target itself (Limit Distance)

    # which way lifts the heel?
    for angle in (np.radians(38.0), -np.radians(38.0)):
        roll = _roll_onto_the_ball(rig, gt, angle)
        profile, smp = sample_rig(rig)
        sole = sole_world(profile, smp, "L")                       # heel, heel, ball, ball
        rolled = np.abs(roll) > 0.9 * abs(angle)
        if sole[rolled, :2, 2].min() > 0.02 and np.abs(sole[rolled, 2:, 2]).max() < 0.004:
            break
    else:
        raise AssertionError("no heel roll found on the test rig")

    # raise the hips until the control is out of reach in the rolled frames; the ankle is not
    root, ctrl, ankle = smp.head_w(leg.org_upper), smp.head_w(leg.ik_ctrl), smp.head_w(leg.org_end)
    need = 1.02 * leg.max_reach - np.linalg.norm(ctrl - root, axis=1)[rolled].max()
    torso = action_io.read_bone_local_channels(rig, "torso", f0, f1)["loc"].copy()
    torso[:, 2] += max(need, 0.0) * np.abs(roll / angle)
    action_io.write_bone_local_channels(rig, "torso", f0, loc=torso)
    profile, smp = sample_rig(rig)
    root, ctrl, ankle = smp.head_w(leg.org_upper), smp.head_w(leg.ik_ctrl), smp.head_w(leg.org_end)
    r_ctrl = np.linalg.norm(ctrl - root, axis=1) / leg.max_reach
    r_ankle = np.linalg.norm(ankle - root, axis=1) / leg.max_reach
    aim = commanded_end(leg, smp)
    print(f"   rolled frames: control at {r_ctrl[rolled].max():.3f} of the leg length, ankle at {r_ankle[rolled].max():.3f}, "
          f"ankle {np.linalg.norm(ankle - ctrl, axis=1)[rolled].max() * 100:.1f} cm from the control")
    assert r_ctrl[rolled].max() > 1.01 and r_ankle[rolled].max() < 0.962, (r_ctrl[rolled].max(), r_ankle[rolled].max())   # 0.965: soft IK starts
    # the leg reaches what it is asked to reach (to the precision of Blender's IK solver)
    assert np.linalg.norm(aim - ankle, axis=1)[rolled].max() < 5e-4

    report = s1.analyze(rig)
    over = [i for i in report.issues if i.kind == "HYPEREXT" and i.bone == leg.ik_ctrl
            and rolled[max(i.frame_start - f0, 0) : i.frame_end - f0 + 1].any()]
    assert not over, [(i.frame_start, i.frame_end, i.message) for i in over]

    # plant the foot for the whole stance: the ball stays, the heel keeps rolling, the control is not pulled in
    a, b = gt["plants"]["L"][1]
    pins = [s4.KinematicPin(owner="leg.L", mode="FLOOR", lock_in=a + 1, lock_out=b - 1, blend_in=2, blend_out=2)]
    res = solver.solve(rig, pins, solver.SolveSettings())
    profile, out = sample_rig(rig)
    i, j = a + 1 - f0, b - 1 - f0 + 1
    ball0, ball1 = sole_world(profile, smp, "L")[:, 2:].mean(axis=1), sole_world(profile, out, "L")[:, 2:].mean(axis=1)
    heel1 = sole_world(profile, out, "L")[:, :2].mean(axis=1)
    slide = float(np.linalg.norm(ball1[i:j] - ball1[i], axis=1).max())
    moved = float(np.linalg.norm(out.head_w(leg.org_end) - ankle, axis=1)[i:j].max())
    print(f"   pinned: ball moves {slide * 1000:.2f} mm, heel rises {heel1[i:j, 2].max() * 100:.1f} cm, "
          f"ankle {moved * 1000:.2f} mm from where it was | {res.summary()}")
    assert slide < 0.001, slide
    assert heel1[i:j, 2].max() > 0.02                                     # still rolled
    toe = out.point_w(leg.org_toe, leg.toe_tip_local)
    assert np.abs(toe[i:j, 2]).max() < 0.0025, np.abs(toe[i:j, 2]).max()  # toes on the floor
    # nothing to correct, nothing corrected: the flat and the rolled part of the stance stay as they are
    # (in between, tilts of a few degrees are flattened: a plant does not wobble)
    shift = np.linalg.norm(out.head_w(leg.org_end) - ankle, axis=1)
    settled = (np.abs(roll) < np.radians(0.5)) | (np.abs(roll) > np.radians(10.0))
    settled[:i] = settled[j:] = False
    assert shift[settled].max() < 0.001 and moved < 0.005, (shift[settled].max(), moved)
    assert not res.warnings, res.warnings


def test_unexpected_errors_are_reported_not_swallowed():
    rig, gt, st = ta.make_scene(n_frames=70)
    original = operators.common.run_analyze

    def boom(context, rig):
        raise KeyError("no such control")

    operators.common.run_analyze = boom
    try:
        try:
            bpy.ops.mocap.analyze()
            raise AssertionError("the error must reach the user")
        except RuntimeError as exc:
            text = str(exc)
            assert "KeyError" in text and "no such control" in text and "console" in text, text
    finally:
        operators.common.run_analyze = original
    assert bpy.ops.mocap.analyze() == {"FINISHED"}
    # things the user can act on stay short and cancel the operator
    action_io.get_active_action(rig)
    rig.animation_data.action = None
    try:
        bpy.ops.mocap.filter()
        raise AssertionError("no action: the operator must refuse")
    except RuntimeError as exc:
        assert "Traceback" not in str(exc) and "console" not in str(exc), str(exc)


if __name__ == "__main__":
    run_tests([test_only_feet_that_hold_still_are_plants, test_fast_moves_are_not_spikes, test_toe_bend_is_stable,
               test_toes_of_a_planted_foot_rest_on_the_floor, test_a_foot_is_planted_where_it_stays,
               test_foot_roll_is_part_of_the_pose, test_floating_take_is_cleaned_without_setting_the_floor,
               test_unexpected_errors_are_reported_not_swallowed])
