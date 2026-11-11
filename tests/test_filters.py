"""Stage 2 tests: frequency responses, boundary behaviour, SO(3) filtering, action-level filtering."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from _common import FPS, fresh_rig, run_tests, synth_walk

from mocap_cleanup.core import action_io
from mocap_cleanup.core import math_quat as mq
from mocap_cleanup.stages import stage2_filters as s2


def _gain(method, f, fc, fps=60.0, **kw):
    t = np.arange(1800)
    sig = np.sin(2 * np.pi * f / fps * t)
    out = s2.apply_vector_filter(sig, method, fc, fps, **kw)
    return float(np.sqrt(np.mean(out[400:1400] ** 2) / np.mean(sig[400:1400] ** 2)))


def test_frequency_responses():
    fc = 6.0
    for method in ("WHITTAKER", "MODIFIED_SINC", "BUTTERWORTH", "BSPLINE"):
        g_pass = _gain(method, 0.25 * fc, fc)
        g_cut = _gain(method, fc, fc)
        g_stop = _gain(method, 3.0 * fc, fc)
        print(f"   {method:14s} gain @0.25fc={g_pass:.3f} @fc={g_cut:.3f} @3fc={g_stop:.4f}")
        assert g_pass > 0.97, (method, g_pass)
        lo, hi = (0.35, 0.95) if method == "BSPLINE" else (0.60, 0.80)   # spline response is phase dependent
        assert lo < g_cut < hi, (method, g_cut)
        assert g_stop < 0.08, (method, g_stop)
    # MS kernel: stopband suppression is far better than a plain Savitzky-Golay of similar width
    assert _gain("MODIFIED_SINC", 4.0 * fc, fc) < 0.01
    # order-3 Whittaker is steeper than order-2
    assert _gain("WHITTAKER", 2 * fc, fc, order=3) < _gain("WHITTAKER", 2 * fc, fc, order=2)


def test_boundaries_and_noise():
    rng = np.random.default_rng(3)
    T = 150
    ramp = np.stack([np.linspace(0, 3, T), np.linspace(2, -1, T)], -1)
    for method in s2.FILTER_METHODS:
        out = s2.apply_vector_filter(ramp, method, 6.0, 30.0)
        assert np.abs(out - ramp).max() < 2e-3, f"{method} bends a straight line: {np.abs(out - ramp).max()}"
    t = np.linspace(0, 2 * np.pi, T)
    peak = 0.8 * np.exp(-0.5 * ((t - np.pi) / 0.25) ** 2)
    gt = np.stack([np.sin(t) + peak, np.cos(2 * t), 0.5 * np.sin(3 * t)], -1)
    noisy = gt + rng.normal(0, 0.04, gt.shape)
    raw = np.sqrt(np.mean((noisy - gt) ** 2))
    for method in s2.FILTER_METHODS:
        out = s2.apply_vector_filter(noisy, method, 3.0, 30.0, euro_beta=4.0)
        rmse = np.sqrt(np.mean((out - gt) ** 2))
        end_err = max(np.abs(out[0] - gt[0]).max(), np.abs(out[-1] - gt[-1]).max())
        print(f"   {method:14s} rmse {raw:.4f} -> {rmse:.4f}  end error {end_err:.3f}")
        assert rmse < (0.85 if method == "ONE_EURO" else 0.6) * raw, (method, rmse, raw)
        assert end_err < 0.11, (method, end_err)   # ~2.5 sigma of the raw noise: no boundary blow-up
    apex = int(np.argmax(gt[:, 0]))
    ms = s2.filter_modified_sinc(noisy, 3.0, 30.0)
    assert abs(ms[apex, 0] - gt[apex, 0]) < 0.06
    # weighted Whittaker bridges a corrupted block when its weight is zero
    bad = gt.copy()
    bad[60:70] += 3.0
    w = np.ones(T)
    w[60:70] = 0.0
    rec = s2.filter_whittaker(bad, 8.0, 30.0, weights=w)
    assert np.abs(rec[60:70] - gt[60:70]).max() < 0.08, np.abs(rec[60:70] - gt[60:70]).max()


def test_so3_intrinsic():
    rng = np.random.default_rng(4)
    T = 160
    ang = np.linspace(0, 4 * np.pi, T)                       # two full turns
    gt = mq.quat_exp(np.stack([np.zeros(T), np.zeros(T), 0.5 * ang], -1))
    noisy = mq.quat_multiply(gt, mq.quat_exp(rng.normal(0, 0.03, (T, 3))))
    fn = lambda v: s2.filter_whittaker(v, 2.5, 30.0)
    out = s2.filter_quaternions_intrinsic(noisy, fn)
    e0 = np.mean(mq.quat_angle_between(noisy, gt))
    e1 = np.mean(mq.quat_angle_between(out, gt))
    assert e1 < 0.55 * e0, (e0, e1)
    assert np.all(np.sum(out[:-1] * out[1:], axis=1) > 0)
    batch = np.stack([noisy, noisy[::-1]], axis=1)             # (T, 2, 4)
    out_b = s2.filter_quaternions_intrinsic(batch, fn)
    assert np.allclose(np.abs(np.sum(out_b[:, 0] * out, axis=1)), 1.0, atol=1e-9)


def test_cutoff_estimate():
    rng = np.random.default_rng(5)
    fps, T = 60.0, 900
    t = np.arange(T) / fps
    clean = np.stack([np.sin(2 * np.pi * 1.2 * t) + 0.4 * np.sin(2 * np.pi * 2.8 * t), np.cos(2 * np.pi * 0.7 * t)], -1)
    sigma = 0.02
    fc, noise = s2.estimate_cutoff(clean + rng.normal(0, sigma, clean.shape), fps)
    print(f"   estimated cutoff {fc:.2f} Hz, noise {noise:.4f} (true {sigma})")
    assert 3.0 <= fc <= 12.0, fc
    assert 0.5 * sigma < noise < 1.6 * sigma, noise


def test_filter_action():
    rig = fresh_rig()
    gt = synth_walk(rig, n_frames=120)
    f0, f1, T = gt["f0"], gt["f1"], gt["T"]
    rng = np.random.default_rng(6)
    bones = ["torso", "foot_ik.L", "hand_ik.R"]
    clean = {b: action_io.read_bone_local_channels(rig, b, f0, f1) for b in bones}
    noisy = {}
    for b in bones:
        loc = clean[b]["loc"] + rng.normal(0, 0.006, (T, 3))
        quat = mq.quat_multiply(clean[b]["quat"], mq.quat_exp(rng.normal(0, 0.01, (T, 3))))
        noisy[b] = (loc, quat)
        action_io.write_bone_local_channels(rig, b, f0, loc=loc, quat=quat)

    # range-limited: outside the window nothing may change
    n = s2.filter_action(rig, "WHITTAKER", 4.0, frame_start=f0 + 40, frame_end=f0 + 80, blend_frames=5)
    assert n >= 3
    cur = action_io.read_bone_local_channels(rig, "torso", f0, f1)["loc"]
    assert np.array_equal(cur[:35], noisy["torso"][0][:35].astype(np.float32).astype(np.float64))
    assert np.abs(cur[45:75] - clean["torso"]["loc"][45:75]).mean() < 0.7 * np.abs(noisy["torso"][0][45:75] - clean["torso"]["loc"][45:75]).mean()

    for method in s2.FILTER_METHODS:
        for b in bones:
            action_io.write_bone_local_channels(rig, b, f0, loc=noisy[b][0], quat=noisy[b][1])
        s2.filter_action(rig, method, 5.0, bone_names=bones, euro_beta=2.0)
        for b in bones:
            ch = action_io.read_bone_local_channels(rig, b, f0, f1)
            e_loc0 = np.sqrt(np.mean((noisy[b][0] - clean[b]["loc"]) ** 2))
            e_loc1 = np.sqrt(np.mean((ch["loc"] - clean[b]["loc"]) ** 2))
            e_rot0 = np.mean(mq.quat_angle_between(noisy[b][1], clean[b]["quat"]))
            e_rot1 = np.mean(mq.quat_angle_between(ch["quat"], clean[b]["quat"]))
            assert e_loc1 < 0.85 * e_loc0, (method, b, e_loc0, e_loc1)
            assert e_rot1 < 0.9 * e_rot0, (method, b, e_rot0, e_rot1)

    # adaptive: only frames with low confidence are touched
    for b in bones:
        action_io.write_bone_local_channels(rig, b, f0, loc=noisy[b][0], quat=noisy[b][1])
    conf = {b: np.ones(T) for b in bones}
    conf["torso"][50:60] = 0.1
    s2.filter_action(rig, "WHITTAKER", 6.0, bone_names=bones, confidence=conf, adaptive=True)
    cur = action_io.read_bone_local_channels(rig, "torso", f0, f1)["loc"]
    ref = noisy["torso"][0].astype(np.float32).astype(np.float64)
    assert np.array_equal(cur[:40], ref[:40]) and np.array_equal(cur[70:], ref[70:])
    assert np.abs(cur[50:60] - ref[50:60]).max() > 1e-4
    cur_f = action_io.read_bone_local_channels(rig, "foot_ik.L", f0, f1)["loc"]
    assert np.array_equal(cur_f, noisy["foot_ik.L"][0].astype(np.float32).astype(np.float64))


if __name__ == "__main__":
    run_tests([test_frequency_responses, test_boundaries_and_noise, test_so3_intrinsic, test_cutoff_estimate, test_filter_action])
