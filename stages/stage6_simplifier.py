"""Stage 6 - keyframe reduction with an optional residual detail layer.

Dense mocap curves are replaced by a sparse set of Bezier keys that stays within a tolerance of
the original (error-bounded fitting in the spirit of Roberts et al. 2018). All components of a
transform property share their key frames, so the Dope Sheet stays readable and quaternions stay
consistent. What the sparse curve cannot express is not thrown away: it goes to a *residual* action
on an NLA track below the (now ``COMBINE``-blended) active action. Sparse keys + residual reproduce
the cleaned motion (to within Blender's single precision NLA maths, < 0.05 degrees); muting the
track shows the plain sparse animation.
"""

from __future__ import annotations

from dataclasses import dataclass

import bpy
import numpy as np

from ..core import action_io, raw_snapshot
from ..core.dsp import solve_banded_spd
from ..core.math_quat import quat_conjugate, quat_hemisphere_continuous, quat_multiply, quat_normalize
from ..core.rig_inspector import inspect_rig

RESIDUAL_FLAG = "mc_is_residual"
RESIDUAL_LINK = "mc_residual"
TRACK_NAME = "MoCap Residual Detail"


@dataclass
class SimplifyResult:
    bones: int = 0
    curves: int = 0
    keys_before: int = 0
    keys_after: int = 0
    max_loc_error: float = 0.0
    max_rot_error: float = 0.0
    residual: bool = False

    def summary(self) -> str:
        pct = 100.0 * self.keys_after / max(self.keys_before, 1)
        tail = " + residual layer" if self.residual else ""
        return (f"{self.keys_before} -> {self.keys_after} keys ({pct:.1f}%) on {self.bones} bones, max error "
                f"{self.max_loc_error * 1000:.2f} mm / {np.degrees(self.max_rot_error):.2f} deg{tail}")


# ---------------------------------------------------------------------------
# Sparse cubic Hermite fitting (shared key frames for all columns)
# ---------------------------------------------------------------------------

def _basis(keys: np.ndarray, T: int):
    t = np.arange(T)
    seg = np.clip(np.searchsorted(keys, t, side="right") - 1, 0, len(keys) - 2)
    h = (keys[seg + 1] - keys[seg]).astype(np.float64)
    u = (t - keys[seg]) / h
    u2, u3 = u * u, u * u * u
    return seg, h, 2 * u3 - 3 * u2 + 1, u3 - 2 * u2 + u, -2 * u3 + 3 * u2, u3 - u2


def hermite_eval(keys: np.ndarray, values: np.ndarray, tangents: np.ndarray, T: int) -> np.ndarray:
    """Evaluate the cubic Hermite spline (what Blender's Bezier curve does with handles at 1/3) on ``0 .. T-1``."""
    seg, h, h00, h10, h01, h11 = _basis(keys, T)
    return (h00[:, None] * values[seg] + h01[:, None] * values[seg + 1]
            + (h * h10)[:, None] * tangents[seg] + (h * h11)[:, None] * tangents[seg + 1])


def fit_tangents(Y: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Least-squares tangents ``(K, D)`` for an interpolating Hermite spline with keys on the data."""
    T, K = len(Y), len(keys)
    seg, h, h00, h10, h01, h11 = _basis(keys, T)
    a, b = h * h10, h * h11
    r = Y - (h00[:, None] * Y[keys][seg] + h01[:, None] * Y[keys][seg + 1])
    diag = np.zeros(K)
    off = np.zeros(K)
    rhs = np.zeros((K, Y.shape[1]))
    np.add.at(diag, seg, a * a)
    np.add.at(diag, seg + 1, b * b)
    np.add.at(off, seg, a * b)
    np.add.at(rhs, seg, a[:, None] * r)
    np.add.at(rhs, seg + 1, b[:, None] * r)
    # prior: finite-difference slopes (decides where the data says nothing, e.g. adjacent keys)
    m0 = np.gradient(Y, axis=0)[keys]
    eps = 1e-6 * max(float(diag.mean()), 1e-12) + 1e-12
    diag += eps
    rhs += eps * m0
    if K <= 400:
        N = np.diag(diag) + np.diag(off[:-1], 1) + np.diag(off[:-1], -1)
        return np.linalg.solve(N, rhs)
    return solve_banded_spd(np.stack([diag, off]), rhs)


def fit_keys(Y: np.ndarray, tol: float, error_fn=None, forced=None, max_rounds: int = 60):
    """Choose key frames so that the Hermite spline stays within ``tol`` of ``Y (T, D)``.

    ``error_fn(fit, Y) -> (T,)`` defaults to the Euclidean distance per frame. Returns
    ``(keys (K,), tangents (K, D), error (T,))``.
    """
    Y = np.asarray(Y, dtype=np.float64)
    T = len(Y)
    if error_fn is None:
        error_fn = lambda F, G: np.linalg.norm(F - G, axis=1)
    forced_set = {0, T - 1} | {int(f) for f in (forced or []) if 0 <= int(f) < T}
    keys = np.array(sorted(forced_set), dtype=np.int64)

    def evaluate(k):
        m = fit_tangents(Y, k)
        return m, error_fn(hermite_eval(k, Y[k], m, T), Y)

    def grow(k):
        m, err = evaluate(k)
        for _ in range(max_rounds):
            if err.max() <= tol:
                break
            new = []
            for i in range(len(k) - 1):
                s, e = k[i] + 1, k[i + 1]
                if e > s:
                    j = int(np.argmax(err[s:e])) + s
                    if err[j] > tol:
                        new.append(j)
            if not new:
                break
            k = np.array(sorted(set(k.tolist()) | set(new)), dtype=np.int64)
            m, err = evaluate(k)
        return k, m, err

    keys, m, err = grow(keys)
    # prune keys that a neighbouring segment can absorb
    if len(keys) > 2:
        keep = keys.tolist()
        i = 1
        changed = False
        while i < len(keep) - 1:
            if keep[i] in forced_set:
                i += 1
                continue
            a, b = keep[i - 1], keep[i + 1]
            ka = np.array([0, b - a], dtype=np.int64)
            ia, ib = int(np.searchsorted(keys, a)), int(np.searchsorted(keys, b))
            local = hermite_eval(ka, Y[[a, b]], m[[ia, ib]], b - a + 1)
            if error_fn(local, Y[a : b + 1]).max() <= 0.6 * tol:
                del keep[i]
                changed = True
            else:
                i += 1
        if changed:
            keys, m, err = grow(np.array(keep, dtype=np.int64))
    return keys, m, err


def _quat_error(F: np.ndarray, G: np.ndarray) -> np.ndarray:
    Fn = quat_normalize(F)
    return 2.0 * np.arccos(np.clip(np.abs(np.sum(Fn * G, axis=1)), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Residual layer bookkeeping
# ---------------------------------------------------------------------------

def get_residual_action(rig):
    act = action_io.get_active_action(rig)
    if act is None:
        return None
    name = act.get(RESIDUAL_LINK, "")
    res = bpy.data.actions.get(name) if name else None
    return res if (res is not None and res.get(RESIDUAL_FLAG)) else None


def _residual_track(rig):
    ad = rig.animation_data
    if ad is None:
        return None
    for tr in ad.nla_tracks:
        if tr.name == TRACK_NAME:
            return tr
    return None


def has_residual(rig) -> bool:
    return get_residual_action(rig) is not None and _residual_track(rig) is not None


def residual_enabled(rig) -> bool:
    tr = _residual_track(rig)
    return tr is not None and not tr.mute


def set_residual_enabled(rig, enabled: bool) -> None:
    tr = _residual_track(rig)
    if tr is not None:
        tr.mute = not enabled
        rig.update_tag()


def _prop_kind(data_path: str) -> str:
    if data_path.endswith("rotation_quaternion"):
        return "QUAT"
    if data_path.endswith(".scale"):
        return "SCALE"
    return "ADD"


def _combine(kind: str, residual: np.ndarray, sparse: np.ndarray) -> np.ndarray:
    """What Blender's COMBINE blending makes of ``residual`` (lower) and ``sparse`` (upper)."""
    if kind == "QUAT":
        return quat_multiply(quat_normalize(residual), quat_normalize(sparse))
    if kind == "SCALE":
        return residual * sparse
    return residual + sparse


def _residual_of(kind: str, dense: np.ndarray, sparse: np.ndarray) -> np.ndarray:
    if kind == "QUAT":
        return quat_multiply(dense, quat_conjugate(quat_normalize(sparse)))
    if kind == "SCALE":
        return dense / np.where(np.abs(sparse) < 1e-9, 1.0, sparse)
    return dense - sparse


def merge_residual(rig) -> int:
    """Bake sparse keys + residual back into dense curves on the active action and remove the layer."""
    res = get_residual_action(rig)
    act = action_io.get_active_action(rig)
    track = _residual_track(rig)
    count = 0
    if res is not None and act is not None:
        f0, f1 = action_io.get_action_frame_range(rig, res)
        cb_res = action_io.get_active_channelbag(rig, res)
        cb = action_io.ensure_channelbag(rig, act)
        groups: dict[str, list] = {}
        for fc in (cb_res.fcurves if cb_res is not None else []):
            groups.setdefault(fc.data_path, []).append(fc)
        for path, fcs in groups.items():
            kind = _prop_kind(path)
            width = 4 if kind == "QUAT" else 3
            ident = (1.0, 0.0, 0.0, 0.0) if kind == "QUAT" else ((1.0,) * 3 if kind == "SCALE" else (0.0,) * 3)
            R = np.tile(np.array(ident, dtype=np.float64), (f1 - f0 + 1, 1))
            S = R.copy()
            present = []
            for i in range(width):
                fr = cb_res.fcurves.find(path, index=i)
                fs = cb.fcurves.find(path, index=i)
                if fr is not None:
                    R[:, i] = action_io.sample_fcurve_range(fr, f0, f1)
                if fs is not None:
                    S[:, i] = action_io.sample_fcurve_range(fs, f0, f1)
                    present.append(i)
            dense = _combine(kind, R, S)
            if kind == "QUAT":
                dense = quat_hemisphere_continuous(dense)[0]
            bone = action_io.fcurve_bone_name(fcs[0])
            for i in present:
                action_io.write_fcurve_range(cb, path, i, f0, dense[:, i], group_name=bone)
                count += 1
    ad = rig.animation_data
    if track is not None and ad is not None:
        ad.nla_tracks.remove(track)
    if ad is not None:
        ad.action_blend_type = "REPLACE"
    if res is not None:
        bpy.data.actions.remove(res)
    if act is not None and RESIDUAL_LINK in act:
        del act[RESIDUAL_LINK]
    rig.update_tag()
    return count


def _attach_residual(rig, res, f0: int) -> None:
    ad = rig.animation_data
    track = _residual_track(rig) or ad.nla_tracks.new()
    track.name = TRACK_NAME
    for st in list(track.strips):
        track.strips.remove(st)
    strip = track.strips.new(TRACK_NAME, int(f0), res)
    slot = action_io._find_slot(rig, res)
    if slot is not None and hasattr(strip, "action_slot"):
        strip.action_slot = slot
    # REPLACE at the bottom of the stack passes the residual through untouched. (A COMBINE strip
    # is raised to the power of its influence via acos(w) in single precision, which quantises
    # near-identity quaternions - exactly what a residual is - to steps of ~0.04 degrees.)
    strip.blend_type = "REPLACE"
    strip.extrapolation = "HOLD"
    strip.influence = 1.0
    track.mute = False
    ad.action_blend_type = "COMBINE"
    ad.action_extrapolation = "HOLD"
    ad.action_influence = 1.0


# ---------------------------------------------------------------------------
# Simplify
# ---------------------------------------------------------------------------

def simplify(rig, *, bone_names: list[str] | None = None, tol_loc: float = 0.002, tol_rot_deg: float = 0.5,
             residual: bool = True, forced_frames: dict[str, list[int]] | None = None, progress=None) -> SimplifyResult:
    """Reduce the keys of the active action. ``tol_loc`` is in metres for a 1.75 m character.

    ``forced_frames``: bone -> scene frames that must stay keys (e.g. contact transitions).
    """
    act = action_io.get_active_action(rig)
    if act is None:
        raise RuntimeError("The rig has no action")
    ad = rig.animation_data
    if residual and any(tr.name != TRACK_NAME and len(tr.strips) for tr in ad.nla_tracks):
        raise RuntimeError("The rig already uses NLA strips; a residual layer needs an empty NLA stack")
    if has_residual(rig):
        merge_residual(rig)
    raw_snapshot.ensure_raw_snapshot(rig)
    raw_snapshot.drop_solve_base(rig)            # the solve is final once the curves are sparse

    profile = inspect_rig(rig)
    unit, scale = profile.unit, max(profile.scale, 1e-9)
    f0, f1 = action_io.get_action_frame_range(rig)
    T = f1 - f0 + 1
    if T < 4:
        raise RuntimeError("The action is too short")
    names = action_io.get_animated_bone_names(rig)
    if bone_names is not None:
        names = [b for b in names if b in set(bone_names)]
    cb = action_io.ensure_channelbag(rig, act)
    tol_l = tol_loc * unit / scale
    tol_r = np.radians(tol_rot_deg)
    out = SimplifyResult(residual=residual)
    res_act = None
    cb_res = None
    if residual:
        res_act = bpy.data.actions.new(f".{act.name}.residual")
        res_act.use_fake_user = True
        res_act[RESIDUAL_FLAG] = True
        cb_res = action_io.ensure_channelbag(rig, res_act)

    for n_done, bone in enumerate(names):
        if progress is not None:
            progress(n_done / max(len(names), 1))
        ch = action_io.read_bone_local_channels(rig, bone, f0, f1)
        forced = [f - f0 for f in (forced_frames or {}).get(bone, [])]
        jobs = []
        if ch["has_loc"]:
            jobs.append(("location", ch["loc"], tol_l, None, "loc"))
        if ch["has_rot"]:
            if ch["rotation_mode"] == "QUATERNION":
                jobs.append(("rotation_quaternion", quat_hemisphere_continuous(ch["quat"])[0], tol_r, _quat_error, "rot"))
            elif ch["rotation_mode"] != "AXIS_ANGLE":
                jobs.append(("rotation_euler", ch["raw_rot"], tol_r,
                             lambda F, G: np.abs(F - G).max(axis=1), "rot"))
        if ch["has_scale"]:
            jobs.append(("scale", ch["scale"], 1e-3, None, "scale"))
        touched = False
        for prop, Y, tol, err_fn, what in jobs:
            path = action_io.bone_data_path(bone, prop)
            comps = [i for i in range(Y.shape[1]) if cb.fcurves.find(path, index=i) is not None]
            if not comps:
                continue
            keys, m, err = fit_keys(Y, tol, err_fn, forced)
            K = len(keys)
            h_next = np.diff(keys, append=keys[-1] + (keys[-1] - keys[-2])).astype(np.float64)
            h_prev = np.diff(keys, prepend=keys[0] - (keys[1] - keys[0])).astype(np.float64)
            frames = keys.astype(np.float64) + f0
            for i in comps:
                fc = cb.fcurves.find(path, index=i)
                out.keys_before += len(fc.keyframe_points)
                hl = np.stack([frames - h_prev / 3.0, Y[keys, i] - m[:, i] * h_prev / 3.0], axis=1)
                hr = np.stack([frames + h_next / 3.0, Y[keys, i] + m[:, i] * h_next / 3.0], axis=1)
                action_io.write_fcurve_sparse(cb, path, i, frames, Y[keys, i], hl, hr, group_name=bone)
                out.keys_after += K
                out.curves += 1
            if what == "loc":
                out.max_loc_error = max(out.max_loc_error, float(err.max()) * scale)
            elif what == "rot":
                out.max_rot_error = max(out.max_rot_error, float(err.max()))
            touched = True
            if residual:
                kind = _prop_kind(path)
                sparse = Y.copy()
                for i in comps:
                    sparse[:, i] = action_io.sample_fcurve_range(cb.fcurves.find(path, index=i), f0, f1)
                R = _residual_of(kind, Y, sparse)
                if kind == "QUAT":
                    R = quat_hemisphere_continuous(R)[0]
                    comps_r = range(4)
                else:
                    comps_r = comps
                for i in comps_r:
                    action_io.write_fcurve_range(cb_res, path, i, f0, R[:, i], group_name=bone)
        out.bones += int(touched)

    if residual:
        act[RESIDUAL_LINK] = res_act.name
        _attach_residual(rig, res_act, f0)
    rig.update_tag()
    if progress is not None:
        progress(1.0)
    return out
