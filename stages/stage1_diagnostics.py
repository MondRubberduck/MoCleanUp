"""Stage 1 - diagnostics, error marking and structural repair.

Detectors (each produces :class:`Issue` intervals and feeds a per-bone confidence curve):

* ``FLIP``         quaternion hemisphere flips, Euler wraps and one-frame rotation jumps
* ``SPIKE``        isolated outliers (Hampel deviation confirmed by the 4th difference); a velocity
                   step such as a foot strike is *not* a spike and is left alone
* ``JITTER``       bursts of high-frequency noise (running median of the 4th difference)
* ``FROZEN``       duplicated frames (dropouts) inside otherwise moving data
* ``HYPEREXT``     limb reach above the safe ratio, ``KNEE_POP`` hinge snaps at full extension
* ``PENETRATION``  soles below the floor, ``SKATE`` planted feet that slide

Repairs: flips are rewritten continuously; spikes and dropouts are cut out and reconstructed with
C2 minimum-jerk bridges (rotations in the tangent space of SO(3), SQUAD for very large turns), long
gaps with low-rank inter-joint completion (Gloersen & Federolf 2016) when other joints still carry
information.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import bpy
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from ..core import action_io, sampler
from ..core.dsp import close_small_gaps, find_runs, whittaker_lambda_for_cutoff, whittaker_smooth
from ..core.math_quat import (
    quat_angle_between,
    quat_conjugate,
    quat_exp,
    quat_hemisphere_continuous,
    quat_log,
    quat_multiply,
    quat_normalize,
    quat_slerp,
    quat_squad_bridge,
    quintic_blend_weight,
)
from ..core.rig_inspector import commanded_end, inspect_rig, sole_contacts, solver_bone_names
from . import stage4_contacts_pins as s4
from .stage2_filters import estimate_cutoff, scene_fps

ISSUE_LABELS = {
    "FLIP": "Rotation flip",
    "SPIKE": "Tracking spike",
    "JITTER": "Jitter",
    "FROZEN": "Frozen frames",
    "HYPEREXT": "Over-extension",
    "KNEE_POP": "Joint pop",
    "PENETRATION": "Floor penetration",
    "SKATE": "Foot slide",
}
# Which tool fixes what (used by the "Fix This Issue" button)
ISSUE_FIX = {
    "FLIP": "REPAIR", "SPIKE": "REPAIR", "FROZEN": "REPAIR", "JITTER": "FILTER",
    "HYPEREXT": "SOLVE", "KNEE_POP": "SOLVE", "PENETRATION": "SOLVE", "SKATE": "CONTACTS",
}


@dataclass
class Issue:
    kind: str
    frame_start: int
    frame_end: int
    bone: str = ""                 # control whose curves show the problem ("" = whole body)
    severity: float = 0.5          # 0..1
    message: str = ""

    @property
    def label(self) -> str:
        return ISSUE_LABELS.get(self.kind, self.kind)


@dataclass
class DiagnosticReport:
    rig_name: str
    f0: int
    f1: int
    fps: float
    issues: list[Issue] = field(default_factory=list)
    confidence: dict[str, np.ndarray] = field(default_factory=dict)      # bone -> (T,) 1 = clean
    spike_masks: dict[str, np.ndarray] = field(default_factory=dict)     # bone -> (T,) frames to rebuild
    spike_parts: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)   # bone -> {'loc' | 'quat': (T,)}
    spike_sure: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)    # ... the ones beyond doubt
    joints: dict[str, np.ndarray] = field(default_factory=dict)          # 'Hips', 'Knee L', ... -> world (T, 3)
    frozen_runs: list[tuple[int, int]] = field(default_factory=list)     # index runs, whole body
    frozen_bones: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    flip_bones: list[str] = field(default_factory=list)
    noise: dict[str, tuple[float, float]] = field(default_factory=dict)  # bone -> (loc rms, rot rms [rad])
    suggested_cutoff: float = 0.0
    floor_estimate: float | None = None
    checksum: str = ""

    @property
    def T(self) -> int:
        return self.f1 - self.f0 + 1

    def count(self, kind: str) -> int:
        return sum(1 for i in self.issues if i.kind == kind)

    def summary(self) -> str:
        if not self.issues:
            return "No problems found"
        parts = [f"{self.count(k)} {ISSUE_LABELS[k].lower()}" for k in ISSUE_LABELS if self.count(k)]
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Signal level detectors
# ---------------------------------------------------------------------------

def fourth_difference(x: np.ndarray) -> np.ndarray:
    """``|x[t] - cubic prediction from its 4 neighbours|`` per frame: norm over columns, shape ``(T,)``."""
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    T = len(x)
    e = np.zeros(T)
    if T >= 5:
        d = (x[:-4] - 4.0 * x[1:-3] + 6.0 * x[2:-2] - 4.0 * x[3:-1] + x[4:]) / 6.0
        e[2:-2] = np.linalg.norm(d, axis=1)
    return e


def hampel_deviation(x: np.ndarray, radius: int = 3) -> np.ndarray:
    """Distance of every sample from the running median of its window, ``(T,)``."""
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    if len(x) < 3:
        return np.zeros(len(x))
    padded = np.pad(x, ((radius, radius), (0, 0)), mode="edge")
    med = np.median(sliding_window_view(padded, 2 * radius + 1, axis=0), axis=-1)
    return np.linalg.norm(x - med, axis=1)


def running_median(y: np.ndarray, radius: int = 3) -> np.ndarray:
    padded = np.pad(np.asarray(y, dtype=np.float64), radius, mode="edge")
    return np.median(sliding_window_view(padded, 2 * radius + 1), axis=-1)


def rotation_tangent_curve(quat: np.ndarray) -> np.ndarray:
    """Unwrapped rotation curve ``(T, 3)``: cumulative sum of the relative rotation vectors."""
    q = quat_hemisphere_continuous(quat)[0]
    cum = np.zeros((len(q), 3))
    if len(q) > 1:
        cum[1:] = np.cumsum(quat_log(quat_multiply(quat_conjugate(q[:-1]), q[1:])), axis=0)
    return cum


def _unpredictable(x: np.ndarray, s: int, en: int, noise: float, reach: int = 4, ratio: float = 3.0) -> bool:
    """Leave-out test for the frames ``s..en`` of ``x (T, D)`` (see ``_leave_out_margin``).

    Too few neighbours to tell (the ends of a take) count as unpredictable.
    """
    margin = _leave_out_margin(x, s, en, noise, reach)
    return bool(np.isnan(margin) or margin > ratio)


SURE_SPIKE = 10.0
UNSURE_NOTE = "? Could be a fast move: Clean Up Take leaves it alone, Fix This Issue rebuilds it"


def sure_spikes(x: np.ndarray, mask: np.ndarray, noise: float, margin: float = SURE_SPIKE) -> np.ndarray:
    """The frames of ``mask`` (spike runs) that are spikes beyond doubt.

    A glitch in otherwise smooth data is hundreds of times further from its neighbours' curve than
    they are themselves. A frame inside a fast, uneven move is a few times further: it could be the
    move. Only the first kind is safe to rebuild without asking; so is nothing at the ends of a
    take, where there are too few neighbours to tell.
    """
    out = np.zeros(len(mask), dtype=bool)
    for s, en in find_runs(mask):
        m = _leave_out_margin(x, s, en, noise)
        if not np.isnan(m) and m >= margin:
            out[s : en + 1] = True
    return out


def _leave_out_margin(x: np.ndarray, s: int, en: int, noise: float, reach: int = 4) -> float:
    """How far the frames ``s..en`` of ``x (T, D)`` are from what their neighbours predict.

    A cubic is fitted to the ``reach`` frames on either side. A spike sits far from it compared to
    how well the cubic describes those neighbours; the peak of an overshoot, the apex of a fast
    swing or a frame inside an abrupt move does not. Returns the ratio of the two (``nan`` with
    fewer than two neighbours on a side).
    """
    T = len(x)
    left = np.arange(max(s - reach, 0), s)
    right = np.arange(en + 1, min(en + 1 + reach, T))
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    idx = np.concatenate([left, right])
    mid = 0.5 * (s + en)
    deg = min(3, len(idx) - 2)
    A = np.vander((idx - mid) / reach, deg + 1)
    coef = np.linalg.lstsq(A, x[idx], rcond=None)[0]
    fit = float(np.sqrt(np.mean(np.sum((A @ coef - x[idx]) ** 2, axis=1))))
    run = np.arange(s, en + 1)
    dev = float(np.linalg.norm(np.vander((run - mid) / reach, deg + 1) @ coef - x[run], axis=1).max())
    return dev / max(fit, noise, 1e-300)


def outlier_scores(x: np.ndarray, floor: float, spike_z: float = 6.5, max_spike_len: int = 4,
                   protect: np.ndarray | None = None, dynamics: float = 0.1):
    """Spike mask and noise scores of a vector signal ``(T, D)``.

    The 4th difference ``e`` (what a cubic through the neighbours cannot explain) is measured
    against a per-frame scale: the clip's own typical level, but never less than ``floor`` nor
    less than ``dynamics`` times the local acceleration. Smooth motion has ``e / |acc| = w^2 / 6``
    (w = angular frequency per frame), white noise has ``e / |acc| ~ 0.57``, so fast clean motion
    stays quiet while noise that dominates the local dynamics scores ~5.7.

    * spike: ``e`` sticks out of its neighbourhood, the frame leaves the local median (Hampel),
      the excursion is larger than the net motion across it *and* the neighbours do not predict
      it. A velocity step (foot strike) has no Hampel deviation, a fast move from one pose to the
      next does not come back, the peak of an overshoot follows from the frames around it
    * ``z_sustained``: running median of ``e`` - bursts of jitter, blind to single events

    ``protect`` marks frames owned by other detectors (dropouts). Returns
    ``(spike_mask (T,), z (T,), z_sustained (T,), sigma)``.
    """
    x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    e = fourth_difference(x)
    h = hampel_deviation(x, 3)
    T = len(e)
    acc = np.zeros(T)
    if T >= 3:
        acc[1:-1] = np.linalg.norm(x[2:] - 2.0 * x[1:-1] + x[:-2], axis=1)
    if protect is not None:
        e = np.where(protect, 0.0, e)
        h = np.where(protect, 0.0, h)
        acc = np.where(protect, 0.0, acc)
    inner = e[2:-2] if T > 8 else e
    sigma = max(float(np.median(inner)) if len(inner) else 0.0, floor)
    scale = np.maximum(sigma, dynamics * running_median(acc, 5))
    e_near = np.maximum(e, np.maximum(np.roll(e, 1), np.roll(e, -1)))
    local = np.maximum(scale, running_median(e, 8))
    cand = (e_near >= spike_z * local) & (h >= 0.5 * e_near)
    spike = np.zeros(T, dtype=bool)
    for s, en in find_runs(cand):
        if en - s + 1 > max_spike_len:
            continue
        a, b = max(s - 1, 0), min(en + 1, T - 1)
        if b - a >= 2:
            # How far do the frames leave the straight way from a to b? Being early or late on it
            # is timing (a fast move, a landing), only what leaves the segment is an excursion.
            chord = x[b] - x[a]
            length2 = float(chord @ chord)
            rel = x[a + 1 : b] - x[a]
            along = np.clip(rel @ chord / length2, 0.0, 1.0) if length2 > 0.0 else np.zeros(len(rel))
            excursion = np.linalg.norm(rel - along[:, None] * chord, axis=1).max()
            if excursion < 0.75 * np.sqrt(length2):
                continue
        if not _unpredictable(x, s, en, 0.5 * sigma):
            continue
        spike[s : en + 1] = True
    z = np.maximum(e, np.where(spike, h, 0.0)) / scale
    z_sustained = running_median(np.where(_dilate(spike, 2), 0.0, e), 3) / scale
    return spike, z, z_sustained, sigma


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze(rig, *, bone_names: list[str] | None = None, floor_z: float = 0.0, floor_object=None,
            spike_z: float = 6.5, jitter_z: float = 3.5, hyperext: float = 0.985, pop_threshold: float = 180.0,
            skate_tol: float = 0.012, penetration_tol: float = 0.006, check_world: bool = True,
            max_issues: int = 600, auto_floor: bool = False, pins: list | None = None) -> DiagnosticReport:
    """Scan the active action of ``rig``. Tolerances are metres for a 1.75 m character.

    ``auto_floor``: check feet against the height they rest at in this take (``report.floor_estimate``)
    instead of ``floor_z`` (ignored when a ``floor_object`` is given).
    ``pins``: the pins of a solved take. A foot they hold is checked for sliding inside the locked
    frames only (it is supposed to move while a pin blends in and out).
    """
    f0, f1 = action_io.get_action_frame_range(rig)
    T = f1 - f0 + 1
    fps = scene_fps()
    report = DiagnosticReport(rig.name, f0, f1, fps, checksum=action_io.action_checksum(rig))
    if T < 8:
        return report
    profile = inspect_rig(rig)
    unit, scale = profile.unit, max(profile.scale, 1e-9)
    animated = action_io.get_animated_bone_names(rig)
    if bone_names is not None:
        wanted = set(bone_names)
        animated = [b for b in animated if b in wanted]
    loc_floor = 0.0004 * unit / scale          # channel units
    rot_floor = np.radians(0.15)
    issues: list[Issue] = []
    cutoff_cols = []

    def add_runs(mask, kind, bone, sev, msg_fn, min_len=1, gap=0):
        mask = close_small_gaps(mask, gap) if gap else mask
        for s, e in find_runs(mask):
            if e - s + 1 >= min_len:
                issues.append(Issue(kind, f0 + s, f0 + e, bone, float(np.clip(sev(s, e), 0.05, 1.0)), msg_fn(s, e)))

    # ---- pass 1: read everything, find dropouts (they own their frames) -----------------------------
    chans = {b: action_io.read_bone_local_channels(rig, b, f0, f1) for b in animated}
    body_change = np.zeros(T - 1)
    for ch in chans.values():
        if ch["has_rot"]:
            body_change = np.maximum(body_change, quat_angle_between(ch["quat"][:-1], ch["quat"][1:]) / 1e-5)
        if ch["has_loc"]:
            body_change = np.maximum(body_change, np.abs(np.diff(ch["loc"], axis=0)).max(axis=1) / (1e-5 * unit / scale))
    frozen_all = np.zeros(T, dtype=bool)
    if len(animated) >= 3:
        report.frozen_runs = _frozen_runs(body_change, eps=1.0, active=60.0, min_len=1, max_len=45, jump_ratio=1.5)
        for s, e in report.frozen_runs:
            frozen_all[s : e + 1] = True
            issues.append(Issue("FROZEN", f0 + s, f0 + e, "", 0.8, f"All channels frozen for {e - s + 1} frames (dropout)"))
    protect_all = _dilate(frozen_all, 4)

    # ---- pass 2: per bone detectors --------------------------------------------------------------------
    for bone in animated:
        ch = chans[bone]
        spike_any = np.zeros(T, dtype=bool)
        sure_any = np.zeros(T, dtype=bool)
        loc_rms = rot_rms = 0.0
        protect = protect_all

        # single channel dropouts (sensor froze while the rest of the body kept moving)
        if ch["has_rot"]:
            step = quat_angle_between(ch["quat"][:-1], ch["quat"][1:])
            # a sensor that drops out freezes the whole bone: a location that keeps moving means the rotation is held on purpose
            still = (np.abs(np.diff(ch["loc"], axis=0)).max(axis=1) < 1e-6 * unit / scale) if ch["has_loc"] else None
            runs = [(s, e) for s, e in _frozen_runs(step, eps=1e-6, active=np.radians(0.4), min_len=2, max_len=45,
                                                    jump_ratio=2.0, still=still) if not frozen_all[s : e + 1].all()]
            if runs:
                report.frozen_bones[bone] = runs
                own = np.zeros(T, dtype=bool)
                for s, e in runs:
                    own[s : e + 1] = True
                    issues.append(Issue("FROZEN", f0 + s, f0 + e, bone, 0.6, f"{bone}: rotation frozen for {e - s + 1} frames"))
                protect = protect_all | _dilate(own, 4)

        # ---- flips ----------------------------------------------------------------------------
        if ch["has_rot"]:
            raw = ch["raw_rot"]
            if ch["rotation_mode"] == "QUATERNION":
                flips = np.concatenate([[False], np.sum(raw[:-1] * raw[1:], axis=1) < 0.0])
                what = "quaternion hemisphere flip"
            elif ch["rotation_mode"] == "AXIS_ANGLE":
                flips = np.zeros(T, dtype=bool)
                what = ""
            else:
                flips = np.concatenate([[False], np.abs(np.diff(raw, axis=0)).max(axis=1) > np.radians(150.0)])
                what = "Euler wrap"
            step = np.concatenate([[0.0], quat_angle_between(ch["quat"][:-1], ch["quat"][1:])])
            flips &= step < np.radians(90.0)
            if flips.any():
                report.flip_bones.append(bone)
                add_runs(flips, "FLIP", bone, lambda s, e: 0.6, lambda s, e, w=what: f"{bone}: {w}")
            jumps = (step > np.radians(100.0)) & ~protect
            if jumps.any():
                add_runs(jumps, "FLIP", bone, lambda s, e: 1.0,
                         lambda s, e: f"{bone}: rotation jumps {np.degrees(step[s:e + 1].max()):.0f} deg in one frame")

        # ---- spikes / jitter ----------------------------------------------------------------------
        z_all = np.zeros(T)
        for key, has, floor_v in (("loc", ch["has_loc"], loc_floor), ("rot", ch["has_rot"], rot_floor)):
            if not has:
                continue
            x = ch["loc"] if key == "loc" else rotation_tangent_curve(ch["quat"])
            if key == "loc" and np.ptp(x, axis=0).max() > 1e-4 * unit / scale and len(cutoff_cols) < 60:
                cutoff_cols.append(x * scale)
            spike, z, z_sus, sigma = outlier_scores(x, floor_v, spike_z, protect=protect)
            # (the 4th difference of unit white noise has a median norm of ~2.14 sigma for 3 columns)
            if key == "loc":
                loc_rms = sigma * scale / 2.14 if sigma > floor_v else 0.0
            else:
                rot_rms = sigma / 2.14 if sigma > floor_v else 0.0
            # Confidence follows *sustained* noise: a single velocity step (foot strike) is not noise.
            z_all = np.maximum(z_all, z_sus)
            if spike.any():
                spike_any |= spike
                sure = sure_spikes(x, spike, 0.5 * sigma)
                sure_any |= sure
                part = "loc" if key == "loc" else "quat"
                report.spike_parts.setdefault(bone, {})[part] = spike.copy()
                report.spike_sure.setdefault(bone, {})[part] = sure
                amp = hampel_deviation(x, 3)
                doubt = lambda s, e, ok=sure: "" if ok[s:e + 1].all() else UNSURE_NOTE
                if key == "loc":
                    msg = lambda s, e, a=amp: f"{bone}: location spike of {a[s:e + 1].max() * scale * 100:.1f} cm{doubt(s, e)}"
                else:
                    msg = lambda s, e, a=amp: f"{bone}: rotation spike of {np.degrees(a[s:e + 1].max()):.1f} deg{doubt(s, e)}"
                add_runs(spike, "SPIKE", bone,
                         lambda s, e, zz=z, ok=sure: (0.6 + 0.4 * min(1.0, zz[s:e + 1].max() / (4 * spike_z))) * (1.0 if ok[s:e + 1].all() else 0.6),
                         msg)
            burst = (z_sus >= jitter_z) & ~spike
            if burst.any():
                label = "location" if key == "loc" else "rotation"
                add_runs(burst, "JITTER", bone,
                         lambda s, e, zz=z_sus: 0.3 + 0.4 * min(1.0, zz[s:e + 1].max() / 6.0),
                         lambda s, e, zz=z_sus, lb=label: f"{bone}: {lb} noise {zz[s:e + 1].max():.1f}x above the clip level",
                         min_len=5, gap=2)
        conf = np.exp(-0.35 * np.maximum(z_all - 2.5, 0.0))
        conf[sure_any] = 0.02                  # the filter ignores these frames; a doubtful spike stays data
        report.confidence[bone] = conf
        report.noise[bone] = (loc_rms, rot_rms)
        if spike_any.any():
            report.spike_masks[bone] = spike_any

    # ---- cutoff suggestion ---------------------------------------------------------------------------------
    if cutoff_cols:
        try:
            report.suggested_cutoff, _ = estimate_cutoff(np.concatenate(cutoff_cols, axis=1), fps)
        except Exception:
            report.suggested_cutoff = 0.0

    # ---- world space checks ------------------------------------------------------------------------------------
    if check_world and profile.is_supported:
        issues += _world_checks(rig, profile, report, f0, f1, fps, floor_z, floor_object, hyperext, pop_threshold,
                                skate_tol, penetration_tol, auto_floor, pins)

    issues.sort(key=lambda i: (-i.severity * (1.0 + 0.02 * min(i.frame_end - i.frame_start, 50))))
    issues = issues[:max_issues]
    issues.sort(key=lambda i: (i.frame_start, -i.severity))
    report.issues = issues
    return report


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if not mask.any():
        return mask
    padded = np.pad(mask, radius, mode="constant")
    return sliding_window_view(padded, 2 * radius + 1).any(axis=-1)


def _frozen_runs(change: np.ndarray, eps: float, active: float, min_len: int, max_len: int,
                 jump_ratio: float = 1.5, still: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Runs of (near) zero frame-to-frame change that are dropouts. Returns frame-index runs to rebuild.

    A dropout sits inside moving data and ends with a snap: one step that makes up for the frozen
    frames, after which the motion goes on at its own speed. A deliberate hold does not. A planted
    foot or a pose that eases in and out has no snap; a pose that is held and then left quickly
    (pose-to-pose animation, a punch) keeps moving fast after its first step. And a dropout
    interrupts the motion, while a hold is something the motion arrives at: it has slowed down
    to a fraction of its speed before it stops.
    ``still`` (per step): the other channels of the same sensor did not change either.
    """
    n = len(change)
    out = []
    for s, e in find_runs(change < eps):
        length = e - s + 1
        if length < min_len or length > max_len:
            continue
        if still is not None and not still[s : e + 1].all():
            continue
        before = change[max(0, s - 6) : s]
        after = change[e + 2 : e + 8]
        if len(before) == 0 or len(after) == 0 or e + 1 >= n:
            continue
        alone = change[e + 1] > jump_ratio * change[e + 2 : e + 4].max()
        arrived = change[s - 1] < 0.25 * before.max()
        if (alone and not arrived and np.median(before) > active and np.median(after) > active
                and change[e + 1] > jump_ratio * np.median(after)):
            # change[t] = x[t + 1] - x[t]: frames s + 1 .. e + 1 repeat frame s
            out.append((s + 1, e + 1))
    return out


def body_joints(profile, smp) -> dict[str, np.ndarray]:
    """World positions ``(T, 3)`` of the joints that show what a pose looks like."""
    out = {}
    for label, bone in (("Hips", profile.torso_ctrl), ("Head", profile.head_bone)):
        if bone and smp.has(bone):
            out[label] = smp.head_w(bone)
    for limb in profile.limbs():
        mid, end = ("Knee", "Foot") if limb.kind == "leg" else ("Elbow", "Hand")
        if smp.has(limb.org_lower):
            out[f"{mid} {limb.side}"] = smp.head_w(limb.org_lower)
        if smp.has(limb.org_end):
            out[f"{end} {limb.side}"] = smp.head_w(limb.org_end)
    return out


def largest_change(before: dict[str, np.ndarray], after: dict[str, np.ndarray], f0: int = 0):
    """``(joint, distance, frame)`` of the largest move between two ``body_joints`` results (``None``: nothing to compare)."""
    best = None
    for name, a in before.items():
        b = after.get(name)
        if b is None or b.shape != a.shape:
            continue
        d = np.linalg.norm(b - a, axis=1)
        i = int(np.argmax(d))
        if best is None or d[i] > best[1]:
            best = (name, float(d[i]), f0 + i)
    return best


def _world_checks(rig, profile, report, f0, f1, fps, floor_z, floor_object, hyperext, pop_threshold, skate_tol,
                  penetration_tol, auto_floor=False, solved_pins=None) -> list[Issue]:
    issues: list[Issue] = []
    unit = profile.unit
    T = f1 - f0 + 1
    smp = sampler.sample_objects([(rig, solver_bone_names(profile, rig.data.bones))], f0, f1)[rig.name]
    report.joints = body_joints(profile, smp)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    floor = s4.Floor(floor_z, floor_object, depsgraph)

    for limb in profile.limbs():
        root = smp.head_w(limb.org_upper)
        mid = smp.head_w(limb.org_lower)
        end = smp.head_w(limb.org_end)
        ikfk = (action_io.read_bone_prop_curve(rig, limb.switch_bone, "IK_FK", f0, f1) if limb.switch_bone
                else np.zeros(T))
        aim = commanded_end(limb, smp)                     # with foot roll the control itself may be far away
        target = np.where((ikfk < 0.5)[:, None], aim if aim is not None else smp.head_w(limb.ik_ctrl), end)
        reach = np.linalg.norm(target - root, axis=1) / max(limb.max_reach, 1e-9)
        name = ("Left " if limb.side == "L" else "Right ") + limb.kind
        over = (reach > hyperext) & (ikfk < 0.5)           # a straight FK limb is a pose, not an IK that runs out of reach
        for s, e in find_runs(over):
            r = float(reach[s : e + 1].max())
            issues.append(Issue("HYPEREXT", f0 + s, f0 + e, limb.ik_ctrl, float(np.clip(0.35 + 8.0 * (r - hyperext), 0.35, 0.9)),
                                f"{name} over-extended (reach {r:.3f})"))
        u, l = mid - root, end - mid
        cosang = np.sum(u * l, axis=1) / np.maximum(np.linalg.norm(u, axis=1) * np.linalg.norm(l, axis=1), 1e-12)
        flex = np.arccos(np.clip(cosang, -1.0, 1.0))
        acc = np.zeros(T)
        acc[1:-1] = np.abs(flex[2:] - 2.0 * flex[1:-1] + flex[:-2]) * fps * fps
        near_straight = np.minimum(flex, np.minimum(np.roll(flex, 1), np.roll(flex, -1))) < np.radians(15.0)
        pop = (acc > pop_threshold) & near_straight
        joint = "knee" if limb.kind == "leg" else "elbow"
        for s, e in find_runs(close_small_gaps(pop, 1)):
            a = float(acc[s : e + 1].max())
            issues.append(Issue("KNEE_POP", f0 + s, f0 + e, limb.ik_ctrl, float(np.clip(0.6 + a / (10.0 * pop_threshold), 0.6, 1.0)),
                                f"{name}: {joint} pop ({a:.0f} rad/s^2 at full extension)"))

    soles = {}
    for S, leg in profile.legs.items():
        if not len(leg.sole_points):
            continue
        soles[leg.key] = sole_contacts(leg, smp.world_m(leg.org_end))
    if soles:
        report.floor_estimate = s4.estimate_floor_height(soles, fps, unit)
        if auto_floor and floor_object is None:
            floor = s4.Floor(report.floor_estimate, None, depsgraph)
        pins, _ = s4.detect_contacts(soles, floor, f0, fps, unit)
        for key, pts in soles.items():
            leg = profile.limb_by_key(key)
            name = "Left foot" if leg.side == "L" else "Right foot"
            depth = (floor.height(pts) - pts[..., 2]).max(axis=1)
            for s, e in find_runs(close_small_gaps(depth > penetration_tol * unit, 2)):
                d = float(depth[s : e + 1].max())
                issues.append(Issue("PENETRATION", f0 + s, f0 + e, leg.ik_ctrl, float(np.clip(0.4 + d / (0.1 * unit), 0.4, 1.0)),
                                    f"{name} {d / unit * 100:.1f} cm below the floor"))
            h = pts[..., 2] - floor.height(pts)
            w = np.exp(-(h - h.min(axis=1, keepdims=True)) / (0.0015 * unit))
            w /= w.sum(axis=1, keepdims=True)
            vel = np.gradient(pts, axis=0)
            pivot_v = np.sum(w[..., None] * vel, axis=1)
            pivot_v[:, 2] = 0.0
            for pin in pins:
                if pin.owner != key:
                    continue
                s, e = pin.lock_in - f0, pin.lock_out - f0
                for held in (solved_pins or ()):
                    if (held.owner == key and held.enabled and held.mode in ("FLOOR", "WORLD")
                            and held.lock_in - f0 <= e and held.lock_out - f0 >= s):
                        s, e = max(s, held.lock_in - f0), min(e, held.lock_out - f0)
                if e - s < 3:
                    continue
                path = np.cumsum(pivot_v[s + 1 : e], axis=0)
                slide = float(np.linalg.norm(path, axis=1).max()) if len(path) else 0.0
                if slide > skate_tol * unit:
                    issues.append(Issue("SKATE", f0 + s, f0 + e, leg.ik_ctrl, float(np.clip(0.4 + slide / (0.1 * unit), 0.4, 1.0)),
                                        f"{name} slides {slide / unit * 100:.1f} cm while planted"))
    return issues


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

def bridge_vectors(x: np.ndarray, mask: np.ndarray, valid: np.ndarray | None = None, context: int = 16) -> np.ndarray:
    """Fill the masked frames of ``x (T, D)`` with minimum-jerk bridges.

    Each gap is solved as a Whittaker problem of order 3 with zero weight inside the gap: the
    result is the C2 quintic spline that continues the surrounding motion, with edge velocity and
    acceleration taken from *all* nearby samples instead of a few finite differences. Longer gaps
    use a stiffer spline, so noise at the edges cannot swing the bridge around.
    ``valid`` (default: everything outside ``mask``) marks the frames that may be trusted.
    """
    x = np.array(x, dtype=np.float64, copy=True)
    squeeze = x.ndim == 1
    if squeeze:
        x = x[:, None]
    valid = ~mask if valid is None else valid
    T = len(x)
    for s, e in find_runs(mask):
        a, b = s - 1, e + 1
        n = e - s + 1
        if a < 0 and b > T - 1:
            continue
        if a < 0:
            x[s : e + 1] = x[b]
            continue
        if b > T - 1:
            x[s : e + 1] = x[a]
            continue
        u = (np.arange(1, n + 1) / (n + 1.0))[:, None]
        i0, i1 = max(0, a - context), min(T - 1, b + context)
        w = valid[i0 : i1 + 1].astype(np.float64)
        w[s - i0 : e - i0 + 1] = 0.0
        if w[: s - i0].sum() < 3 or w[e + 1 - i0 :].sum() < 3:
            x[s : e + 1] = (1.0 - u) * x[a] + u * x[b]
            continue
        lam = whittaker_lambda_for_cutoff(float(np.clip(1.8 / n, 0.05, 0.3)), 1.0, 3)
        z = whittaker_smooth(x[i0 : i1 + 1], lam, 3, weights=w)
        # the spline is smooth, the samples next to the gap are not: meet them exactly
        fix = (1.0 - u) * (x[a] - z[a - i0]) + u * (x[b] - z[b - i0])
        x[s : e + 1] = z[s - i0 : e - i0 + 1] + fix
    return x[:, 0] if squeeze else x


def bridge_quaternions(q: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill the masked frames of ``q (T, 4)``.

    The gap is bridged in the tangent space at its first valid neighbour with the same C2 Hermite
    bridge as translations (velocity and acceleration fitted over several frames on both sides).
    Gaps that span more than ~115 degrees fall back to SQUAD.
    """
    q = quat_hemisphere_continuous(q)[0].copy()
    valid = ~mask
    T = len(q)
    for s, e in find_runs(mask):
        a, b = s - 1, e + 1
        n = e - s + 1
        if a < 0 and b > T - 1:
            continue
        if a < 0:
            q[s : e + 1] = q[b]
            continue
        if b > T - 1:
            q[s : e + 1] = q[a]
            continue
        if float(quat_angle_between(q[a], q[b])) > 2.0:
            qa = q[a - 1] if (a - 1 >= 0 and valid[a - 1]) else q[a]
            qb = q[b + 1] if (b + 1 < T and valid[b + 1]) else q[b]
            q[s : e + 1] = quat_squad_bridge(qa, q[a], q[b], qb, n)
            continue
        i0, i1 = max(0, a - 5), min(T - 1, b + 5)
        local = q[i0 : i1 + 1]
        local = np.where(np.sum(local * q[a], axis=1, keepdims=True) < 0.0, -local, local)
        v = quat_log(quat_multiply(quat_conjugate(q[a])[None, :], local))
        gap = np.zeros(i1 - i0 + 1, dtype=bool)
        gap[s - i0 : e - i0 + 1] = True
        v = bridge_vectors(v, gap, valid=valid[i0 : i1 + 1])
        q[s : e + 1] = quat_multiply(q[a][None, :], quat_exp(v[s - i0 : e - i0 + 1]))
    return quat_hemisphere_continuous(q)[0]


def lowrank_complete(X: np.ndarray, missing: np.ndarray, energy: float = 0.97, max_rank: int = 12,
                     iterations: int = 30) -> np.ndarray:
    """Iterative PCA imputation of the ``missing`` entries of ``X (T, D)`` (initial guess already in place)."""
    obs = ~missing
    cnt = np.maximum(obs.sum(axis=0), 1)
    mu = (X * obs).sum(axis=0) / cnt
    sd = np.sqrt(((X - mu) ** 2 * obs).sum(axis=0) / cnt)
    sd = np.where(sd < 1e-9, 1.0, sd)
    Z = (X - mu) / sd
    Zr = Z
    for _ in range(iterations):
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        cum = np.cumsum(S ** 2) / max(float(np.sum(S ** 2)), 1e-30)
        r = int(min(max_rank, np.searchsorted(cum, energy) + 1, len(S)))
        Zr = (U[:, :r] * S[:r]) @ Vt[:r]
        change = float(np.abs(Zr[missing] - Z[missing]).max()) if missing.any() else 0.0
        Z = np.where(missing, Zr, Z)
        if change < 1e-5:
            break
    return Zr * sd + mu


def _channel_masks(mask) -> dict[str, np.ndarray]:
    """``mask``: one ``(T,)`` array for the whole bone, or ``{'loc': ..., 'quat': ...}``."""
    return mask if isinstance(mask, dict) else {"loc": mask, "quat": mask}


def tame_bridges(x: np.ndarray, mask: np.ndarray, max_detour: float = 2.0, quaternion: bool = False) -> np.ndarray:
    """Replace rebuilt gaps that wander by the straight way between their ends.

    For gaps that were found as dropouts: a frozen run only counts as one when it ends with a jump,
    so what was missed has a clear direction. A bridge that travels more than ``max_detour`` times
    that way was steered by what happens around the gap (a fast move that stopped, another one
    that starts), not by the gap.
    """
    x = np.array(x, dtype=np.float64, copy=True)
    T = len(x)
    for s, e in find_runs(mask):
        a, b = s - 1, e + 1
        if a < 0 or b > T - 1:
            continue
        seg = x[a : b + 1]
        if quaternion:
            way = float(quat_angle_between(seg[:-1], seg[1:]).sum())
            direct = float(quat_angle_between(seg[0], seg[-1]))
        else:
            way = float(np.linalg.norm(np.diff(seg, axis=0), axis=1).sum())
            direct = float(np.linalg.norm(seg[-1] - seg[0]))
        if way <= max_detour * direct + 1e-12:
            continue
        n = e - s + 1
        u = np.arange(1, n + 1) / (n + 1.0)
        if quaternion:
            x[s : e + 1] = quat_slerp(np.broadcast_to(x[a], (n, 4)), np.broadcast_to(x[b], (n, 4)), u)
        else:
            x[s : e + 1] = x[a] + u[:, None] * (x[b] - x[a])
    return x


def reconstruct_channels(chans: dict[str, dict], masks: dict, long_gap: int = 7, window: int = 300,
                         guard: dict[str, np.ndarray] | None = None) -> dict[str, dict]:
    """Rebuild masked frames. ``chans``: bone -> ``read_bone_local_channels`` dict; ``masks``: bone ->
    ``(T,)`` bool for the whole bone, or ``{'loc': (T,), 'quat': (T,)}``. A channel without masked
    frames is not touched (``None`` in the result).

    Returns ``bone -> {'loc': (T, 3) | None, 'quat': (T, 4) | None}`` for the bones in ``masks``.
    Every gap gets a Hermite / SQUAD bridge; gaps of ``long_gap`` frames or more are refined with
    low-rank inter-joint completion when at least a few other channels are intact over the gap.
    ``guard``: bone -> frames whose bridges must not wander (see ``tame_bridges``).
    """
    masks = {bone: _channel_masks(m) for bone, m in masks.items()}
    out: dict[str, dict] = {}
    for bone, m in masks.items():
        ch = chans[bone]
        out[bone] = {
            "loc": bridge_vectors(ch["loc"], m["loc"]) if ch["has_loc"] and m["loc"].any() else None,
            "quat": bridge_quaternions(ch["quat"], m["quat"]) if ch["has_rot"] and m["quat"].any() else None,
        }
    if not masks:
        return out

    def finish():
        for bone, g in (guard or {}).items():
            for key in ("loc", "quat"):
                if bone in out and out[bone][key] is not None and (g & masks[bone][key]).any():
                    out[bone][key] = tame_bridges(out[bone][key], g & masks[bone][key], quaternion=key == "quat")
        return out

    T = len(next(iter(masks.values()))["loc"])
    any_mask = np.zeros(T, dtype=bool)
    for m in masks.values():
        any_mask |= m["loc"] | m["quat"]
    long_runs = [(s, e) for s, e in find_runs(any_mask) if e - s + 1 >= long_gap and s > 0 and e < T - 1]
    if not long_runs:
        return finish()

    # Feature matrix over all animated bones: locations + rotation vectors about the mean rotation
    cols, col_mask, owners = [], [], []
    refs = {}
    nothing = {"loc": np.zeros(T, dtype=bool), "quat": np.zeros(T, dtype=bool)}
    for bone, ch in chans.items():
        cur = out.get(bone, {})
        if ch["has_loc"]:
            m = masks.get(bone, nothing)["loc"]
            x = cur.get("loc") if cur.get("loc") is not None else ch["loc"]
            if np.ptp(x, axis=0).max() > 1e-7:
                cols.append(x)
                col_mask.append(np.repeat(m[:, None], 3, axis=1))
                owners.append((bone, "loc"))
        if ch["has_rot"]:
            m = masks.get(bone, nothing)["quat"]
            q = cur.get("quat") if cur.get("quat") is not None else quat_hemisphere_continuous(ch["quat"])[0]
            ref = quat_normalize(q[~m].mean(axis=0)) if (~m).any() else q[0]
            v = quat_log(quat_multiply(quat_conjugate(ref)[None, :], q))
            if np.ptp(v, axis=0).max() > 1e-7:
                refs[bone] = ref
                cols.append(v)
                col_mask.append(np.repeat(m[:, None], 3, axis=1))
                owners.append((bone, "quat"))
    if len(cols) < 4:
        return finish()
    X = np.concatenate(cols, axis=1)
    M = np.concatenate(col_mask, axis=1)

    for s, e in long_runs:
        w0, w1 = max(0, s - window), min(T, e + 1 + window)
        Xw, Mw = X[w0:w1], M[w0:w1]
        gs, ge = s - w0, e - w0
        intact = ~Mw[gs : ge + 1].any(axis=0)
        if intact.sum() < 9 or (w1 - w0) - (e - s + 1) < 40:
            continue                     # nothing to learn from: keep the bridge
        R = lowrank_complete(Xw, Mw)
        n = ge - gs + 1
        u = (np.arange(1, n + 1) / (n + 1.0))[:, None]
        # remove the offset between model and data at both edges, then fade into the bridge
        fix = (1.0 - u) * (Xw[gs - 1] - R[gs - 1]) + u * (Xw[ge + 1] - R[ge + 1])
        k = max(2, min(4, n // 3))
        idx = np.arange(n)
        fade = quintic_blend_weight(np.minimum(idx + 1, n - idx) / float(k))[:, None]
        model = R[gs : ge + 1] + fix
        blended = fade * model + (1.0 - fade) * Xw[gs : ge + 1]
        gap_cols = Mw[gs : ge + 1].all(axis=0)
        X[s : e + 1][:, gap_cols] = blended[:, gap_cols]

    c = 0
    for bone, kind in owners:
        if bone in out and masks[bone][kind].any():
            if kind == "loc":
                out[bone]["loc"] = X[:, c : c + 3].copy()
            else:
                out[bone]["quat"] = quat_hemisphere_continuous(quat_multiply(refs[bone][None, :], quat_exp(X[:, c : c + 3])))[0]
        c += 3
    return finish()


# ---------------------------------------------------------------------------
# Repair operations on the action
# ---------------------------------------------------------------------------

def fix_flips(rig, bone_names: list[str] | None = None) -> int:
    """Rewrite rotation channels continuously (quaternion hemisphere, Euler unwrap). Returns the bone count."""
    f0, f1 = action_io.get_action_frame_range(rig)
    animated = action_io.get_animated_bone_names(rig)
    if bone_names is not None:
        animated = [b for b in animated if b in set(bone_names)]
    fixed = 0
    for bone in animated:
        ch = action_io.read_bone_local_channels(rig, bone, f0, f1)
        if not ch["has_rot"] or ch["rotation_mode"] == "AXIS_ANGLE":
            continue
        raw = ch["raw_rot"]
        if ch["rotation_mode"] == "QUATERNION":
            bad = bool(np.any(np.sum(raw[:-1] * raw[1:], axis=1) < 0.0))
        else:
            bad = bool(np.any(np.abs(np.diff(raw, axis=0)) > np.radians(150.0)))
        if bad:
            action_io.write_bone_local_channels(rig, bone, f0, quat=quat_hemisphere_continuous(ch["quat"])[0])
            fixed += 1
    return fixed


def fill_gaps(rig, masks: dict, guard: dict[str, np.ndarray] | None = None) -> int:
    """Reconstruct the masked frames of the given bones (see ``reconstruct_channels`` for ``masks``
    and ``guard``). Channels without masked frames keep their curves. Returns the number of rebuilt
    bone-frames."""
    masks = {b: _channel_masks(m) for b, m in masks.items()}
    masks = {b: m for b, m in masks.items() if m["loc"].any() or m["quat"].any()}
    if not masks:
        return 0
    f0, f1 = action_io.get_action_frame_range(rig)
    T = f1 - f0 + 1
    has_long = any(e - s + 1 >= 7 for m in masks.values() for part in m.values() for s, e in find_runs(part))
    names = action_io.get_animated_bone_names(rig) if has_long else list(masks)
    chans = {b: action_io.read_bone_local_channels(rig, b, f0, f1) for b in names if b in rig.pose.bones}
    masks = {b: m for b, m in masks.items() if b in chans and len(m["loc"]) == T}
    result = reconstruct_channels(chans, masks, guard=guard)
    count = 0
    for bone, res in result.items():
        if res["loc"] is None and res["quat"] is None:
            continue
        action_io.write_bone_local_channels(rig, bone, f0, loc=res["loc"], quat=res["quat"])
        count += int((masks[bone]["loc"] | masks[bone]["quat"]).sum())
    return count


def fill_range(rig, frame_start: int, frame_end: int, bone_names: list[str] | None = None) -> int:
    """Cut out ``[frame_start, frame_end]`` on the given (default: all animated) bones and rebuild it."""
    f0, f1 = action_io.get_action_frame_range(rig)
    T = f1 - f0 + 1
    s, e = max(frame_start, f0) - f0, min(frame_end, f1) - f0
    if e < s:
        return 0
    mask = np.zeros(T, dtype=bool)
    mask[s : e + 1] = True
    names = bone_names if bone_names is not None else action_io.get_animated_bone_names(rig)
    return fill_gaps(rig, {b: mask.copy() for b in names})


def repair(rig, report: DiagnosticReport, *, flips: bool = True, spikes: bool = True, frozen: bool = True,
           frame_start: int | None = None, frame_end: int | None = None, bone_names: list[str] | None = None,
           unsure: bool = True) -> dict:
    """Apply the structural repairs suggested by ``report`` (optionally limited to a range / bones).

    ``unsure=False``: rebuild only the spikes that are beyond doubt (``sure_spikes``) - for repairs
    nobody looks at one by one.
    """
    f0, f1 = action_io.get_action_frame_range(rig)
    T = f1 - f0 + 1
    result = {"flips": 0, "spike_frames": 0, "frozen_frames": 0}
    if report.checksum and report.checksum != action_io.action_checksum(rig):
        raise RuntimeError("The animation changed since the last analysis - analyze again")
    wanted = set(bone_names) if bone_names is not None else None
    window = np.ones(T, dtype=bool)
    if frame_start is not None or frame_end is not None:
        window[:] = False
        a = max((frame_start if frame_start is not None else f0) - f0, 0)
        b = min((frame_end if frame_end is not None else f1) - f0, T - 1)
        window[a : b + 1] = True

    if flips:
        names = [b for b in report.flip_bones if wanted is None or b in wanted]
        if names:
            result["flips"] = fix_flips(rig, names)

    masks: dict[str, dict[str, np.ndarray]] = {}
    guard: dict[str, np.ndarray] = {}

    def part(bone):
        return masks.setdefault(bone, {"loc": np.zeros(T, dtype=bool), "quat": np.zeros(T, dtype=bool)})

    spike_total = frozen_total = 0
    if spikes:
        for bone, m in report.spike_masks.items():
            if (wanted is None or bone in wanted) and len(m) == T:
                mm = m & window
                if mm.any():
                    # only the channel that spiked is rebuilt: a location spike leaves the rotation alone
                    done = np.zeros(T, dtype=bool)
                    for key, pm in (report.spike_parts.get(bone) or {"loc": m, "quat": m}).items():
                        if not unsure:
                            pm = pm & report.spike_sure.get(bone, {}).get(key, np.zeros(T, dtype=bool))
                        part(bone)[key] |= pm & window
                        done |= pm & window
                    spike_total += int(done.sum())
    if frozen:
        animated = action_io.get_animated_bone_names(rig)
        for s, e in report.frozen_runs:
            if not window[s : e + 1].any():
                continue
            for bone in animated:
                if wanted is None or bone in wanted:
                    for key in ("loc", "quat"):
                        part(bone)[key][s : min(e, T - 1) + 1] = True
            frozen_total += e - s + 1
        for bone, runs in report.frozen_bones.items():
            if wanted is not None and bone not in wanted:
                continue
            for s, e in runs:
                if window[s : e + 1].any():
                    for key in ("loc", "quat"):
                        part(bone)[key][s : min(e, T - 1) + 1] = True
                    guard.setdefault(bone, np.zeros(T, dtype=bool))[s : min(e, T - 1) + 1] = True
                    frozen_total += e - s + 1
    if masks:
        fill_gaps(rig, masks, guard=guard)
    result["spike_frames"] = spike_total
    result["frozen_frames"] = frozen_total
    return result
