"""Rig inspection: a Rigify / Rig-Bridge control rig, or a plain humanoid skeleton.

Builds a :class:`RigProfile` that names the controls and mechanism bones the cleanup stages need
(IK controls, pole targets, FK/IK switch bones, ``ORG-`` joints, sole geometry, mass segments).

A plain skeleton (a Mixamo, Motive / OptiTrack, Unreal or Character Creator take as it is imported, before any
control rig exists) has no controls: its limb joints stand in for them (``kind == "SKELETON"``), and the solver
writes the result back as rotations of those joints (see ``stages.solver``). Rest-pose geometry is measured
upright, in the object's orientation: FBX imports are usually turned 90 degrees and scaled to 0.01.
Nothing is generated or modified here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from mathutils import Vector

# Segment mass fractions and centre-of-mass positions along the segment (proximal -> distal),
# after Winter, "Biomechanics and Motor Control of Human Movement" (4th ed., table 4.1).
MASS_TRUNK = 0.497
MASS_HEAD = 0.081
MASS_SEGMENTS = {
    "thigh": (0.100, 0.433),
    "shin": (0.0465, 0.433),
    "foot": (0.0145, 0.50),
    "upper_arm": (0.028, 0.436),
    "forearm": (0.016, 0.430),
    "hand": (0.006, 0.506),
}

_SPINE_RE = re.compile(r"^ORG-spine(?:\.(\d+))?$")


@dataclass
class LimbSpec:
    kind: str                     # 'leg' | 'arm'
    side: str                     # 'L' | 'R'
    ik_ctrl: str
    pole_ctrl: str
    ik_base_ctrl: str
    switch_bone: str
    org_upper: str
    org_lower: str
    org_end: str
    org_toe: str = ""
    toe_ctrl: str = ""
    heel_ctrl: str = ""
    ik_target: str = ""           # bone the IK chain aims at (Rigify: MCH-thigh_ik_target.L)
    # Where the rig asks the limb end to be: the point ``ik_aim_local`` of the bone ``ik_aim_bone``
    ik_aim_bone: str = ""
    ik_aim_local: np.ndarray = field(default_factory=lambda: np.zeros(3))
    upper_len: float = 0.4        # world units
    lower_len: float = 0.4
    # ``W_end = W_ik @ x_rest`` in the rest pose (4x4, armature space independent)
    x_rest: np.ndarray = field(default_factory=lambda: np.eye(4))
    # Sole contact points in the end (foot) bone's local space: heel_in, heel_out, ball_in, ball_out
    sole_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    # A sole point with a radius is a roller: its centre sits ``radius`` above the sole and it
    # touches the floor straight below the centre, however the foot is tilted. The ball of the foot
    # rolls like that around the toe joint (which is where the rig's foot roll pivots as well).
    sole_radius: np.ndarray = field(default_factory=lambda: np.zeros(0))
    up_local: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))   # world up in the rest pose
    # Toe tip in the toe bone's local space and the toe hinge (ball centre + lateral axis) in foot space
    toe_tip_local: np.ndarray = field(default_factory=lambda: np.zeros(3))
    toe_in_foot_rest: np.ndarray = field(default_factory=lambda: np.eye(4))
    ball_local: np.ndarray = field(default_factory=lambda: np.zeros(3))
    heel_local: np.ndarray = field(default_factory=lambda: np.zeros(3))
    lateral_local: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    foot_length: float = 0.25     # heel -> toe tip, world units
    end_radius: float = 0.04      # wrist / ankle clearance radius, world units

    @property
    def max_reach(self) -> float:
        return self.upper_len + self.lower_len

    @property
    def is_skeleton(self) -> bool:
        return self.ik_ctrl == self.org_end

    @property
    def label(self) -> str:
        return ("Foot" if self.kind == "leg" else "Hand") + "." + self.side

    @property
    def key(self) -> str:
        return self.kind + "." + self.side


@dataclass
class RigProfile:
    rig_name: str
    is_supported: bool
    kind: str = ""                # 'RIGIFY' | 'SKELETON' | '' (nothing the IK-based tools can work on)
    root_ctrl: str = ""
    torso_ctrl: str = ""
    head_bone: str = ""
    spine_bones: list[str] = field(default_factory=list)   # ORG chain pelvis -> chest (no neck / head)
    neck_bones: list[str] = field(default_factory=list)
    legs: dict[str, LimbSpec] = field(default_factory=dict)
    arms: dict[str, LimbSpec] = field(default_factory=dict)
    # (bone, mass fraction, com position along bone 0..1)
    com_segments: list[tuple[str, float, float]] = field(default_factory=list)
    rest_floor_z: float = 0.0     # armature-space Z of the sole plane in the rest pose
    character_height: float = 1.75
    unit: float = 1.0             # world units per "human metre" (height / 1.75)
    scale: float = 1.0            # armature object scale (assumed uniform)
    message: str = ""

    def limbs(self) -> list[LimbSpec]:
        return list(self.legs.values()) + list(self.arms.values())

    def limb_for_ctrl(self, bone_name: str) -> LimbSpec | None:
        for limb in self.limbs():
            if bone_name in (limb.ik_ctrl, limb.org_end):
                return limb
        return None

    def limb_by_key(self, key: str) -> LimbSpec | None:
        for limb in self.limbs():
            if limb.key == key:
                return limb
        return None


def _first(bones, *names) -> str:
    for n in names:
        if n and n in bones:
            return n
    return ""


def _np4(m) -> np.ndarray:
    return np.array(m, dtype=np.float64)


def _spine_chain(bones) -> list[str]:
    found = []
    for b in bones:
        m = _SPINE_RE.match(b.name)
        if m:
            found.append((int(m.group(1) or 0), b.name))
    return [n for _, n in sorted(found)]


def _upright(rig) -> np.ndarray:
    """Rotation ``(3, 3)`` from armature space to an upright frame (the object's rotation without its scale)."""
    m = np.array(rig.matrix_world.to_3x3(), dtype=np.float64)
    n = np.linalg.norm(m, axis=0)
    return m / np.where(n < 1e-12, 1.0, n)


def inspect_rig(rig) -> RigProfile:
    """Inspect an armature object. ``profile.is_supported`` tells whether IK-based tools can run."""
    if rig is None or rig.type != "ARMATURE":
        raise ValueError("Expected an Armature object")
    bones = rig.data.bones
    scale = float(np.mean([abs(s) for s in rig.matrix_world.to_scale()]))
    profile = RigProfile(rig_name=rig.name, is_supported=False, scale=scale)
    A = _upright(rig)

    profile.root_ctrl = _first(bones, "root")
    profile.torso_ctrl = _first(bones, "torso")
    if not (profile.torso_ctrl and "foot_ik.L" in bones and "foot_ik.R" in bones):
        skel = _inspect_skeleton(rig, profile)
        if skel is not None:
            return skel

    # ---- rest floor level (armature space) -------------------------------------------------
    floor_candidates = []
    for S in ("L", "R"):
        foot = bones.get(f"ORG-foot.{S}")
        if foot is None:
            continue
        for child in foot.children:
            if child.name.startswith("ORG-") and "heel" in child.name.lower() and not child.children:
                floor_candidates += [_uz(A, child.head_local), _uz(A, child.tail_local)]
    if floor_candidates:
        rest_floor = float(min(floor_candidates))
    else:
        zs = [_uz(A, b.head_local) for b in bones if b.name.startswith(("ORG-", "DEF-"))] or \
            [_uz(A, b.head_local) for b in bones]
        zs += [_uz(A, b.tail_local) for b in bones if b.name.startswith(("ORG-toe", "DEF-toe"))]
        rest_floor = float(min(zs)) if zs else 0.0
    profile.rest_floor_z = rest_floor

    top = [max(_uz(A, b.head_local), _uz(A, b.tail_local)) for b in bones if b.name.startswith(("ORG-", "DEF-"))]
    if not top:
        top = [max(_uz(A, b.head_local), _uz(A, b.tail_local)) for b in bones]
    height_arm = max((max(top) - rest_floor) if top else 1.75, 1e-3)
    profile.character_height = height_arm * scale
    profile.unit = profile.character_height / 1.75

    # ---- limbs -------------------------------------------------------------------------------
    for S in ("L", "R"):
        leg = _build_limb(bones, "leg", S, scale, rest_floor, A)
        if leg is not None:
            profile.legs[S] = leg
        arm = _build_limb(bones, "arm", S, scale, rest_floor, A)
        if arm is not None:
            profile.arms[S] = arm
    for limb in profile.limbs():
        limb.ik_target = _ik_target_bone(rig, limb)
        limb.ik_aim_bone, limb.ik_aim_local = _ik_aim(rig, limb.ik_target)

    # ---- spine / head --------------------------------------------------------------------------
    chain = _spine_chain(bones)
    head_ctrl = bones.get("head")
    neck_ctrl = bones.get("neck")
    if chain:
        neck_z = None
        if neck_ctrl is not None:
            neck_z = _uz(A, neck_ctrl.head_local)
        elif head_ctrl is not None:
            neck_z = _uz(A, head_ctrl.head_local)
        if neck_z is not None:
            torso_part = [n for n in chain if _uz(A, bones[n].head_local) < neck_z - 1e-4]
            rest = [n for n in chain if n not in torso_part]
        else:
            torso_part, rest = (chain[:-1], chain[-1:]) if len(chain) > 1 else (chain, [])
        if not torso_part:
            torso_part, rest = chain[:1], chain[1:]
        profile.spine_bones = torso_part
        if rest:
            profile.head_bone = rest[-1]
            profile.neck_bones = rest[:-1]
    if not profile.head_bone:
        profile.head_bone = _first(bones, "ORG-head", "DEF-head", "head")

    _com_segments(profile, bones)

    profile.is_supported = bool(profile.torso_ctrl and len(profile.legs) == 2)
    if profile.is_supported:
        profile.kind = "RIGIFY"
    else:
        profile.message = ("No Rigify control rig and no humanoid skeleton found (a skeleton needs two legs: "
                           "thigh, shin and foot bones hanging from the hips).")
    return profile


def _com_segments(profile: RigProfile, bones) -> None:
    segs: list[tuple[str, float, float]] = []
    if profile.spine_bones:
        lengths = np.array([bones[n].length for n in profile.spine_bones], dtype=np.float64)
        lengths = lengths / max(lengths.sum(), 1e-9)
        for n, frac in zip(profile.spine_bones, lengths):
            segs.append((n, MASS_TRUNK * float(frac), 0.5))
    elif profile.torso_ctrl:
        segs.append((profile.torso_ctrl, MASS_TRUNK, 0.5))
    if profile.head_bone:
        segs.append((profile.head_bone, MASS_HEAD, 0.5))
    for limb in profile.legs.values():
        segs.append((limb.org_upper, *MASS_SEGMENTS["thigh"]))
        segs.append((limb.org_lower, *MASS_SEGMENTS["shin"]))
        segs.append((limb.org_end, *MASS_SEGMENTS["foot"]))
    for limb in profile.arms.values():
        segs.append((limb.org_upper, *MASS_SEGMENTS["upper_arm"]))
        segs.append((limb.org_lower, *MASS_SEGMENTS["forearm"]))
        segs.append((limb.org_end, *MASS_SEGMENTS["hand"]))
    total = sum(m for _, m, _ in segs)
    profile.com_segments = [(b, m / total, c) for b, m, c in segs] if total > 0 else []


# ---------------------------------------------------------------------------
# Plain skeletons
# ---------------------------------------------------------------------------

_FINGER = {"thumb", "index", "middle", "ring", "pinky", "little", "finger", "fingers", "metacarpal", "palm", "carpal"}
_HELPER = {"ik", "fk", "twist", "share", "sharebone", "roll", "end", "nub", "root", "pole", "target", "ctrl", "ctl",
           "mch", "org", "def", "tweak", "gun", "prop", "attach", "weapon", "helper", "corrective", "parent", "socket",
           "ref", "offset", "null", "dummy", "bend", "vis", "widget", "bind"}


def name_tokens(name: str) -> list[str]:
    """Lower-case words of a bone name without its namespace: ``mixamorig:LeftUpLeg`` -> left, up, leg;
    ``LThigh`` -> l, thigh; ``CC_Base_L_Thigh`` -> cc, base, l, thigh."""
    base = name.split(":")[-1].split("|")[-1]
    out: list[str] = []
    for part in re.split(r"[^A-Za-z0-9]+", base):
        out += [t.lower() for t in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+", part)]
    return out


def name_side(name: str) -> str:
    toks = name_tokens(name)
    if "left" in toks or "l" in toks or any(t.startswith("left") for t in toks):
        return "L"
    if "right" in toks or "r" in toks or any(t.startswith("right") for t in toks):
        return "R"
    return ""


def _is_helper(toks) -> bool:
    # also inside joined lower-case words: CC's "toebasesharebone", "forearmtwist01"
    return any(t in _HELPER or "twist" in t or "share" in t for t in toks)


def _is_foot(bone) -> bool:
    toks = name_tokens(bone.name)
    return (("foot" in toks or "ankle" in toks or any(t.endswith("foot") for t in toks)) and not _is_helper(toks)
            and not ({"toe", "toes", "ball", "heel"} & set(toks)))


def _is_hand(bone) -> bool:
    toks = name_tokens(bone.name)
    return (("hand" in toks or "wrist" in toks or any(t.endswith("hand") for t in toks)) and not _is_helper(toks)
            and not (_FINGER & set(toks)))


def _chain(end, min_len: float):
    """``(upper, lower)`` of the two-bone chain that ends in ``end``, or None.

    The joints must follow each other directly (a twist bone in between is a side branch, not part of the
    chain), and the two segments must look like a thigh and a shin (or an upper arm and a forearm)."""
    lower = end.parent
    upper = lower.parent if lower is not None else None
    if upper is None or _is_helper(name_tokens(lower.name)) or _is_helper(name_tokens(upper.name)):
        return None
    a = float(np.linalg.norm(np.array(lower.head_local) - np.array(upper.head_local)))
    b = float(np.linalg.norm(np.array(end.head_local) - np.array(lower.head_local)))
    if a < min_len or b < min_len or not 0.4 < a / b < 2.5:
        return None
    return upper, lower


def _ancestors(bone) -> list:
    out = []
    while bone is not None:
        out.append(bone)
        bone = bone.parent
    return out


def _common_ancestor(a, b):
    up = {x.name for x in _ancestors(b)}
    for x in _ancestors(a):
        if x.name in up:
            return x
    return None


def _pick_sides(cands: list, A: np.ndarray, forward: np.ndarray) -> dict:
    """``{'L': (end, upper, lower), 'R': ...}`` from chain candidates, by name or else by position."""
    out: dict = {}
    for c in cands:
        s = name_side(c[0].name)
        if s and s not in out:
            out[s] = c
    if len(out) < 2 and len(cands) >= 2:
        # No side in the names: the character's left is up x forward (facing -Y, left is +X).
        left = np.cross(np.array([0.0, 0.0, 1.0]), forward)
        pts = [A @ np.array(c[0].head_local) for c in cands[:2]]
        mid = 0.5 * (pts[0] + pts[1])
        first_left = float(np.dot(pts[0] - mid, left)) > 0.0
        out = {"L": cands[0] if first_left else cands[1], "R": cands[1] if first_left else cands[0]}
    return out


def _toe_of(foot):
    kids = [c for c in foot.children if not _is_helper(name_tokens(c.name))]
    for c in kids:
        if {"toe", "toes", "ball", "toebase"} & set(name_tokens(c.name)):
            return c
    for c in kids:
        if "toe" in c.name.lower():
            return c
    return max(kids, key=lambda c: c.length) if kids else None


def _inspect_skeleton(rig, base: RigProfile) -> RigProfile | None:
    """Profile of a plain humanoid skeleton, None when the armature is not one."""
    bones = rig.data.bones
    A = _upright(rig)
    scale = base.scale
    zs = [_uz(A, b.head_local) for b in bones]
    if not zs:
        return None
    span = max(zs) - min(zs)
    min_len = 0.05 * max(span, 1e-6)

    legs = [(b, *ch) for b in bones if _is_foot(b) for ch in [_chain(b, min_len)] if ch is not None]
    if len(legs) < 2:
        return None
    fwd = np.zeros(3)
    for end, _u, _l in legs:
        toe = _toe_of(end)
        tip = np.array(toe.head_local) if toe is not None else np.array(end.tail_local)
        fwd += A @ (tip - np.array(end.head_local))
    fwd[2] = 0.0
    fwd = fwd / np.linalg.norm(fwd) if np.linalg.norm(fwd) > 1e-9 else np.array([0.0, -1.0, 0.0])
    leg_sides = _pick_sides(legs, A, fwd)
    if set(leg_sides) != {"L", "R"}:
        return None
    hips = _common_ancestor(leg_sides["L"][1], leg_sides["R"][1])
    if hips is None or hips.name in (leg_sides["L"][1].name, leg_sides["R"][1].name):
        return None

    arms = [(b, *ch) for b in bones if _is_hand(b) for ch in [_chain(b, min_len)] if ch is not None
            and hips.name in {x.name for x in _ancestors(ch[0])}]
    arm_sides = _pick_sides(arms, A, fwd) if len(arms) >= 2 else {}

    profile = RigProfile(rig_name=rig.name, is_supported=True, kind="SKELETON", scale=scale)
    profile.torso_ctrl = hips.name
    profile.root_ctrl = _ancestors(hips)[-1].name
    # spine: hips up to the bone both arms hang from (the chest)
    chest = None
    if set(arm_sides) == {"L", "R"}:
        chest = _common_ancestor(arm_sides["L"][1], arm_sides["R"][1])
    path = []
    if chest is not None and hips.name in {x.name for x in _ancestors(chest)}:
        for x in _ancestors(chest):
            path.append(x.name)
            if x.name == hips.name:
                break
        path.reverse()
    profile.spine_bones = path or [hips.name]
    top = bones[profile.spine_bones[-1]]
    heads = [d for d in top.children_recursive if "head" in name_tokens(d.name) and not _is_helper(name_tokens(d.name))]
    if heads:
        head = min(heads, key=lambda d: len(_ancestors(d)))
        profile.head_bone = head.name
        below = {y.name for y in _ancestors(top)}
        profile.neck_bones = [x.name for x in reversed(_ancestors(head.parent)) if x.name not in below]

    # rest floor and height, upright
    floor_pts = []
    for end, _u, _l in leg_sides.values():
        floor_pts += [end.head_local, end.tail_local]
        toe = _toe_of(end)
        if toe is not None:
            floor_pts += [toe.head_local, toe.tail_local] + [d.head_local for d in toe.children_recursive]
    rest_floor = min(_uz(A, p) for p in floor_pts)
    top_pts = [b.head_local for b in bones]
    # Without a heel or toe-tip bone the lowest joint is the ball of the foot, a couple of centimetres above
    # the sole. The ankle joint sits at about 4.6 % of the body height above it (Winter, table 4.1).
    ankle_z = float(np.mean([_uz(A, end.head_local) for end, _u, _l in leg_sides.values()]))
    est_top = max(_uz(A, p) for p in top_pts)
    rest_floor = min(rest_floor, ankle_z - 0.046 * max(est_top - rest_floor, 1e-6))
    if profile.head_bone:
        top_pts.append(bones[profile.head_bone].tail_local)
    top_z = max(_uz(A, p) for p in top_pts)
    if profile.head_bone and bones[profile.head_bone].length < 0.03 * span:
        top_z += 0.075 * (top_z - rest_floor)          # a head joint without a head bone: the skull is ~7% of the height
    profile.rest_floor_z = rest_floor
    profile.character_height = max(top_z - rest_floor, 1e-3) * scale
    profile.unit = profile.character_height / 1.75

    for S, (end, upper, lower) in leg_sides.items():
        profile.legs[S] = _skeleton_limb("leg", S, upper, lower, end, A, scale, rest_floor)
    for S, (end, upper, lower) in arm_sides.items():
        profile.arms[S] = _skeleton_limb("arm", S, upper, lower, end, A, scale, rest_floor)
    _com_segments(profile, bones)
    return profile


def _skeleton_limb(kind: str, S: str, upper, lower, end, A: np.ndarray, scale: float, rest_floor: float) -> LimbSpec:
    limb = LimbSpec(
        kind=kind, side=S, ik_ctrl=end.name, pole_ctrl="", ik_base_ctrl="", switch_bone="",
        org_upper=upper.name, org_lower=lower.name, org_end=end.name,
        upper_len=max((lower.head_local - upper.head_local).length, 1e-5) * scale,
        lower_len=max((end.head_local - lower.head_local).length, 1e-5) * scale,
    )
    limb.end_radius = 0.35 * end.length * scale
    if kind == "leg":
        toe = _toe_of(end)
        limb.org_toe = toe.name if toe is not None else ""
        _foot_geometry(limb, end, toe, None, A, scale, rest_floor)
    return limb


def _uz(A: np.ndarray, p) -> float:
    """Upright height of an armature-space point."""
    return float(A[2] @ np.array(p, dtype=np.float64))


def _build_limb(bones, kind: str, S: str, scale: float, rest_floor: float, A: np.ndarray) -> LimbSpec | None:
    if kind == "leg":
        ik = f"foot_ik.{S}"
        upper, lower, end = f"ORG-thigh.{S}", f"ORG-shin.{S}", f"ORG-foot.{S}"
        pole, base, switch = f"thigh_ik_target.{S}", f"thigh_ik.{S}", f"thigh_parent.{S}"
    else:
        ik = f"hand_ik.{S}"
        upper, lower, end = f"ORG-upper_arm.{S}", f"ORG-forearm.{S}", f"ORG-hand.{S}"
        pole, base, switch = f"upper_arm_ik_target.{S}", f"upper_arm_ik.{S}", f"upper_arm_parent.{S}"
    if ik not in bones or upper not in bones or lower not in bones or end not in bones:
        return None

    b_ik, b_up, b_lo, b_end = bones[ik], bones[upper], bones[lower], bones[end]
    limb = LimbSpec(
        kind=kind,
        side=S,
        ik_ctrl=ik,
        pole_ctrl=_first(bones, pole),
        ik_base_ctrl=_first(bones, base),
        switch_bone=_first(bones, switch),
        org_upper=upper,
        org_lower=lower,
        org_end=end,
        upper_len=max((b_lo.head_local - b_up.head_local).length, 1e-5) * scale,
        lower_len=max((b_end.head_local - b_lo.head_local).length, 1e-5) * scale,
    )
    m_ik = _np4(b_ik.matrix_local)
    m_end = _np4(b_end.matrix_local)
    limb.x_rest = np.linalg.inv(m_ik) @ m_end
    limb.end_radius = 0.35 * b_end.length * scale

    if kind == "arm":
        return limb

    # ---- foot geometry -------------------------------------------------------------------------
    limb.org_toe = _first(bones, f"ORG-toe.{S}")
    limb.toe_ctrl = _first(bones, f"toe_ik.{S}", f"toe.{S}")
    limb.heel_ctrl = _first(bones, f"foot_heel_ik.{S}")
    heel_marker = None
    for child in b_end.children:
        if child.name.startswith("ORG-") and "heel" in child.name.lower() and not child.children:
            heel_marker = child
            break
    _foot_geometry(limb, b_end, bones[limb.org_toe] if limb.org_toe else None, heel_marker, A, scale, rest_floor)
    return limb


def _foot_geometry(limb: LimbSpec, b_end, b_toe, heel_marker, A: np.ndarray, scale: float, rest_floor: float) -> None:
    """Sole points, toe hinge and heel of a foot. Measured upright (``A``: armature -> upright rotation; the
    floor ``rest_floor`` is an upright height), stored in the foot bone's own space."""
    m_end = _np4(b_end.matrix_local)
    inv_end = np.linalg.inv(m_end)
    up = np.array([0.0, 0.0, 1.0])

    def U(p):                                          # armature -> upright
        return A @ np.array(p, dtype=np.float64)

    def to_local(inv_m, p_up):                         # upright point -> bone space
        return (inv_m @ np.append(A.T @ p_up, 1.0))[:3]

    ankle = U(b_end.head_local)
    ball = U(b_end.tail_local)
    toe_tip = None
    if b_toe is not None:
        ball = U(b_toe.head_local)
        toe_tip = U(b_toe.tail_local)

    forward = ball - ankle
    forward[2] = 0.0
    n = np.linalg.norm(forward)
    forward = forward / n if n > 1e-6 else np.array([0.0, -1.0, 0.0])
    lateral = np.cross(forward, up)

    foot_len_xy = max(float(np.linalg.norm((ball - ankle)[:2])), 1e-4)
    if heel_marker is not None:
        h_in = U(heel_marker.head_local)
        h_out = U(heel_marker.tail_local)
        half_w = 0.5 * float(np.linalg.norm((h_out - h_in)[:2]))
        lat = (h_out - h_in)
        lat[2] = 0.0
        if np.linalg.norm(lat) > 1e-6:
            lateral = lat / np.linalg.norm(lat)
        heel_c = 0.5 * (h_in + h_out)
    else:
        half_w = 0.2 * foot_len_xy
        heel_c = ankle - forward * 0.3 * foot_len_xy
    half_w = max(half_w, 0.1 * foot_len_xy)
    heel_c[2] = rest_floor
    ball_c = ball.copy()
    ball_c[2] = rest_floor
    pts = np.stack([
        heel_c - lateral * half_w,
        heel_c + lateral * half_w,
        ball_c - lateral * half_w * 1.15,
        ball_c + lateral * half_w * 1.15,
    ])
    limb.sole_points = np.stack([to_local(inv_end, p) for p in pts])
    joint_height = float(np.clip(ball[2] - rest_floor, 0.0, 0.35 * foot_len_xy)) if b_toe is not None else 0.0
    limb.sole_radius = np.array([0.0, 0.0, joint_height, joint_height])
    limb.up_local = inv_end[:3, :3] @ (A.T @ up)
    limb.heel_local = to_local(inv_end, heel_c)
    limb.ball_local = to_local(inv_end, ball_c)
    limb.lateral_local = inv_end[:3, :3] @ (A.T @ lateral)
    if toe_tip is not None:
        m_toe = _np4(b_toe.matrix_local)
        tip = toe_tip.copy()
        tip[2] = rest_floor
        limb.toe_tip_local = to_local(np.linalg.inv(m_toe), tip)
        limb.toe_in_foot_rest = inv_end @ m_toe
        limb.foot_length = float(np.linalg.norm((tip - heel_c)[:2])) * scale
    else:
        limb.foot_length = float(np.linalg.norm((ball_c - heel_c)[:2])) * scale
    limb.end_radius = max(float(ankle[2] - rest_floor), 0.0) * scale


def sole_contacts(limb: LimbSpec, W: np.ndarray) -> np.ndarray:
    """World-space contact points ``(T, K, 3)`` of a foot whose end bone has the world matrices ``W (T, 4, 4)``."""
    if not len(limb.sole_points):
        return np.zeros((len(W), 0, 3))
    centres = limb.sole_points + limb.sole_radius[:, None] * limb.up_local[None, :]
    pts = np.einsum("tij,kj->tki", W[:, :3, :3], centres) + W[:, None, :3, 3]
    pts[..., 2] -= limb.sole_radius[None, :] * np.linalg.norm(W[:, :3, 2], axis=1)[:, None]
    return pts


def _ik_target_bone(rig, limb: LimbSpec) -> str:
    """Name of the bone that the two-bone IK chain of ``limb`` aims at, ``""`` if there is none.

    Found by geometry, not by name: an IK constraint whose chain runs from the limb root to the
    limb end and whose target bone rests at the limb end.
    """
    bones = rig.data.bones
    root, end = bones[limb.org_upper].head_local, bones[limb.org_end].head_local
    tol = 0.02 * max((end - root).length, 1e-6)
    found = ""
    for pb in rig.pose.bones:
        for con in pb.constraints:
            if con.type != "IK" or con.target is None or con.subtarget not in bones or con.chain_count < 1:
                continue
            first = pb.bone
            for _ in range(con.chain_count - 1):
                if first.parent is None:
                    break
                first = first.parent
            if ((pb.bone.tail_local - end).length > tol or (first.head_local - root).length > tol
                    or (bones[con.subtarget].head_local - end).length > tol):
                continue
            if not con.mute:
                return con.subtarget
            found = found or con.subtarget
    return found


def _ik_aim(rig, target: str):
    """``(bone, point in that bone's space)`` marking the commanded limb end for the IK target bone ``target``.

    Usually the head of the IK target itself. Rigify however clamps that bone to the (slightly
    stretched) length of the limb with a Limit Distance constraint - the foot then stops short of
    what was asked for. The commanded position is where the bone's parent carries it.
    """
    if not target:
        return "", np.zeros(3)
    bones = rig.data.bones
    bone = bones[target]
    tol = 1e-4 * max(bone.length, 1e-6)
    clamped = False
    for con in rig.pose.bones[target].constraints:
        if con.type == "LIMIT_DISTANCE":
            clamped = True
        elif not (con.type == "COPY_LOCATION" and con.subtarget in bones and con.head_tail < 1e-6
                  and (bones[con.subtarget].head_local - bone.head_local).length < tol):
            return target, np.zeros(3)                 # driven by something else: take it as it is
    if clamped and bone.parent is not None:
        return bone.parent.name, np.array(bone.parent.matrix_local.inverted() @ bone.head_local)
    return target, np.zeros(3)


def commanded_end(limb: LimbSpec, smp):
    """World positions ``(T, 3)`` the IK chain of ``limb`` is asked to reach, ``None`` if the rig does not expose them.

    This differs from the IK control by whatever foot roll does (a heel roll of 40 degrees moves
    the ankle by a hand's width) and from the limb end itself where the limb runs out of length.
    """
    if limb.ik_aim_bone and smp.has(limb.ik_aim_bone):
        return smp.point_w(limb.ik_aim_bone, limb.ik_aim_local)
    return None


def solver_bone_names(profile: RigProfile, bones) -> list[str]:
    """All bones whose pose matrices the contact / kinematics solver needs from a sample."""
    names: set[str] = set()

    def add(*ns):
        for n in ns:
            if n and n in bones:
                names.add(n)
                parent = bones[n].parent
                if parent is not None:
                    names.add(parent.name)

    add(profile.root_ctrl, profile.torso_ctrl, profile.head_bone)
    add(*profile.spine_bones, *profile.neck_bones)
    for limb in profile.limbs():
        add(limb.ik_ctrl, limb.pole_ctrl, limb.ik_base_ctrl, limb.toe_ctrl, limb.heel_ctrl, limb.ik_target,
            limb.ik_aim_bone, limb.org_upper, limb.org_lower, limb.org_end, limb.org_toe)
    return [b.name for b in bones if b.name in names]


def bone_head_tail_world(rig, bone_name: str) -> tuple[Vector, Vector]:
    b = rig.data.bones[bone_name]
    mw = rig.matrix_world
    return mw @ b.head_local, mw @ b.tail_local
