"""The problem track of the MoCap Editor: one row per body part, a band wherever Analyze found something.

Drawn into a Dope Sheet that shows no channels (see ``ui.editor``), so the frame ruler, scrubbing, zoom and
pan are Blender's own. Bands use the issue colours of the heat map; the labels and row plates use the
theme of the Dope Sheet, so the track looks like the editor it is drawn in.
"""

from __future__ import annotations

from . import pin_tracks

GROUPS = (
    ("head", "Head"), ("torso", "Body"),
    ("arm.L", "Arm L"), ("hand.L", "Hand L"), ("arm.R", "Arm R"), ("hand.R", "Hand R"),
    ("leg.L", "Leg L"), ("leg.R", "Leg R"), ("other", "Other"),
)
GROUP_LABELS = dict(GROUPS)
KIND_LABELS = {
    "FLIP": "Flip", "SPIKE": "Spike", "KNEE_POP": "Pop", "JITTER": "Jitter", "HYPEREXT": "Over-stretch",
    "FROZEN": "Dropout", "PENETRATION": "Below floor", "SKATE": "Foot slide",
}
_FINGERS = ("finger", "thumb", "palm", "f_index", "f_middle", "f_ring", "f_pinky")


def group_of(profile, bone_name: str) -> str:
    """Row of the track a bone belongs to."""
    from ..ui import common
    low = (bone_name or "").lower()
    side = "L" if low.endswith((".l", "_l", "left")) else ("R" if low.endswith((".r", "_r", "right")) else "")
    if side and any(k in low for k in _FINGERS):
        return "hand." + side
    if any(k in low for k in ("head", "neck")):
        return "head"
    owner = common.owner_of_bone(profile, bone_name) if bone_name else ""
    return owner or "other"


def theme_colors(context):
    """``(text, plate, row)`` RGBA from the Dope Sheet theme."""
    try:
        th = context.preferences.themes[0].dopesheet_editor
        text = (*th.space.text, 1.0)
        plate = (*th.space_list.list, 0.95)
        back = th.space.back
        row = (back[0] + 0.035, back[1] + 0.035, back[2] + 0.035, 0.55)
        return text, plate, row
    except Exception:
        return (0.9, 0.9, 0.9, 1.0), (0.16, 0.16, 0.16, 0.95), (0.2, 0.2, 0.2, 0.55)


def layout(context, region, rig) -> dict:
    """Pixel layout: rows of the body parts that have issues, and one band per issue."""
    data = rig.mocap_cleanup
    from ..ui import common
    profile = common.profile_of(rig)
    s = pin_tracks.ui_scale(context)
    x0, sx, _y0, _sy = pin_tracks.view_mapping(region)
    row_h, gap = 18.0 * s, 3.0 * s
    top = region.height - 30.0 * s                    # below the frame ruler
    keys = {}
    for i, issue in enumerate(data.issues):
        keys.setdefault(group_of(profile, issue.bone), []).append(i)
    rows, bands = [], []
    for g, _label in GROUPS:
        if g not in keys:
            continue
        y1 = top - len(rows) * (row_h + gap)
        y0 = y1 - row_h
        if y0 < 0:
            break
        rows.append((g, y0, y1))
        for i in keys[g]:
            issue = data.issues[i]
            xa = x0 + sx * (issue.frame_start - 0.5)
            xb = max(x0 + sx * (issue.frame_end + 0.5), xa + 2.0)
            bands.append({"index": i, "kind": issue.kind, "x0": xa, "x1": xb, "y0": y0 + 2.0 * s, "y1": y1 - 2.0 * s,
                          "severity": float(issue.severity)})
    return {"rows": rows, "bands": bands, "label_w": 64.0 * s, "scale": s}


def pins_top(context, region, rig) -> float:
    """Top of the pin rows in the MoCap Editor: right under the problem rows."""
    lay = layout(context, region, rig)
    s = lay["scale"]
    if lay["rows"]:
        return lay["rows"][-1][1] - 8.0 * s
    return region.height - 30.0 * s


def hit_test(lay: dict, mx: float, my: float, grab: float = 3.0):
    """Index of the issue under the mouse (the shortest band wins where they overlap), or -1."""
    best, width = -1, None
    for b in lay["bands"]:
        if b["y0"] - grab <= my <= b["y1"] + grab and b["x0"] - grab <= mx <= b["x1"] + grab:
            w = b["x1"] - b["x0"]
            if width is None or w < width:
                best, width = b["index"], w
    return best


def draw(context, region, rig, rects, lines, text) -> None:
    """Draw the track with the overlay's primitives (``rects(list, color)``, ``lines(points, color, w)``,
    ``text(x, y, str, size, color)``)."""
    from .draw import ISSUE_COLORS
    data = rig.mocap_cleanup
    if not len(data.issues):
        return
    lay = layout(context, region, rig)
    if not lay["rows"]:
        return
    s = lay["scale"]
    txt, plate, row = theme_colors(context)
    rects([(0.0, y0, region.width, y1) for _g, y0, y1 in lay["rows"]], row)
    stale = 0.55 if data.analysis_stale else 1.0
    by_color: dict[tuple, list] = {}
    active = None
    for b in lay["bands"]:
        if b["x1"] < lay["label_w"] or b["x0"] > region.width:
            continue
        c = ISSUE_COLORS.get(b["kind"], (0.8, 0.8, 0.8))
        a = round((0.45 + 0.5 * min(max(b["severity"], 0.0), 1.0)) * stale, 2)
        by_color.setdefault((*c, a), []).append((b["x0"], b["y0"], b["x1"], b["y1"]))
        if b["index"] == data.active_issue:
            active = b
    for col, rs in by_color.items():
        rects(rs, col)
    if active is not None:
        xa, xb, ya, yb = active["x0"], active["x1"], active["y0"], active["y1"]
        lines([(xa, ya), (xb, ya), (xb, ya), (xb, yb), (xb, yb), (xa, yb), (xa, yb), (xa, ya)], (1.0, 1.0, 1.0, 0.95), 1.5)
    rects([(0.0, y0, lay["label_w"], y1) for _g, y0, y1 in lay["rows"]], plate)
    for g, y0, y1 in lay["rows"]:
        text(6.0 * s, y0 + 5.0 * s, GROUP_LABELS.get(g, g), 11.0 * s, txt)
    # legend: the kinds that occur, right-aligned along the bottom edge
    kinds = [k for k in KIND_LABELS if any(b["kind"] == k for b in lay["bands"])]
    x = region.width - 8.0 * s
    y = 20.0 * s
    if True:
        for k in reversed(kinds):
            label = KIND_LABELS[k]
            w = (len(label) * 6.2 + 18.0) * s
            x -= w
            rects([(x, y, x + 9.0 * s, y + 9.0 * s)], (*ISSUE_COLORS.get(k, (0.8, 0.8, 0.8)), 0.95))
            text(x + 13.0 * s, y + 1.0 * s, label, 10.0 * s, txt)
