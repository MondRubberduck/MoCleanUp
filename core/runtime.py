"""Runtime caches shared between stages, operators and overlays (never saved to the .blend)."""

from __future__ import annotations

# rig name -> DiagnosticReport (stage 1)
REPORTS: dict[str, object] = {}
# rig name -> dict with sampled world data for the viewport HUD (centre of mass, soles, contacts)
MOTION: dict[str, dict] = {}
# rig name -> (token, samples, follow table) for fast contact re-solves
SOLVE_SAMPLES: dict[str, tuple] = {}
# rig name -> SolveResult of the last contact solve
SOLVE_RESULTS: dict[str, object] = {}
# (action name, data path, index) -> (f0, values) raw ghost curves for the Graph Editor
GHOSTS: dict[tuple, tuple] = {}
# rig name -> (cache key, RigProfile)
PROFILES: dict[str, tuple] = {}


def clear_rig(name: str) -> None:
    for store in (REPORTS, MOTION, SOLVE_SAMPLES, SOLVE_RESULTS, PROFILES):
        store.pop(name, None)
    GHOSTS.clear()


def clear_all() -> None:
    for store in (REPORTS, MOTION, SOLVE_SAMPLES, SOLVE_RESULTS, GHOSTS, PROFILES):
        store.clear()
