#!/usr/bin/env bash
# Run the headless test suite: tests/run.sh [blender.exe] [test names...]
# Example: tests/run.sh "/c/Program Files/Blender Foundation/Blender 5.3/blender.exe" test_core test_filters
#
# Each test file generates (and caches in the temp dir) a real Rigify human rig, so Rigify must be
# available in the Blender that runs it. A hung Blender is killed after 10 minutes.
set -u
BLENDER="${1:-blender}"
shift || true
HERE="$(cd "$(dirname "$0")" && pwd -W 2>/dev/null || pwd)"
TESTS=("$@")
if [ ${#TESTS[@]} -eq 0 ]; then
  TESTS=(test_core test_filters test_diagnostics test_contacts test_kinematics test_drift test_simplifier
         test_addon test_ui test_robustness test_skeleton)
  # static check, plain Python: every name that is used is defined
  # the first Python that actually runs (on Windows "python3" can be the Microsoft Store stub)
  PY=""
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c "" >/dev/null 2>&1; then PY="$c"; break; fi
  done
  if [ -n "$PY" ]; then
    echo "=== check_names ==="
    "$PY" "$HERE/check_names.py" || exit 1
  fi
fi
FAIL=0
for t in "${TESTS[@]}"; do
  echo "=== $t ==="
  timeout 600 "$BLENDER" -b --factory-startup --python "$HERE/$t.py" 2>&1 \
    | grep -E "^\[(PASS|FAIL)\]|RESULT:|Traceback|^  File|^\w*Error|assert" || true
  if [ "${PIPESTATUS[0]}" -ne 0 ]; then FAIL=1; fi
done
exit $FAIL
