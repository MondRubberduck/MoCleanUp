#!/usr/bin/env bash
# usage: all.sh [5.2|5.3]  - run the whole suite, print PASS/FAIL lines and results only
cd "$(dirname "$0")"
V="${1:-5.3}"
for t in test_core test_filters test_diagnostics test_contacts test_kinematics test_drift test_simplifier test_addon test_ui test_robustness; do
  echo "=== $t ($V) ==="
  timeout 500 ./t.sh $t $V 400 2>&1 | grep -E "^\[(PASS|FAIL)\]|RESULT:|Error:|Traceback|AssertionError" | grep -v "residual detail layer"
done
