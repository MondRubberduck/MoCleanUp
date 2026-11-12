#!/usr/bin/env bash
# usage: d33.sh <fixture> <rig> <L|R> <a> <b> [step]
S="C:/Users/kondi/AppData/Local/hermes/cache/scratch/mc"
export BLENDER_USER_RESOURCES="$S/userres"; export TEMP='C:\Users\kondi\AppData\Local\hermes\cache\scratch\mc\tmp'; export TMP="$TEMP"
f="$1"; shift
timeout 300 "/c/Program Files/Blender Foundation/Blender 5.3/blender.exe" -b --factory-startup -y --python "$S/dbg33.py" -- "$S/fx/$f.blend" "$@" 2>&1 | grep -E "^Solved|^pins|^leg|^  |Error|Trace" | cut -c1-${W:-250}
