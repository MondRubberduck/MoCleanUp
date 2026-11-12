#!/usr/bin/env bash
# usage: q.sh <fixture name in fx/ | full path> <rig> [real_take options...]   (source tree, Blender 5.3 unless V=5.2)
S="C:/Users/kondi/AppData/Local/hermes/cache/scratch/mc"
export BLENDER_USER_RESOURCES="$S/userres"; export TEMP='C:\Users\kondi\AppData\Local\hermes\cache\scratch\mc\tmp'; export TMP="$TEMP"
B="/c/Program Files/Blender Foundation/Blender ${V:-5.3}/blender.exe"
f="$1"; case "$f" in *.blend) ;; *) f="$S/fx/$f.blend";; esac
rig="$2"; shift 2
timeout ${TMO:-560} "$B" -b --factory-startup -y --python "C:/Users/kondi/Documents/GIT_Folder/Mocap_Blender/tests/real_take.py" -- "$f" "$rig" source=1 "$@" 2>&1 \
  | grep -E "^rig |^  |^RESULT|Traceback|Error|^    [a-zA-Z_]+\(|File \"" | grep -v -E "anim.driver|Saved:|^  set |character meshes" | cut -c1-${W:-260}
