"""Tiny patch helper: python rp.py <patchfile.py>.  The patch file defines PATCHES = [(relpath, old, new), ...]."""
import io
import os
import runpy
import sys

ROOT = r"C:\Users\kondi\Documents\GIT_Folder\Mocap_Blender"
ns = runpy.run_path(sys.argv[1])
for rel, old, new in ns["PATCHES"]:
    p = os.path.join(ROOT, rel)
    s = io.open(p, encoding="utf-8").read()
    if old not in s:
        print("NOT FOUND in", rel, ":", old[:90].replace("\n", "\\n"))
        sys.exit(1)
    if s.count(old) > 1:
        print("AMBIGUOUS in", rel, ":", old[:90].replace("\n", "\\n"))
        sys.exit(1)
    io.open(p, "w", encoding="utf-8", newline="\n").write(s.replace(old, new))
print("patched", len(ns["PATCHES"]))
