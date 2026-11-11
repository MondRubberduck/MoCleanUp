"""Static check (plain Python, no Blender): every name that is read somewhere in the package is defined.

    python tests/check_names.py

Catches a function that was deleted or renamed while something still calls it - without having to
run the code path that would hit it.
"""

import ast
import builtins
import io
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def undefined_names(path: str) -> list[tuple[int, str]]:
    tree = ast.parse(io.open(path, encoding="utf-8").read())
    defined = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__package__", "__spec__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Import):
            defined.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            defined.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            defined.add(node.id)
        elif isinstance(node, ast.arg):
            defined.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            defined.update(node.names)
    return [(n.lineno, n.id) for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in defined]


def main() -> int:
    bad = 0
    for folder, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", "dist", ".git"}]
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.join(folder, name)
                for line, ident in undefined_names(path):
                    print(f"{os.path.relpath(path, ROOT)}:{line}: undefined name '{ident}'")
                    bad += 1
    print("RESULT: no undefined names" if not bad else f"RESULT: {bad} undefined names")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
