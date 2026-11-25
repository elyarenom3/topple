"""Import guard: the estimator and the planner never import the ground truth (Section 4.1)."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

FORBIDDEN = ("topple.harness", "harness.truth", "topple.validation")
PROTECTED = ("perception", "planning")


def imports_of(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if node.level:
                mod = "." * node.level + mod
            out.append(mod)
    return out


def assert_no_truth_imports(root: Path | None = None) -> list[str]:
    """Return a list of violations (empty if clean). Raises AssertionError on violations."""
    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    bad = []
    for pkg in PROTECTED:
        for py in (root / pkg).rglob("*.py"):
            for imp in imports_of(py):
                if any(f in imp for f in FORBIDDEN) or imp.startswith("..harness") or imp.startswith("..validation"):
                    bad.append(f"{py.relative_to(root)} imports {imp}")
    # runtime check: importing the protected packages must not load the harness
    for pkg in PROTECTED:
        before = set(sys.modules)
        __import__(f"topple.{pkg}")
        loaded = set(sys.modules) - before
        if any(m.startswith("topple.harness") for m in loaded):
            bad.append(f"importing topple.{pkg} loaded topple.harness")
    assert not bad, "ground truth import violations: " + "; ".join(bad)
    return bad
