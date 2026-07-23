"""Repo-root pytest bootstrap.

Guarantees the canonical ``pipeline``/``tests`` tree at the repository root
always wins name resolution over the historical ``BasketballElo/pipeline``
duplicate, even if some other tool has already poisoned ``sys.path`` with the
``BasketballElo`` directory (both trees contain a top-level ``pipeline``
package, so import order determines which one Python resolves).

This file must not import from ``BasketballElo``.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent

# Remove any already-imported "pipeline*" modules that did not come from the
# canonical root, then force the root to the front of sys.path so a fresh
# `import pipeline` always resolves to the production tree.
_root_str = str(_REPO_ROOT)
if _root_str in sys.path:
    sys.path.remove(_root_str)
sys.path.insert(0, _root_str)

for _name in list(sys.modules):
    if _name == "pipeline" or _name.startswith("pipeline."):
        _mod = sys.modules[_name]
        _file = getattr(_mod, "__file__", None)
        if _file and not str(Path(_file).resolve()).startswith(_root_str):
            del sys.modules[_name]
