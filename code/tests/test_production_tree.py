"""Task 001: the root pipeline/tests tree is the only production code tree.

`BasketballElo/pipeline` is a historical duplicate. It must never be the
source of a module resolved as `pipeline` (or any `pipeline.*` submodule) by
production code or tests.
"""
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORICAL_DUPLICATE = REPO_ROOT / "BasketballElo"

# A representative sample of modules touched by the tasks in this phase.
_SAMPLE_SUBMODULES = [
    "pipeline",
    "pipeline.config",
    "pipeline.ingest",
    "pipeline.dates",
    "pipeline.preprocess",
    "pipeline.stints",
    "pipeline.teamstats",
    "pipeline.features",
    "pipeline.simulate",
    "pipeline.game_updates",
    "pipeline.artifacts",
    "pipeline.game_results",
]


def test_pipeline_modules_resolve_to_repo_root():
    """Every imported `pipeline` module must live under the repo root, never
    under the historical `BasketballElo/pipeline` duplicate."""
    printed = []
    for mod_name in _SAMPLE_SUBMODULES:
        mod = importlib.import_module(mod_name)
        mod_file = Path(mod.__file__).resolve()
        printed.append(f"{mod_name} -> {mod_file}")
        assert str(mod_file).startswith(str(REPO_ROOT)), (
            f"{mod_name} resolved outside the repo root: {mod_file}"
        )
        assert HISTORICAL_DUPLICATE not in mod_file.parents, (
            f"{mod_name} resolved to the historical duplicate tree: {mod_file}"
        )
    for line in printed:
        print(line)
    assert len(printed) == len(_SAMPLE_SUBMODULES)


def test_no_pipeline_module_currently_loaded_from_duplicate():
    """Scan every already-imported module named `pipeline*` in this process."""
    offenders = []
    for name, mod in list(sys.modules.items()):
        if not (name == "pipeline" or name.startswith("pipeline.")):
            continue
        mod_file = getattr(mod, "__file__", None)
        if mod_file is None:
            continue
        resolved = Path(mod_file).resolve()
        if HISTORICAL_DUPLICATE in resolved.parents:
            offenders.append((name, str(resolved)))
    assert not offenders, f"pipeline modules resolved from historical duplicate: {offenders}"


def test_historical_duplicate_is_not_a_top_level_package_on_path():
    """`BasketballElo` itself must not be importable as a top-level package
    (it has no `__init__.py`), so `import BasketballElo.pipeline` cannot
    silently succeed from the repo root."""
    assert not (HISTORICAL_DUPLICATE / "__init__.py").exists()
