"""Job module registry: add a new experiment by registering one entry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ModuleInfo:
    id: str
    tab: str
    label: str
    heaviness: str
    param_schema: dict
    runner: Callable[[str, dict[str, Any]], list[str]]


_REGISTRY: dict[str, ModuleInfo] = {}


def register(
    module_id: str,
    *,
    tab: str,
    label: str,
    heaviness: str = "light",
    param_schema: dict | None = None,
):
    """Decorator to register a job runner."""
    def deco(fn: Callable):
        _REGISTRY[module_id] = ModuleInfo(
            id=module_id,
            tab=tab,
            label=label,
            heaviness=heaviness,
            param_schema=param_schema or {},
            runner=fn,
        )
        return fn
    return deco


def get_module(module_id: str) -> ModuleInfo | None:
    return _REGISTRY.get(module_id)


# Alias used by older call sites
get = get_module


def list_modules(tab: str | None = None) -> list[ModuleInfo]:
    rows = list(_REGISTRY.values())
    if tab and tab != "all":
        rows = [m for m in rows if m.tab in (tab, "all")]
    return rows


def list_module_dicts(tab: str | None = None) -> list[dict[str, Any]]:
    return [
        {
            "id": m.id,
            "tab": m.tab,
            "label": m.label,
            "heaviness": m.heaviness,
            "param_schema": m.param_schema,
        }
        for m in list_modules(tab)
    ]


def all_specs() -> dict[str, ModuleInfo]:
    return dict(_REGISTRY)
