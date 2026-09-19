"""CSV export helpers."""
from __future__ import annotations

import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.config import EXPORTS_DIR
from dashboard.paths import results_csv_path


def export_csv(
    run_id: str | None,
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    filename: str | None = None,
) -> str:
    path = results_csv_path(run_id)
    if path is None or not path.exists():
        raise FileNotFoundError("No results CSV")
    df = pd.read_csv(path)
    if columns:
        use = [c for c in columns if c in df.columns]
        if use:
            df = df[use]
    for k, v in (filters or {}).items():
        if k in df.columns:
            df = df[df[k].astype(str) == str(v)]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    name = filename or f"export_{run_id or 'latest'}_{ts}.csv"
    out = EXPORTS_DIR / Path(name).name
    df.to_csv(out, index=False)
    return str(out)


def bundle_exports(paths: list[str], zip_name: str | None = None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    zip_path = EXPORTS_DIR / (zip_name or f"bundle_{ts}.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in paths:
            fp = Path(p)
            if fp.exists() and fp.is_file():
                zf.write(fp, arcname=fp.name)
    return str(zip_path)
