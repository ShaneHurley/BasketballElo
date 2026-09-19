"""Pydantic request/response models for the dashboard API."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class JobCreate(BaseModel):
    module: str
    tab: str = "all"
    params: dict[str, Any] = Field(default_factory=dict)
    run_id: Optional[str] = None


class JobStatus(BaseModel):
    id: str
    module: str
    tab: str
    status: str
    progress_pct: float = 0.0
    stage: str = ""
    eta_seconds: Optional[float] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    result_paths: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)
    heaviness: str = "light"


class RunSummary(BaseModel):
    run_id: str
    n_games: int = 0
    spread_mae: Optional[float] = None
    total_mae: Optional[float] = None
    ats_hit_rate: Optional[float] = None
    approx_roi: Optional[float] = None
    actionable_bets: Optional[int] = None
    quote_source: Optional[str] = None
    ats_status: Optional[str] = None
    research_only_banner: Optional[str] = None


class ExportRequest(BaseModel):
    run_id: Optional[str] = None
    columns: list[str] = Field(default_factory=list)
    filter_expr: Optional[str] = None  # simple col==val only; not eval
    filename: Optional[str] = None
    tab: str = "all"


class ChartRequest(BaseModel):
    chart_id: str
    run_id: Optional[str] = None
