"""FastAPI dashboard for kernel profile regression analysis.

Reads CSV/JSON data produced by generate_report.py and serves an interactive
Chart.js dashboard — following the same pattern as perf-regression/dashboard.py.

Data is organized under a root report directory with timestamp folders:
    report/
    └── 20260304_093600/       ← timestamp folder (one per generate_report run)
        ├── index.json
        ├── rocm700_TP8/       ← config group folder
        │   ├── runs.csv, metrics.csv, changes.csv, kernel_diffs.json, meta.json
        └── rocm720_TP8/
            └── ...

Usage:
    # Generate data first:
    python -m report.generate_report /home/yichiche/benchmark_runs/

    # Launch dashboard:
    python -m report.dashboard
    # or:
    cd /home/yichiche/agent-box && uvicorn report.dashboard:app --host 0.0.0.0 --port 8081
"""

import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
AGENT_BOX_DIR = BASE_DIR.parent
HOST_HOME_DIR = os.environ.get("AGENT_BOX_HOST_HOME", str(AGENT_BOX_DIR.parent))
REPORT_ROOT = Path(os.environ.get("KERNEL_PROFILE_DATA_DIR", str(Path(HOST_HOME_DIR) / "report")))
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
DASHBOARD_PORT = 8081

_TS_RE = re.compile(r"^\d{8}_\d{6}$")

app = FastAPI(title="Kernel Profile Regression Dashboard")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ── Data loading helpers ────────────────────────────────────────────────────

def _list_report_timestamps() -> list[str]:
    """Return available timestamp folder names under REPORT_ROOT, sorted descending."""
    if not REPORT_ROOT.is_dir():
        return []
    timestamps = []
    for d in REPORT_ROOT.iterdir():
        if d.is_dir() and _TS_RE.match(d.name):
            timestamps.append(d.name)
    timestamps.sort(reverse=True)
    return timestamps


def _resolve_report_dir(report: Optional[str] = None) -> Path:
    """Resolve the active report directory (timestamp folder).

    If report is given, use it directly. Otherwise, find the latest timestamp folder.
    """
    if report:
        return REPORT_ROOT / report
    timestamps = _list_report_timestamps()
    if timestamps:
        return REPORT_ROOT / timestamps[0]
    return REPORT_ROOT


def _list_groups(report: Optional[str] = None) -> list[str]:
    """Return config group folder names from index.json or by scanning."""
    data_dir = _resolve_report_dir(report)
    idx = data_dir / "index.json"
    if idx.exists():
        with open(idx) as f:
            return json.load(f).get("groups", [])
    # Fallback: scan for subdirs containing meta.json
    groups = []
    if data_dir.is_dir():
        for d in sorted(data_dir.iterdir()):
            if d.is_dir() and (d / "meta.json").exists():
                groups.append(d.name)
    return groups


def _load_csv(group: str, filename: str, report: Optional[str] = None) -> list[dict]:
    path = _resolve_report_dir(report) / group / filename
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_json(group: str, filename: str, report: Optional[str] = None):
    path = _resolve_report_dir(report) / group / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_root_json(filename: str, report: Optional[str] = None):
    path = _resolve_report_dir(report) / filename
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/reports")
async def get_reports():
    """Return list of available report timestamps (newest first)."""
    timestamps = _list_report_timestamps()
    result = []
    for ts in timestamps:
        index_data = _load_root_json("index.json", report=ts)
        result.append({
            "name": ts,
            "generated_at": index_data.get("generated_at", "") if index_data else "",
            "n_groups": len(index_data.get("groups", [])) if index_data else 0,
        })
    return JSONResponse(result)


@app.get("/api/groups")
async def get_groups(report: Optional[str] = Query(None)):
    """Return list of config groups."""
    groups = _list_groups(report=report)
    result = []
    for g in groups:
        meta = _load_json(g, "meta.json", report=report) or {}
        result.append({
            "name": g,
            "rocm_version": meta.get("rocm_version", ""),
            "tp_size": meta.get("tp_size", 0),
            "n_runs": meta.get("n_runs", 0),
            "n_significant": meta.get("n_significant", 0),
            "threshold": meta.get("threshold", 0),
        })
    return JSONResponse(result)


@app.get("/api/{group}/meta")
async def get_meta(group: str, report: Optional[str] = Query(None)):
    meta = _load_json(group, "meta.json", report=report)
    return JSONResponse(meta or {})


@app.get("/api/{group}/chart-data")
async def chart_data(group: str, stage: Optional[str] = Query(None), report: Optional[str] = Query(None)):
    """Return Chart.js formatted data for time trend charts."""
    rows = _load_csv(group, "metrics.csv", report=report)
    changes_rows = _load_csv(group, "changes.csv", report=report)

    sig_set = set()
    for c in changes_rows:
        if c.get("significant") == "True":
            sig_set.add((c["new_dir"], c["layer_type"], c["stage"]))

    component_colors = {
        "attention": "#4E79A7",
        "moe": "#F28E2B",
        "linear": "#59A14F",
        "comm": "#E15759",
        "quant": "#B07AA1",
    }
    component_fields = {
        "attention": "avg_attention_us",
        "moe": "avg_moe_us",
        "linear": "avg_linear_us",
        "comm": "avg_comm_us",
        "quant": "avg_quant_us",
    }

    stages_to_show = [stage] if stage and stage != "all" else ["prefill", "decode"]
    charts = {}

    for stg in stages_to_show:
        stg_rows = [r for r in rows if r["stage"] == stg]
        layer_types = sorted(set(r["layer_type"] for r in stg_rows))

        total_datasets = {}
        for lt in layer_types:
            lt_rows = sorted(
                [r for r in stg_rows if r["layer_type"] == lt],
                key=lambda r: r["date"],
            )
            if not lt_rows:
                continue

            data_points = []
            point_colors = []
            for r in lt_rows:
                x_date = f"{r['date'][:4]}-{r['date'][4:6]}-{r['date'][6:]}"
                y_val = float(r["avg_total_us"])
                is_regression = (r["dir_name"], lt, stg) in sig_set
                data_points.append({
                    "x": x_date,
                    "y": round(y_val, 2),
                    "dir": r["dir_name"],
                    "layers": int(r["layer_count"]),
                })
                point_colors.append("#FF0000" if is_regression else "#333")

            total_datasets[lt] = {
                "label": f"{lt} — Total",
                "data": data_points,
                "borderColor": "#333",
                "backgroundColor": "#33333333",
                "pointBackgroundColor": point_colors,
                "pointRadius": [8 if c == "#FF0000" else 4 for c in point_colors],
                "pointStyle": ["triangle" if c == "#FF0000" else "circle" for c in point_colors],
                "tension": 0.1,
                "borderWidth": 2,
            }

        component_datasets = {}
        for lt in layer_types:
            lt_rows = sorted(
                [r for r in stg_rows if r["layer_type"] == lt],
                key=lambda r: r["date"],
            )
            for comp_name, comp_field in component_fields.items():
                data_points = []
                for r in lt_rows:
                    x_date = f"{r['date'][:4]}-{r['date'][4:6]}-{r['date'][6:]}"
                    y_val = float(r[comp_field])
                    data_points.append({"x": x_date, "y": round(y_val, 2), "dir": r["dir_name"]})

                if not any(p["y"] > 0 for p in data_points):
                    continue

                key = f"{lt}-{comp_name}"
                color = component_colors[comp_name]
                component_datasets[key] = {
                    "label": f"{lt} — {comp_name.capitalize()}",
                    "data": data_points,
                    "borderColor": color,
                    "backgroundColor": color + "33",
                    "pointRadius": 3,
                    "tension": 0.1,
                    "borderWidth": 1.5,
                    "borderDash": [4, 3],
                }

        charts[stg] = {
            "label": f"{stg.capitalize()} Avg Layer Time (us)",
            "total_datasets": list(total_datasets.values()),
            "component_datasets": list(component_datasets.values()),
        }

    return JSONResponse(charts)


@app.get("/api/{group}/changes")
async def get_changes(group: str, significant_only: bool = Query(False), report: Optional[str] = Query(None)):
    """Return change events as JSON."""
    rows = _load_csv(group, "changes.csv", report=report)
    if significant_only:
        rows = [r for r in rows if r.get("significant") == "True"]

    result = []
    for r in rows:
        result.append({
            "layer_type": r["layer_type"],
            "stage": r["stage"],
            "old_dir": r["old_dir"],
            "old_date": r["old_date"],
            "new_dir": r["new_dir"],
            "new_date": r["new_date"],
            "old_time_us": float(r["old_time_us"]),
            "new_time_us": float(r["new_time_us"]),
            "pct_change": float(r["pct_change"]),
            "significant": r["significant"] == "True",
        })
    return JSONResponse(result)


@app.get("/api/{group}/kernel-diffs")
async def get_kernel_diffs(group: str, report: Optional[str] = Query(None)):
    data = _load_json(group, "kernel_diffs.json", report=report)
    return JSONResponse(data or [])


@app.get("/api/{group}/runs")
async def get_runs(group: str, report: Optional[str] = Query(None)):
    rows = _load_csv(group, "runs.csv", report=report)
    return JSONResponse(rows)


@app.get("/api/health")
async def health(report: Optional[str] = Query(None)):
    groups = _list_groups(report=report)
    index_data = _load_root_json("index.json", report=report)
    report_dir = _resolve_report_dir(report)
    return JSONResponse({
        "status": "ok" if groups else "no_data",
        "data_dir": str(report_dir),
        "report": report_dir.name if report_dir != REPORT_ROOT else None,
        "n_groups": len(groups),
        "groups": groups,
        "threshold": index_data.get("threshold") if index_data else None,
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("report.dashboard:app", host="0.0.0.0", port=DASHBOARD_PORT, reload=True)
