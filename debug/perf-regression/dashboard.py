"""FastAPI web dashboard for viewing benchmark results and regression alerts."""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import config as _cfg
from config import DASHBOARD_PORT, DB_PATH, TEMPLATES_DIR, STATIC_DIR, ROCM_VERSIONS
from collector import get_connection, init_db, get_version_snapshot

logger = logging.getLogger(__name__)
app = FastAPI(title="SGLang ROCm Perf Regression Dashboard")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.on_event("startup")
def startup():
    init_db()
    logger.info("Dashboard database path: %s", DB_PATH)


def _get_conn() -> sqlite3.Connection:
    return get_connection()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/api/models")
async def get_models():
    """Return distinct model names from the database."""
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT DISTINCT model_name FROM benchmark_runs ORDER BY model_name"
        ).fetchall()
        return JSONResponse([r["model_name"] for r in rows])
    finally:
        conn.close()


@app.get("/api/chart-data")
async def chart_data(
    rocm_version: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
    tp_size: Optional[str] = Query(None),
    mtp_enabled: Optional[str] = Query(None),
    ep_size: Optional[str] = Query(None),
    dp_size: Optional[str] = Query(None),
    days: int = Query(30),
    concurrency: Optional[str] = Query(None),
):
    """Return pre-formatted Chart.js data for the 6 metric charts + accuracy."""
    conn = _get_conn()
    try:
        # Build WHERE clause
        where_parts = ["br.status = 'completed'"]
        params: list = []

        if model_name and model_name != "all":
            where_parts.append("br.model_name = ?")
            params.append(model_name)

        if rocm_version and rocm_version != "all":
            where_parts.append("br.rocm_version = ?")
            params.append(rocm_version)

        if tp_size and tp_size != "all":
            where_parts.append("br.tp_size = ?")
            params.append(int(tp_size))

        if mtp_enabled and mtp_enabled != "all":
            where_parts.append("br.mtp_enabled = ?")
            params.append(int(mtp_enabled))

        if ep_size and ep_size != "all":
            if ep_size == "none":
                where_parts.append("br.ep_size IS NULL")
            else:
                where_parts.append("br.ep_size = ?")
                params.append(int(ep_size))

        if dp_size and dp_size != "all":
            if dp_size == "none":
                where_parts.append("br.dp_size IS NULL")
            else:
                where_parts.append("br.dp_size = ?")
                params.append(int(dp_size))

        if days > 0:
            where_parts.append(
                f"br.build_date >= strftime('%Y%m%d', 'now', '-{days} days')"
            )

        where_clause = " AND ".join(where_parts)

        # Build concurrency filter
        conc_filter = ""
        if concurrency and concurrency != "all":
            conc_filter = " AND bm.concurrency = ?"
            params_metrics = params + [int(concurrency)]
        else:
            params_metrics = list(params)

        # Fetch metric data
        metrics_query = f"""
            SELECT
                br.build_date,
                br.image_tag,
                br.rocm_version,
                br.sglang_version,
                br.tp_size,
                br.mtp_enabled,
                br.ep_size,
                br.dp_size,
                bm.concurrency,
                bm.output_throughput,
                bm.total_throughput,
                bm.median_ttft_ms,
                bm.median_itl_ms,
                bm.median_e2e_latency_ms,
                bm.p99_e2e_latency_ms
            FROM benchmark_metrics bm
            JOIN benchmark_runs br ON bm.run_id = br.id
            WHERE {where_clause}{conc_filter}
            ORDER BY br.build_date, br.rocm_version, br.tp_size, br.mtp_enabled, br.ep_size, br.dp_size, bm.concurrency
        """
        rows = conn.execute(metrics_query, params_metrics).fetchall()

        # Fetch accuracy data
        accuracy_query = f"""
            SELECT
                br.build_date,
                br.image_tag,
                br.rocm_version,
                br.sglang_version,
                br.tp_size,
                br.mtp_enabled,
                br.ep_size,
                br.dp_size,
                ar.accuracy_pct
            FROM accuracy_results ar
            JOIN benchmark_runs br ON ar.run_id = br.id
            WHERE {where_clause}
            ORDER BY br.build_date, br.rocm_version, br.tp_size, br.mtp_enabled, br.ep_size, br.dp_size
        """
        accuracy_rows = conn.execute(accuracy_query, params).fetchall()

        # Fetch regression alerts for marking points
        alerts_query = f"""
            SELECT
                ra.metric_name,
                ra.concurrency,
                br.build_date,
                br.rocm_version
            FROM regression_alerts ra
            JOIN benchmark_runs br ON ra.run_id = br.id
            WHERE {where_clause}
        """
        alert_rows = conn.execute(alerts_query, params).fetchall()
        alert_set = set()
        for a in alert_rows:
            alert_set.add(
                (a["metric_name"], a["concurrency"], a["build_date"], a["rocm_version"])
            )

        # Build chart datasets
        # Group by (rocm_version, mtp_enabled, concurrency) for metric charts
        metric_charts = {
            "output_throughput": {"label": "Output Throughput (tok/s)", "datasets": {}},
            "total_throughput": {"label": "Total Throughput (tok/s)", "datasets": {}},
            "median_ttft_ms": {"label": "Median TTFT (ms)", "datasets": {}},
            "median_itl_ms": {"label": "Median ITL (ms)", "datasets": {}},
            "median_e2e_latency_ms": {"label": "Median E2E Latency (ms)", "datasets": {}},
            "p99_e2e_latency_ms": {"label": "P99 E2E Latency (ms)", "datasets": {}},
        }

        # Color palette: ROCm version x TP size x concurrency
        # ROCm 700 (cool): TP2=sapphire, TP4=teal, TP8=indigo
        # ROCm 720 (warm): TP2=coral,    TP4=bronze, TP8=crimson
        # Each (rocm, tp) has 5 shades for concurrency (dark -> light)
        def _adjust_shade(hex_color: str, factor: float) -> str:
            """Lighten (factor>1) or darken (factor<1) a hex color."""
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            if factor > 1.0:
                r = min(255, int(r + (255 - r) * (factor - 1.0)))
                g = min(255, int(g + (255 - g) * (factor - 1.0)))
                b = min(255, int(b + (255 - b) * (factor - 1.0)))
            else:
                r = max(0, int(r * factor))
                g = max(0, int(g * factor))
                b = max(0, int(b * factor))
            return f"#{r:02X}{g:02X}{b:02X}"

        rocm_tp_base = {
            ("700", 2): "#60A5FA",  # sky blue
            ("700", 4): "#34D399",  # mint green
            ("700", 8): "#A78BFA",  # lavender
            ("720", 2): "#FB923C",  # light orange
            ("720", 4): "#FACC15",  # sunny yellow
            ("720", 8): "#F472B6",  # bubblegum pink
        }
        # Concurrency shade factors: lower concurrency = darker, higher = lighter
        conc_shade = {1: 0.70, 2: 0.85, 4: 1.0, 8: 1.25, 16: 1.50}

        rocm_tp_conc_colors = {}
        for (rocm, tp), base in rocm_tp_base.items():
            for conc, shade in conc_shade.items():
                rocm_tp_conc_colors[(rocm, tp, conc)] = _adjust_shade(base, shade)

        for row in rows:
            mtp = row["mtp_enabled"]
            tp = row["tp_size"]
            ep = row["ep_size"]
            dp = row["dp_size"]
            mtp_label = "MTP" if mtp else "non-MTP"
            ep_label = f"/EP{ep}" if ep is not None else ""
            dp_label = f"/DP{dp}" if dp is not None else ""
            ep_key = f"-ep{ep}" if ep is not None else ""
            dp_key = f"-dp{dp}" if dp is not None else ""
            key = f"rocm{row['rocm_version']}-tp{tp}-mtp{mtp}{ep_key}{dp_key}-c{row['concurrency']}"
            conc = row["concurrency"]
            color = rocm_tp_conc_colors.get(
                (row["rocm_version"], tp, conc), "#607D8B"
            )
            # MTP = diamond, non-MTP = circle
            point_style = "rectRot" if mtp else "circle"

            for metric_name in metric_charts:
                val = row[metric_name]
                if val is None:
                    continue

                ds = metric_charts[metric_name]["datasets"]
                if key not in ds:
                    ds[key] = {
                        "label": f"ROCm {row['rocm_version']} / TP{tp} / {mtp_label}{ep_label}{dp_label} / c={conc}",
                        "data": [],
                        "borderColor": color,
                        "backgroundColor": color + "33",
                        "pointRadius": 5 if mtp else 4,
                        "pointStyle": point_style,
                        "pointBackgroundColor": [],
                        "tension": 0.1,
                    }

                is_regression = (
                    metric_name,
                    conc,
                    row["build_date"],
                    row["rocm_version"],
                ) in alert_set

                ds[key]["data"].append(
                    {
                        "x": f"{row['build_date'][:4]}-{row['build_date'][4:6]}-{row['build_date'][6:]}",
                        "y": round(val, 2),
                        "tag": row["image_tag"],
                        "sglang": row["sglang_version"],
                    }
                )
                ds[key]["pointBackgroundColor"].append(
                    "#FF0000" if is_regression else color
                )

        # Convert datasets dict to list
        charts_output = {}
        for metric_name, chart_info in metric_charts.items():
            charts_output[metric_name] = {
                "label": chart_info["label"],
                "datasets": list(chart_info["datasets"].values()),
            }

        # Accuracy chart
        acc_datasets: dict = {}
        for row in accuracy_rows:
            mtp = row["mtp_enabled"]
            tp = row["tp_size"]
            ep = row["ep_size"]
            dp = row["dp_size"]
            mtp_label = "MTP" if mtp else "non-MTP"
            ep_label = f"/EP{ep}" if ep is not None else ""
            dp_label = f"/DP{dp}" if dp is not None else ""
            ep_key = f"-ep{ep}" if ep is not None else ""
            dp_key = f"-dp{dp}" if dp is not None else ""
            key = f"rocm{row['rocm_version']}-tp{tp}-mtp{mtp}{ep_key}{dp_key}"
            color = rocm_tp_base.get(
                (row["rocm_version"], tp), "#607D8B"
            )
            point_style = "rectRot" if mtp else "circle"

            if key not in acc_datasets:
                acc_datasets[key] = {
                    "label": f"ROCm {row['rocm_version']} / TP{tp} / {mtp_label}{ep_label}{dp_label}",
                    "data": [],
                    "borderColor": color,
                    "backgroundColor": color + "33",
                    "pointRadius": 5 if mtp else 4,
                    "pointStyle": point_style,
                    "pointBackgroundColor": [],
                    "tension": 0.1,
                }

            is_regression = (
                "accuracy_pct",
                None,
                row["build_date"],
                row["rocm_version"],
            ) in alert_set

            acc_datasets[key]["data"].append(
                {
                    "x": f"{row['build_date'][:4]}-{row['build_date'][4:6]}-{row['build_date'][6:]}",
                    "y": round(row["accuracy_pct"], 2),
                    "tag": row["image_tag"],
                    "sglang": row["sglang_version"],
                }
            )
            acc_datasets[key]["pointBackgroundColor"].append(
                "#FF0000" if is_regression else color
            )

        charts_output["accuracy_pct"] = {
            "label": "GSM8K Accuracy (%)",
            "datasets": list(acc_datasets.values()),
        }

        return JSONResponse(charts_output)
    finally:
        conn.close()


@app.get("/api/runs")
async def get_runs(
    rocm_version: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
    tp_size: Optional[str] = Query(None),
    mtp_enabled: Optional[str] = Query(None),
    ep_size: Optional[str] = Query(None),
    dp_size: Optional[str] = Query(None),
    days: int = Query(30),
    limit: int = Query(100),
):
    """Return run history."""
    conn = _get_conn()
    try:
        where_parts = []
        params: list = []

        if model_name and model_name != "all":
            where_parts.append("br.model_name = ?")
            params.append(model_name)

        if rocm_version and rocm_version != "all":
            where_parts.append("br.rocm_version = ?")
            params.append(rocm_version)

        if tp_size and tp_size != "all":
            where_parts.append("br.tp_size = ?")
            params.append(int(tp_size))

        if mtp_enabled and mtp_enabled != "all":
            where_parts.append("br.mtp_enabled = ?")
            params.append(int(mtp_enabled))

        if ep_size and ep_size != "all":
            if ep_size == "none":
                where_parts.append("br.ep_size IS NULL")
            else:
                where_parts.append("br.ep_size = ?")
                params.append(int(ep_size))

        if dp_size and dp_size != "all":
            if dp_size == "none":
                where_parts.append("br.dp_size IS NULL")
            else:
                where_parts.append("br.dp_size = ?")
                params.append(int(dp_size))

        if days > 0:
            where_parts.append(
                f"br.build_date >= strftime('%Y%m%d', 'now', '-{days} days')"
            )

        where_clause = " AND ".join(where_parts) if where_parts else "1=1"
        params.append(limit)

        query = f"""
            SELECT
                br.id, br.image_tag, br.sglang_version, br.rocm_version,
                br.build_date, br.model_name, br.tp_size, br.mtp_enabled,
                br.ep_size, br.dp_size,
                br.run_timestamp, br.status,
                br.error_message, br.duration_total_sec
            FROM benchmark_runs br
            WHERE {where_clause}
            ORDER BY br.build_date DESC
            LIMIT ?
        """
        runs = conn.execute(query, params).fetchall()

        result = []
        for run in runs:
            run_dict = dict(run)

            # Fetch metrics for this run
            metrics = conn.execute(
                """SELECT * FROM benchmark_metrics WHERE run_id = ?
                   ORDER BY concurrency""",
                (run["id"],),
            ).fetchall()
            run_dict["metrics"] = [dict(m) for m in metrics]

            # Fetch accuracy
            accuracy = conn.execute(
                "SELECT * FROM accuracy_results WHERE run_id = ?",
                (run["id"],),
            ).fetchone()
            if accuracy:
                acc_dict = dict(accuracy)
                acc_dict.pop("raw_output", None)  # Don't send raw output to frontend
                run_dict["accuracy"] = acc_dict
            else:
                run_dict["accuracy"] = None

            # Fetch versions
            versions = conn.execute(
                """SELECT library_name, version, source_type, git_commit
                   FROM version_snapshots WHERE run_id = ?
                   ORDER BY library_name""",
                (run["id"],),
            ).fetchall()
            run_dict["versions"] = [dict(v) for v in versions]

            result.append(run_dict)

        return JSONResponse(result)
    finally:
        conn.close()


@app.get("/api/versions/{run_id}")
async def get_versions(run_id: int):
    """Return version snapshot for a given run."""
    conn = _get_conn()
    try:
        versions = get_version_snapshot(conn, run_id)
        return JSONResponse(versions)
    finally:
        conn.close()


@app.delete("/api/runs/{run_id}")
async def delete_run(run_id: int):
    """Delete a benchmark run and all related data."""
    conn = _get_conn()
    try:
        # Check the run exists
        row = conn.execute(
            "SELECT id, image_tag FROM benchmark_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if not row:
            return JSONResponse({"error": "Run not found"}, status_code=404)

        conn.execute("DELETE FROM version_snapshots WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM regression_alerts WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM accuracy_results WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM benchmark_metrics WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM benchmark_runs WHERE id = ?", (run_id,))
        conn.commit()
        logger.info("Deleted run id=%d (%s)", run_id, row["image_tag"])
        return JSONResponse({"ok": True})
    finally:
        conn.close()


@app.get("/api/version-diff")
async def version_diff_api(run_a: int = Query(...), run_b: int = Query(...)):
    """Compare versions between two runs."""
    conn = _get_conn()
    try:
        from version_diff import compute_version_diff, suggest_root_cause
        diff = compute_version_diff(conn, run_a, run_b)
        suggestions = suggest_root_cause(diff)
        return JSONResponse({"diff": diff, "suggestions": suggestions})
    finally:
        conn.close()


@app.get("/api/alerts")
async def get_alerts(
    rocm_version: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
    tp_size: Optional[str] = Query(None),
    mtp_enabled: Optional[str] = Query(None),
    ep_size: Optional[str] = Query(None),
    dp_size: Optional[str] = Query(None),
    days: int = Query(30),
    acknowledged: Optional[bool] = Query(None),
):
    """Return regression alerts."""
    conn = _get_conn()
    try:
        where_parts = ["1=1"]
        params: list = []

        if model_name and model_name != "all":
            where_parts.append("br.model_name = ?")
            params.append(model_name)

        if rocm_version and rocm_version != "all":
            where_parts.append("br.rocm_version = ?")
            params.append(rocm_version)

        if tp_size and tp_size != "all":
            where_parts.append("br.tp_size = ?")
            params.append(int(tp_size))

        if mtp_enabled and mtp_enabled != "all":
            where_parts.append("br.mtp_enabled = ?")
            params.append(int(mtp_enabled))

        if ep_size and ep_size != "all":
            if ep_size == "none":
                where_parts.append("br.ep_size IS NULL")
            else:
                where_parts.append("br.ep_size = ?")
                params.append(int(ep_size))

        if dp_size and dp_size != "all":
            if dp_size == "none":
                where_parts.append("br.dp_size IS NULL")
            else:
                where_parts.append("br.dp_size = ?")
                params.append(int(dp_size))

        if days > 0:
            where_parts.append(
                f"br.build_date >= strftime('%Y%m%d', 'now', '-{days} days')"
            )

        if acknowledged is not None:
            where_parts.append("ra.acknowledged = ?")
            params.append(1 if acknowledged else 0)

        where_clause = " AND ".join(where_parts)

        query = f"""
            SELECT
                ra.id, ra.run_id, ra.metric_name, ra.concurrency,
                ra.current_value, ra.baseline_value, ra.regression_pct,
                ra.acknowledged, ra.created_at,
                br.image_tag, br.build_date, br.rocm_version, br.sglang_version,
                br.tp_size, br.mtp_enabled, br.ep_size, br.dp_size
            FROM regression_alerts ra
            JOIN benchmark_runs br ON ra.run_id = br.id
            WHERE {where_clause}
            ORDER BY ra.created_at DESC
        """
        alerts = conn.execute(query, params).fetchall()
        return JSONResponse([dict(a) for a in alerts])
    finally:
        conn.close()


@app.get("/api/variant-comparison")
async def variant_comparison(
    build_date: str = Query(...),
    rocm_version: Optional[str] = Query(None),
):
    """Return metrics for all variants of a specific build, for side-by-side comparison."""
    conn = _get_conn()
    try:
        where = "br.build_date = ? AND br.status = 'completed'"
        params: list = [build_date]
        if rocm_version and rocm_version != "all":
            where += " AND br.rocm_version = ?"
            params.append(rocm_version)

        runs = conn.execute(
            f"""
            SELECT br.id, br.model_name, br.image_tag, br.rocm_version,
                   br.tp_size, br.mtp_enabled, br.ep_size, br.dp_size
            FROM benchmark_runs br WHERE {where}
            ORDER BY br.tp_size, br.mtp_enabled, br.ep_size, br.dp_size
            """,
            params,
        ).fetchall()

        result = []
        for run in runs:
            metrics = conn.execute(
                "SELECT * FROM benchmark_metrics WHERE run_id = ? ORDER BY concurrency",
                (run["id"],),
            ).fetchall()
            accuracy = conn.execute(
                "SELECT accuracy_pct FROM accuracy_results WHERE run_id = ?",
                (run["id"],),
            ).fetchone()
            result.append({
                "model_name": run["model_name"],
                "image_tag": run["image_tag"],
                "rocm_version": run["rocm_version"],
                "tp_size": run["tp_size"],
                "mtp_enabled": run["mtp_enabled"],
                "ep_size": run["ep_size"],
                "dp_size": run["dp_size"],
                "metrics": [dict(m) for m in metrics],
                "accuracy_pct": accuracy["accuracy_pct"] if accuracy else None,
            })
        return JSONResponse(result)
    finally:
        conn.close()


@app.get("/api/health")
async def health():
    """System health check."""
    import shutil

    disk = shutil.disk_usage("/")
    conn = _get_conn()
    try:
        total_runs = conn.execute("SELECT COUNT(*) as cnt FROM benchmark_runs").fetchone()["cnt"]
        completed_runs = conn.execute(
            "SELECT COUNT(*) as cnt FROM benchmark_runs WHERE status='completed'"
        ).fetchone()["cnt"]
        failed_runs = conn.execute(
            "SELECT COUNT(*) as cnt FROM benchmark_runs WHERE status='failed'"
        ).fetchone()["cnt"]
        alert_count = conn.execute(
            "SELECT COUNT(*) as cnt FROM regression_alerts WHERE acknowledged=0"
        ).fetchone()["cnt"]

        return JSONResponse(
            {
                "status": "ok",
                "disk_free_gb": round(disk.free / (1024 ** 3), 1),
                "total_runs": total_runs,
                "completed_runs": completed_runs,
                "failed_runs": failed_runs,
                "unacknowledged_alerts": alert_count,
            }
        )
    finally:
        conn.close()


if __name__ == "__main__":
    import uvicorn

    init_db()
    uvicorn.run(app, host="0.0.0.0", port=DASHBOARD_PORT)
