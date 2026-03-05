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
from collector import (
    get_connection, init_db, get_version_snapshot, upsert_target, get_targets, delete_target,
    get_profile_connection, init_profile_db, resolve_result_dir,
)

logger = logging.getLogger(__name__)
app = FastAPI(title="SGLang ROCm Perf Regression Dashboard")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.on_event("startup")
def startup():
    init_db()
    init_profile_db()
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
        # Build WHERE clause (exclude profile runs from charts)
        where_parts = ["br.status = 'completed'", "br.is_profile_run = 0"]
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

        # Fetch target baselines for chart annotations
        target_tp = int(tp_size) if tp_size and tp_size != "all" else None
        target_mtp = int(mtp_enabled) if mtp_enabled and mtp_enabled != "all" else None
        target_conc = int(concurrency) if concurrency and concurrency != "all" else None
        target_ep = ep_size if ep_size else None
        target_dp = dp_size if dp_size else None
        targets = get_targets(
            conn,
            model_name=model_name,
            tp_size=target_tp,
            mtp_enabled=target_mtp,
            ep_size=target_ep,
            dp_size=target_dp,
            concurrency=target_conc,
        )
        target_lines: dict = {}
        for t in targets:
            for metric in ["output_throughput", "total_throughput", "median_ttft_ms",
                           "median_itl_ms", "median_e2e_latency_ms"]:
                val = t.get(metric)
                if val is not None:
                    target_lines.setdefault(metric, []).append({
                        "value": val,
                        "concurrency": t["concurrency"],
                        "platform": t["platform"],
                        "tp_size": t["tp_size"],
                        "mtp_enabled": t["mtp_enabled"],
                    })
        charts_output["targets"] = target_lines

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
        where_parts = ["br.is_profile_run = 0"]
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


@app.get("/api/runs/{run_id}/server-log")
async def get_server_log(run_id: int, tail: int = Query(200)):
    """Return the last N lines of the server.log for a failed run."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT result_dir, status FROM benchmark_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if not row:
            return JSONResponse({"error": "Run not found"}, status_code=404)
        rd = resolve_result_dir(row["result_dir"])
        if not rd:
            return JSONResponse({"log": None, "reason": "No result directory recorded"})
        log_path = rd / "server.log"
        try:
            if not log_path.exists():
                return JSONResponse({"log": None, "reason": f"server.log not found in {result_dir}"})
            lines = log_path.read_text(errors="replace").splitlines()
            tail_lines = lines[-tail:] if len(lines) > tail else lines
            return JSONResponse({"log": "\n".join(tail_lines)})
        except OSError:
            return JSONResponse({"log": None, "reason": f"server.log not accessible in {result_dir}"})
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
        where_parts = ["br.is_profile_run = 0"]
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
        where = "br.build_date = ? AND br.status = 'completed' AND br.is_profile_run = 0"
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


@app.get("/api/profile-runs")
async def get_profile_runs(
    rocm_version: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
    tp_size: Optional[str] = Query(None),
    ep_size: Optional[str] = Query(None),
    dp_size: Optional[str] = Query(None),
    days: int = Query(30),
    limit: int = Query(100),
):
    """Return profile run history from the profile database."""
    conn = get_profile_connection()
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
        params.append(limit)

        query = f"""
            SELECT
                br.id, br.image_tag, br.sglang_version, br.rocm_version,
                br.build_date, br.model_name, br.tp_size, br.mtp_enabled,
                br.ep_size, br.dp_size,
                br.run_timestamp, br.status,
                br.error_message, br.duration_total_sec, br.result_dir
            FROM benchmark_runs br
            WHERE {where_clause}
            ORDER BY br.build_date DESC
            LIMIT ?
        """
        runs = conn.execute(query, params).fetchall()

        # Batch-query profile_scores to check which runs have scores in DB
        run_ids = [run["id"] for run in runs]
        runs_with_scores: set = set()
        if run_ids:
            placeholders = ",".join("?" * len(run_ids))
            score_rows = conn.execute(
                f"SELECT DISTINCT run_id FROM profile_scores WHERE run_id IN ({placeholders})",
                run_ids,
            ).fetchall()
            runs_with_scores = {r["run_id"] for r in score_rows}

        result = []
        for run in runs:
            run_dict = dict(run)
            # Check DB first, fall back to filesystem
            if run["id"] in runs_with_scores:
                run_dict["has_evaluation_summary"] = True
            else:
                rd = resolve_result_dir(run_dict.get("result_dir"))
                summary_path = rd / "trace_analysis" / "evaluation_summary.csv" if rd else None
                try:
                    run_dict["has_evaluation_summary"] = bool(summary_path and summary_path.exists())
                except OSError:
                    run_dict["has_evaluation_summary"] = False
            diff_path = resolve_result_dir(run_dict.get("result_dir"))
            diff_path = diff_path / "trace_analysis" / "kernel_diff.json" if diff_path else None
            try:
                run_dict["has_kernel_diff"] = bool(diff_path and diff_path.exists())
            except OSError:
                run_dict["has_kernel_diff"] = False
            run_dict.pop("result_dir", None)
            result.append(run_dict)

        return JSONResponse(result)
    finally:
        conn.close()


@app.get("/api/profile-runs/{run_id}/evaluation-summary")
async def get_evaluation_summary(run_id: int):
    """Return the evaluation_summary.csv content for a profile run."""
    conn = get_profile_connection()
    try:
        row = conn.execute(
            "SELECT result_dir FROM benchmark_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if not row:
            return JSONResponse({"error": "Run not found"}, status_code=404)

        rd = resolve_result_dir(row["result_dir"])
        if not rd:
            return JSONResponse({"csv": None, "reason": "No result directory recorded"})
        summary_path = rd / "trace_analysis" / "evaluation_summary.csv"
        try:
            if not summary_path.exists():
                return JSONResponse({"csv": None, "reason": "evaluation_summary.csv not found"})
            return JSONResponse({"csv": summary_path.read_text(errors="replace")})
        except OSError:
            return JSONResponse({"csv": None, "reason": "evaluation_summary.csv not accessible"})
    finally:
        conn.close()


@app.get("/api/profile-chart-data")
async def profile_chart_data(
    rocm_version: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
    tp_size: Optional[str] = Query(None),
    days: int = Query(30),
):
    """Return Chart.js data for profile structural scores (S1-S4) over time."""
    conn = get_profile_connection()
    try:
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
        if days > 0:
            where_parts.append(
                f"br.build_date >= strftime('%Y%m%d', 'now', '-{days} days')"
            )

        where_clause = " AND ".join(where_parts)
        runs = conn.execute(
            f"""
            SELECT br.id, br.build_date, br.image_tag, br.rocm_version,
                   br.sglang_version, br.tp_size
            FROM benchmark_runs br
            WHERE {where_clause}
            ORDER BY br.build_date, br.rocm_version, br.tp_size
            """,
            params,
        ).fetchall()

        STRUCTURAL_METRICS = [
            "S1 Prefill Ordering",
            "S2 Architecture Sig",
            "S3 Round Consistency",
            "S4 Type Sequence",
        ]
        TARGET_METRICS = set(STRUCTURAL_METRICS)

        # Batch-query all profile_scores for these runs in one go
        run_ids = [r["id"] for r in runs]
        scores_by_run: dict[int, dict[str, float]] = {}
        if run_ids:
            placeholders = ",".join("?" * len(run_ids))
            score_rows = conn.execute(
                f"""SELECT run_id, metric, score FROM profile_scores
                    WHERE run_id IN ({placeholders})
                    AND section = 'structural'""",
                run_ids,
            ).fetchall()
            for sr in score_rows:
                if sr["metric"] in TARGET_METRICS:
                    scores_by_run.setdefault(sr["run_id"], {})[sr["metric"]] = sr["score"]

        datasets: dict = {}

        for run in runs:
            scores = scores_by_run.get(run["id"])
            if not scores:
                continue

            date_str = run["build_date"]
            x_val = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            rocm = run["rocm_version"]
            tp = run["tp_size"]
            variant_key = f"rocm{rocm}-tp{tp}"
            variant_label = f"ROCm {rocm} / TP{tp}"

            for metric in STRUCTURAL_METRICS:
                if metric not in scores:
                    continue
                key = f"{variant_key}-{metric}"
                if key not in datasets:
                    datasets[key] = {
                        "label": f"{variant_label} / {metric}",
                        "variant": variant_key,
                        "metric": metric,
                        "data": [],
                    }
                datasets[key]["data"].append({
                    "x": x_val,
                    "y": round(scores[metric], 2),
                    "tag": run["image_tag"],
                    "sglang": run["sglang_version"],
                })

        return JSONResponse({
            "label": "Profile Structural Scores",
            "datasets": list(datasets.values()),
        })
    finally:
        conn.close()


@app.get("/api/profile-runs/{run_id}/kernel-diff")
async def get_kernel_diff(run_id: int):
    """Return the pre-computed kernel_diff.json for a profile run."""
    import json as _json

    conn = get_profile_connection()
    try:
        row = conn.execute(
            "SELECT result_dir FROM benchmark_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if not row:
            return JSONResponse({"error": "Run not found"}, status_code=404)

        rd = resolve_result_dir(row["result_dir"])
        if not rd:
            return JSONResponse({"diff": None, "reason": "No result directory recorded"})

        diff_path = rd / "trace_analysis" / "kernel_diff.json"
        try:
            if not diff_path.exists():
                return JSONResponse({"diff": None, "reason": "No kernel diff available (no previous run to compare)"})
            diff = _json.loads(diff_path.read_text(errors="replace"))
            return JSONResponse({"diff": diff})
        except OSError:
            return JSONResponse({"diff": None, "reason": "Kernel diff not accessible"})
    finally:
        conn.close()


@app.get("/api/profile-diff")
async def profile_diff_on_demand(
    run_a: int = Query(..., description="Profile run ID for baseline (old)"),
    run_b: int = Query(..., description="Profile run ID for candidate (new)"),
):
    """On-demand kernel diff between any two profile runs."""
    from trace_diff import compare_trace_files

    conn = get_profile_connection()
    try:
        row_a = conn.execute(
            "SELECT id, image_tag, result_dir FROM benchmark_runs WHERE id = ?",
            (run_a,),
        ).fetchone()
        row_b = conn.execute(
            "SELECT id, image_tag, result_dir FROM benchmark_runs WHERE id = ?",
            (run_b,),
        ).fetchone()

        if not row_a or not row_b:
            return JSONResponse({"error": "One or both runs not found"}, status_code=404)

        rd_a = resolve_result_dir(row_a["result_dir"])
        rd_b = resolve_result_dir(row_b["result_dir"])
        if not rd_a or not rd_b:
            return JSONResponse({"error": "Missing result directory for one or both runs"}, status_code=400)

        xlsx_a = str(rd_a / "trace_analysis" / "profile.csv.xlsx")
        xlsx_b = str(rd_b / "trace_analysis" / "profile.csv.xlsx")

        diff = compare_trace_files(xlsx_a, xlsx_b)
        if diff is None:
            return JSONResponse({
                "diff": None,
                "reason": "Comparison failed (missing xlsx files or parse error)",
            })

        diff["tag_a"] = row_a["image_tag"]
        diff["tag_b"] = row_b["image_tag"]
        diff["run_id_a"] = run_a
        diff["run_id_b"] = run_b
        return JSONResponse({"diff": diff})
    finally:
        conn.close()


@app.delete("/api/profile-runs/{run_id}")
async def delete_profile_run(run_id: int):
    """Delete a profile run from the profile database."""
    conn = get_profile_connection()
    try:
        row = conn.execute(
            "SELECT id, image_tag FROM benchmark_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if not row:
            return JSONResponse({"error": "Profile run not found"}, status_code=404)

        conn.execute("DELETE FROM version_snapshots WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM benchmark_runs WHERE id = ?", (run_id,))
        conn.commit()
        logger.info("Deleted profile run id=%d (%s)", run_id, row["image_tag"])
        return JSONResponse({"ok": True})
    finally:
        conn.close()


@app.get("/api/targets")
async def get_targets_api(
    model_name: Optional[str] = Query(None),
    tp_size: Optional[str] = Query(None),
    mtp_enabled: Optional[str] = Query(None),
    ep_size: Optional[str] = Query(None),
    dp_size: Optional[str] = Query(None),
    concurrency: Optional[str] = Query(None),
):
    """Return target baselines."""
    conn = _get_conn()
    try:
        targets = get_targets(
            conn,
            model_name=model_name,
            tp_size=int(tp_size) if tp_size and tp_size != "all" else None,
            mtp_enabled=int(mtp_enabled) if mtp_enabled and mtp_enabled != "all" else None,
            ep_size=ep_size,
            dp_size=dp_size,
            concurrency=int(concurrency) if concurrency and concurrency != "all" else None,
        )
        return JSONResponse(targets)
    finally:
        conn.close()


@app.post("/api/targets")
async def create_target(request: Request):
    """Create or update a target baseline."""
    body = await request.json()
    conn = _get_conn()
    try:
        target_id = upsert_target(
            conn,
            platform=body.get("platform", "B200"),
            model_name=body["model_name"],
            tp_size=int(body["tp_size"]),
            mtp_enabled=bool(int(body["mtp_enabled"])),
            ep_size=int(body["ep_size"]) if body.get("ep_size") not in (None, "", "none") else None,
            dp_size=int(body["dp_size"]) if body.get("dp_size") not in (None, "", "none") else None,
            concurrency=int(body["concurrency"]),
            image_tag=body.get("image_tag") or None,
            metrics={
                "output_throughput": float(body["output_throughput"]) if body.get("output_throughput") else None,
                "total_throughput": float(body["total_throughput"]) if body.get("total_throughput") else None,
                "median_ttft_ms": float(body["median_ttft_ms"]) if body.get("median_ttft_ms") else None,
                "median_itl_ms": float(body["median_itl_ms"]) if body.get("median_itl_ms") else None,
                "median_e2e_latency_ms": float(body["median_e2e_latency_ms"]) if body.get("median_e2e_latency_ms") else None,
            },
        )
        logger.info("Upserted target baseline id=%d platform=%s", target_id, body.get("platform", "B200"))
        return JSONResponse({"ok": True, "id": target_id})
    except (KeyError, ValueError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    finally:
        conn.close()


@app.delete("/api/targets/{target_id}")
async def delete_target_api(target_id: int):
    """Delete a target baseline."""
    conn = _get_conn()
    try:
        if delete_target(conn, target_id):
            logger.info("Deleted target baseline id=%d", target_id)
            return JSONResponse({"ok": True})
        return JSONResponse({"error": "Target not found"}, status_code=404)
    finally:
        conn.close()


@app.get("/api/regression-check")
async def regression_check(
    tp_improve: float = Query(2.0, description="Total throughput improvement threshold %"),
    tp_regress: float = Query(2.0, description="Total throughput regression threshold %"),
    e2e_improve: float = Query(2.0, description="E2E latency improvement threshold %"),
    e2e_regress: float = Query(2.0, description="E2E latency regression threshold %"),
    window: int = Query(5, description="Number of previous runs for moving-average baseline"),
    rocm_version: Optional[str] = Query(None),
    model_name: Optional[str] = Query(None),
    concurrency: Optional[str] = Query(None),
    tp_size: Optional[str] = Query(None),
    mtp_enabled: Optional[str] = Query(None),
    ep_size: Optional[str] = Query(None),
    dp_size: Optional[str] = Query(None),
    days: int = Query(30),
):
    """Re-evaluate regressions on-the-fly with per-metric, per-direction thresholds (read-only)."""
    from regression import check_regressions_dynamic

    thresholds = {
        "total_throughput": {"improved": tp_improve, "regressed": tp_regress},
        "median_e2e_latency_ms": {"improved": e2e_improve, "regressed": e2e_regress},
    }

    conn = _get_conn()
    try:
        alerts = check_regressions_dynamic(
            conn,
            thresholds=thresholds,
            window=window,
            rocm_version=rocm_version if rocm_version != "all" else None,
            model_name=model_name if model_name != "all" else None,
            concurrency=int(concurrency) if concurrency and concurrency != "all" else None,
            tp_size=int(tp_size) if tp_size and tp_size != "all" else None,
            mtp_enabled=int(mtp_enabled) if mtp_enabled and mtp_enabled != "all" else None,
            ep_size=int(ep_size) if ep_size and ep_size not in ("all", "none") else None,
            dp_size=int(dp_size) if dp_size and dp_size not in ("all", "none") else None,
            days=days,
        )
        return JSONResponse({
            "thresholds": thresholds,
            "count": len(alerts),
            "alerts": alerts,
        })
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
    init_profile_db()
    uvicorn.run(app, host="0.0.0.0", port=DASHBOARD_PORT)
