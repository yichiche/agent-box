"""Regression detection using moving-average baselines."""

import logging
import sqlite3
from typing import Optional

from config import (
    MONITORED_METRICS,
    REGRESSION_THRESHOLD_PCT,
    REGRESSION_WINDOW,
    ACCURACY_REGRESSION_THRESHOLD,
)
from collector import get_connection

logger = logging.getLogger(__name__)


def detect_regressions(
    conn: sqlite3.Connection,
    run_id: int,
    threshold_pct: Optional[float] = None,
) -> list[dict]:
    """Detect regressions for a completed run by comparing to the moving average
    of the previous N completed runs with the same rocm_version.

    Args:
        threshold_pct: Override for REGRESSION_THRESHOLD_PCT. If None, uses config default.

    Returns a list of regression alert dicts.
    """
    # Get current run info
    run = conn.execute(
        "SELECT id, rocm_version, build_date, model_name, tp_size, mtp_enabled, "
        "ep_size, dp_size, is_profile_run "
        "FROM benchmark_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    if run is None:
        logger.warning("Run %d not found", run_id)
        return []

    # Never run regression detection on profiling runs
    if run["is_profile_run"]:
        logger.info("Skipping regression detection for profile run %d", run_id)
        return []

    rocm_version = run["rocm_version"]
    model_name = run["model_name"]
    tp_size = run["tp_size"]
    mtp_enabled = run["mtp_enabled"]
    ep_size = run["ep_size"]
    dp_size = run["dp_size"]
    alerts = []

    # Get concurrencies for this run
    concurrencies = [
        row["concurrency"]
        for row in conn.execute(
            "SELECT DISTINCT concurrency FROM benchmark_metrics WHERE run_id = ?",
            (run_id,),
        ).fetchall()
    ]

    effective_threshold = threshold_pct if threshold_pct is not None else REGRESSION_THRESHOLD_PCT

    for metric_name, direction in MONITORED_METRICS:
        for conc in concurrencies:
            alert = _check_metric_regression(
                conn, run_id, rocm_version, model_name, tp_size, mtp_enabled,
                metric_name, conc, direction, effective_threshold,
                ep_size=ep_size, dp_size=dp_size,
            )
            if alert:
                alerts.append(alert)

    # Check accuracy regression
    accuracy_alert = _check_accuracy_regression(
        conn, run_id, rocm_version, model_name, tp_size, mtp_enabled,
        ep_size=ep_size, dp_size=dp_size,
    )
    if accuracy_alert:
        alerts.append(accuracy_alert)

    # Insert alerts into DB
    for alert in alerts:
        conn.execute(
            """INSERT INTO regression_alerts
               (run_id, metric_name, concurrency, current_value, baseline_value, regression_pct)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                alert["run_id"],
                alert["metric_name"],
                alert.get("concurrency"),
                alert["current_value"],
                alert["baseline_value"],
                alert["regression_pct"],
            ),
        )
    conn.commit()

    if alerts:
        logger.warning("Detected %d regression(s) for run %d", len(alerts), run_id)
    else:
        logger.info("No regressions detected for run %d", run_id)

    return alerts


def _check_metric_regression(
    conn: sqlite3.Connection,
    run_id: int,
    rocm_version: str,
    model_name: str,
    tp_size: int,
    mtp_enabled: int,
    metric_name: str,
    concurrency: int,
    direction: str,
    threshold_pct: float,
    ep_size: Optional[int] = None,
    dp_size: Optional[int] = None,
) -> Optional[dict]:
    """Check if a metric at a specific concurrency has regressed."""
    # Get current value
    current_row = conn.execute(
        f"SELECT {metric_name} FROM benchmark_metrics WHERE run_id = ? AND concurrency = ?",
        (run_id, concurrency),
    ).fetchone()
    if current_row is None or current_row[metric_name] is None:
        return None
    current_value = current_row[metric_name]

    # Get previous N completed runs (same rocm_version, model, tp_size, mtp_enabled)
    build_date = conn.execute(
        "SELECT build_date FROM benchmark_runs WHERE id = ?", (run_id,)
    ).fetchone()["build_date"]

    previous_values = conn.execute(
        f"""SELECT bm.{metric_name}
            FROM benchmark_metrics bm
            JOIN benchmark_runs br ON bm.run_id = br.id
            WHERE br.rocm_version = ?
              AND br.model_name = ?
              AND br.tp_size = ?
              AND br.mtp_enabled = ?
              AND br.ep_size IS ?
              AND br.dp_size IS ?
              AND br.status = 'completed'
              AND br.is_profile_run = 0
              AND br.build_date < ?
              AND bm.concurrency = ?
              AND bm.{metric_name} IS NOT NULL
            ORDER BY br.build_date DESC
            LIMIT ?""",
        (rocm_version, model_name, tp_size, mtp_enabled,
         ep_size, dp_size, build_date, concurrency, REGRESSION_WINDOW),
    ).fetchall()

    if len(previous_values) < 2:
        # Not enough data for a meaningful baseline
        return None

    baseline = sum(row[0] for row in previous_values) / len(previous_values)
    if baseline == 0:
        return None

    if direction == "higher_better":
        # Regression if current < baseline * (1 - threshold/100)
        regression_pct = ((baseline - current_value) / baseline) * 100
        if current_value < baseline * (1 - threshold_pct / 100):
            return {
                "run_id": run_id,
                "metric_name": metric_name,
                "concurrency": concurrency,
                "current_value": current_value,
                "baseline_value": baseline,
                "regression_pct": round(regression_pct, 2),
            }
    else:
        # lower_better: regression if current > baseline * (1 + threshold/100)
        regression_pct = ((current_value - baseline) / baseline) * 100
        if current_value > baseline * (1 + threshold_pct / 100):
            return {
                "run_id": run_id,
                "metric_name": metric_name,
                "concurrency": concurrency,
                "current_value": current_value,
                "baseline_value": baseline,
                "regression_pct": round(regression_pct, 2),
            }

    return None


def _check_accuracy_regression(
    conn: sqlite3.Connection,
    run_id: int,
    rocm_version: str,
    model_name: str,
    tp_size: int,
    mtp_enabled: int,
    ep_size: Optional[int] = None,
    dp_size: Optional[int] = None,
) -> Optional[dict]:
    """Check if accuracy has regressed beyond the threshold."""
    current_row = conn.execute(
        "SELECT accuracy_pct FROM accuracy_results WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if current_row is None or current_row["accuracy_pct"] is None:
        return None
    current_value = current_row["accuracy_pct"]

    build_date = conn.execute(
        "SELECT build_date FROM benchmark_runs WHERE id = ?", (run_id,)
    ).fetchone()["build_date"]

    previous_values = conn.execute(
        """SELECT ar.accuracy_pct
           FROM accuracy_results ar
           JOIN benchmark_runs br ON ar.run_id = br.id
           WHERE br.rocm_version = ?
             AND br.model_name = ?
             AND br.tp_size = ?
             AND br.mtp_enabled = ?
             AND br.ep_size IS ?
             AND br.dp_size IS ?
             AND br.status = 'completed'
             AND br.is_profile_run = 0
             AND br.build_date < ?
             AND ar.accuracy_pct IS NOT NULL
           ORDER BY br.build_date DESC
           LIMIT ?""",
        (rocm_version, model_name, tp_size, mtp_enabled,
         ep_size, dp_size, build_date, REGRESSION_WINDOW),
    ).fetchall()

    if len(previous_values) < 2:
        return None

    baseline = sum(row[0] for row in previous_values) / len(previous_values)

    if current_value < baseline - ACCURACY_REGRESSION_THRESHOLD:
        regression_pct = round(baseline - current_value, 2)
        return {
            "run_id": run_id,
            "metric_name": "accuracy_pct",
            "concurrency": None,
            "current_value": current_value,
            "baseline_value": baseline,
            "regression_pct": regression_pct,
        }

    return None


def _check_metric_change(
    conn: sqlite3.Connection,
    run_id: int,
    rocm_version: str,
    model_name: str,
    tp_size: int,
    mtp_enabled: int,
    metric_name: str,
    concurrency: int,
    direction: str,
    improve_threshold_pct: float,
    regress_threshold_pct: float,
    ep_size: Optional[int] = None,
    dp_size: Optional[int] = None,
    window: int = REGRESSION_WINDOW,
) -> Optional[dict]:
    """Check if a metric changed beyond separate improve/regress thresholds."""
    current_row = conn.execute(
        f"SELECT {metric_name} FROM benchmark_metrics WHERE run_id = ? AND concurrency = ?",
        (run_id, concurrency),
    ).fetchone()
    if current_row is None or current_row[metric_name] is None:
        return None
    current_value = current_row[metric_name]

    build_date = conn.execute(
        "SELECT build_date FROM benchmark_runs WHERE id = ?", (run_id,)
    ).fetchone()["build_date"]

    previous_values = conn.execute(
        f"""SELECT bm.{metric_name}
            FROM benchmark_metrics bm
            JOIN benchmark_runs br ON bm.run_id = br.id
            WHERE br.rocm_version = ?
              AND br.model_name = ?
              AND br.tp_size = ?
              AND br.mtp_enabled = ?
              AND br.ep_size IS ?
              AND br.dp_size IS ?
              AND br.status = 'completed'
              AND br.is_profile_run = 0
              AND br.build_date < ?
              AND bm.concurrency = ?
              AND bm.{metric_name} IS NOT NULL
            ORDER BY br.build_date DESC
            LIMIT ?""",
        (rocm_version, model_name, tp_size, mtp_enabled,
         ep_size, dp_size, build_date, concurrency, window),
    ).fetchall()

    min_required = 1 if window == 1 else 2
    if len(previous_values) < min_required:
        return None

    baseline = sum(row[0] for row in previous_values) / len(previous_values)
    if baseline == 0:
        return None

    pct_change = ((current_value - baseline) / baseline) * 100

    if direction == "higher_better":
        is_improved = pct_change > 0
    else:
        is_improved = pct_change < 0

    label = "improved" if is_improved else "regressed"
    active_threshold = improve_threshold_pct if is_improved else regress_threshold_pct

    if abs(pct_change) > active_threshold:
        return {
            "run_id": run_id,
            "metric_name": metric_name,
            "concurrency": concurrency,
            "current_value": current_value,
            "baseline_value": baseline,
            "change_pct": round(pct_change, 2),
            "direction": label,
        }

    return None


def check_regressions_dynamic(
    conn: sqlite3.Connection,
    thresholds: Optional[dict] = None,
    window: Optional[int] = None,
    rocm_version: Optional[str] = None,
    model_name: Optional[str] = None,
    concurrency: Optional[int] = None,
    tp_size: Optional[int] = None,
    mtp_enabled: Optional[int] = None,
    ep_size: Optional[int] = None,
    dp_size: Optional[int] = None,
    days: int = 30,
) -> list[dict]:
    """Re-evaluate metric changes on-the-fly with per-metric, per-direction thresholds.

    Args:
        thresholds: Dict mapping metric_name -> {"improved": float, "regressed": float}.
                    If None, uses REGRESSION_THRESHOLD_PCT for all.
        window: Number of previous runs for moving-average baseline.
                If None, uses REGRESSION_WINDOW from config.

    Detects both improvements and regressions. Does NOT write to the database.
    """
    where_parts = ["br.status = 'completed'", "br.is_profile_run = 0"]
    params: list = []

    if rocm_version and rocm_version != "all":
        where_parts.append("br.rocm_version = ?")
        params.append(rocm_version)
    if model_name and model_name != "all":
        where_parts.append("br.model_name = ?")
        params.append(model_name)
    if tp_size is not None:
        where_parts.append("br.tp_size = ?")
        params.append(tp_size)
    if mtp_enabled is not None:
        where_parts.append("br.mtp_enabled = ?")
        params.append(mtp_enabled)
    if ep_size is not None:
        where_parts.append("br.ep_size IS ?")
        params.append(ep_size)
    if dp_size is not None:
        where_parts.append("br.dp_size IS ?")
        params.append(dp_size)

    where_parts.append(
        "br.build_date >= strftime('%Y%m%d', 'now', ?)"
    )
    params.append(f"-{days} days")

    where_sql = " AND ".join(where_parts)

    runs = conn.execute(
        f"""SELECT br.id, br.image_tag, br.build_date, br.rocm_version,
                   br.model_name, br.tp_size, br.mtp_enabled, br.ep_size, br.dp_size
            FROM benchmark_runs br
            WHERE {where_sql}
            ORDER BY br.build_date""",
        params,
    ).fetchall()

    effective_window = window if window is not None else REGRESSION_WINDOW
    default_thresh = REGRESSION_THRESHOLD_PCT
    alerts = []
    for run in runs:
        for metric_name, direction in MONITORED_METRICS:
            if thresholds and metric_name in thresholds:
                mt = thresholds[metric_name]
                improve_thresh = mt.get("improved", default_thresh)
                regress_thresh = mt.get("regressed", default_thresh)
            else:
                improve_thresh = default_thresh
                regress_thresh = default_thresh

            if concurrency is not None:
                concurrencies = [{"concurrency": concurrency}]
            else:
                concurrencies = conn.execute(
                    "SELECT DISTINCT concurrency FROM benchmark_metrics WHERE run_id = ?",
                    (run["id"],),
                ).fetchall()
            for conc_row in concurrencies:
                alert = _check_metric_change(
                    conn, run["id"], run["rocm_version"], run["model_name"],
                    run["tp_size"], run["mtp_enabled"],
                    metric_name, conc_row["concurrency"], direction,
                    improve_thresh, regress_thresh,
                    ep_size=run["ep_size"], dp_size=run["dp_size"],
                    window=effective_window,
                )
                if alert:
                    alert["image_tag"] = run["image_tag"]
                    alert["build_date"] = run["build_date"]
                    alert["rocm_version"] = run["rocm_version"]
                    alert["model_name"] = run["model_name"]
                    alert["tp_size"] = run["tp_size"]
                    alert["mtp_enabled"] = run["mtp_enabled"]
                    alert["ep_size"] = run["ep_size"]
                    alert["dp_size"] = run["dp_size"]
                    alerts.append(alert)

    return alerts
