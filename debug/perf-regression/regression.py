"""Regression detection using moving-average baselines."""

import logging
import sqlite3
from typing import Optional

from config import (
    MONITORED_METRICS,
    METRIC_THRESHOLDS,
    REGRESSION_THRESHOLD_PCT,
    REGRESSION_WINDOW,
    ACCURACY_REGRESSION_THRESHOLD,
)
from collector import get_connection

logger = logging.getLogger(__name__)


def detect_regressions(
    conn: sqlite3.Connection,
    run_id: int,
) -> list[dict]:
    """Detect regressions for a completed run by comparing to the moving average
    of the previous N completed runs with the same rocm_version.

    Returns a list of regression alert dicts.
    """
    # Get current run info
    run = conn.execute(
        "SELECT id, rocm_version, build_date, model_name, tp_size, mtp_enabled, "
        "ep_size, dp_size "
        "FROM benchmark_runs WHERE id = ?",
        (run_id,),
    ).fetchone()
    if run is None:
        logger.warning("Run %d not found", run_id)
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

    for metric_name, direction in MONITORED_METRICS:
        threshold = METRIC_THRESHOLDS.get(metric_name, REGRESSION_THRESHOLD_PCT)

        for conc in concurrencies:
            alert = _check_metric_regression(
                conn, run_id, rocm_version, model_name, tp_size, mtp_enabled,
                metric_name, conc, direction, threshold,
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
