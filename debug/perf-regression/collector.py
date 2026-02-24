"""SQLite schema management and benchmark data ingestion."""

import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import config as _cfg
from config import DB_PATH

logger = logging.getLogger(__name__)

# ── Schema ─────────────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS benchmark_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_tag TEXT NOT NULL,
    sglang_version TEXT,
    rocm_version TEXT,
    build_date TEXT,
    model_name TEXT NOT NULL,
    tp_size INTEGER NOT NULL DEFAULT 8,
    mtp_enabled INTEGER NOT NULL DEFAULT 1,
    run_timestamp TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    error_message TEXT,
    result_dir TEXT,
    duration_total_sec REAL
);

CREATE TABLE IF NOT EXISTS benchmark_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES benchmark_runs(id) ON DELETE CASCADE,
    concurrency INTEGER NOT NULL,
    completed_requests INTEGER,
    benchmark_duration_s REAL,
    request_throughput REAL,
    input_throughput REAL,
    output_throughput REAL,
    total_throughput REAL,
    mean_e2e_latency_ms REAL,
    median_e2e_latency_ms REAL,
    p90_e2e_latency_ms REAL,
    p99_e2e_latency_ms REAL,
    mean_ttft_ms REAL,
    median_ttft_ms REAL,
    p99_ttft_ms REAL,
    mean_itl_ms REAL,
    median_itl_ms REAL,
    p95_itl_ms REAL,
    p99_itl_ms REAL,
    mean_tpot_ms REAL,
    median_tpot_ms REAL,
    UNIQUE(run_id, concurrency)
);

CREATE TABLE IF NOT EXISTS accuracy_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES benchmark_runs(id) ON DELETE CASCADE,
    num_questions INTEGER,
    num_correct INTEGER,
    accuracy_pct REAL,
    raw_output TEXT,
    UNIQUE(run_id)
);

CREATE TABLE IF NOT EXISTS regression_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES benchmark_runs(id) ON DELETE CASCADE,
    metric_name TEXT NOT NULL,
    concurrency INTEGER,
    current_value REAL,
    baseline_value REAL,
    regression_pct REAL,
    acknowledged INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS version_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES benchmark_runs(id) ON DELETE CASCADE,
    library_name TEXT NOT NULL,
    version TEXT,
    source_type TEXT NOT NULL DEFAULT 'pip',
    git_commit TEXT,
    git_commit_date TEXT,
    git_commit_subject TEXT,
    UNIQUE(run_id, library_name)
);

CREATE INDEX IF NOT EXISTS idx_runs_rocm_status ON benchmark_runs(rocm_version, status);
CREATE INDEX IF NOT EXISTS idx_runs_build_date ON benchmark_runs(build_date);
CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON benchmark_metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_accuracy_run_id ON accuracy_results(run_id);
CREATE INDEX IF NOT EXISTS idx_alerts_run_id ON regression_alerts(run_id);
CREATE INDEX IF NOT EXISTS idx_version_snapshots_run_id ON version_snapshots(run_id);
"""


def get_connection() -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode and foreign keys."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


def _migrate_add_tp_mtp_columns(conn: sqlite3.Connection) -> None:
    """Add tp_size and mtp_enabled columns if they don't exist (schema migration)."""
    cols = {
        row[1] for row in conn.execute("PRAGMA table_info(benchmark_runs)").fetchall()
    }
    if "tp_size" not in cols:
        conn.execute("ALTER TABLE benchmark_runs ADD COLUMN tp_size INTEGER NOT NULL DEFAULT 8")
        logger.info("Migrated: added tp_size column to benchmark_runs")
    if "mtp_enabled" not in cols:
        conn.execute("ALTER TABLE benchmark_runs ADD COLUMN mtp_enabled INTEGER NOT NULL DEFAULT 1")
        logger.info("Migrated: added mtp_enabled column to benchmark_runs")
    conn.commit()


def _migrate_drop_unique_constraint(conn: sqlite3.Connection) -> None:
    """Remove UNIQUE(image_tag, model_name, tp_size, mtp_enabled) from benchmark_runs.

    This allows multiple runs for the same image+model+tp+mtp combination,
    enabling future schema extensions without unique-key conflicts.
    Deduplication is handled in application code (is_already_benchmarked).

    Idempotent: skips if the table already lacks the UNIQUE constraint.
    Uses SQLite rename-and-copy since ALTER TABLE cannot drop constraints.
    """
    table_sql = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='benchmark_runs'"
    ).fetchone()
    if table_sql is None:
        return
    if "UNIQUE" not in table_sql[0]:
        return

    logger.info("Migrating benchmark_runs: dropping UNIQUE constraint...")

    conn.execute("ALTER TABLE benchmark_runs RENAME TO benchmark_runs_old")
    conn.executescript("""
        CREATE TABLE benchmark_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_tag TEXT NOT NULL,
            sglang_version TEXT,
            rocm_version TEXT,
            build_date TEXT,
            model_name TEXT NOT NULL,
            tp_size INTEGER NOT NULL DEFAULT 8,
            mtp_enabled INTEGER NOT NULL DEFAULT 1,
            run_timestamp TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            error_message TEXT,
            result_dir TEXT,
            duration_total_sec REAL
        );
    """)
    conn.execute("""
        INSERT INTO benchmark_runs
            (id, image_tag, sglang_version, rocm_version, build_date,
             model_name, tp_size, mtp_enabled,
             run_timestamp, status, error_message, result_dir, duration_total_sec)
        SELECT
            id, image_tag, sglang_version, rocm_version, build_date,
            model_name, tp_size, mtp_enabled,
            run_timestamp, status, error_message, result_dir, duration_total_sec
        FROM benchmark_runs_old
    """)
    conn.execute("DROP TABLE benchmark_runs_old")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_runs_rocm_status ON benchmark_runs(rocm_version, status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_runs_build_date ON benchmark_runs(build_date)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_runs_image_model ON benchmark_runs(image_tag, model_name, tp_size, mtp_enabled)"
    )

    conn.commit()
    logger.info("Migration complete: UNIQUE constraint removed from benchmark_runs")


def init_db(conn: Optional[sqlite3.Connection] = None) -> None:
    """Create tables if they don't exist."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True
    conn.executescript(SCHEMA_SQL)
    _migrate_add_tp_mtp_columns(conn)
    _migrate_drop_unique_constraint(conn)
    conn.commit()
    if close:
        conn.close()
    logger.info("Database initialized: %s", DB_PATH)


def is_already_benchmarked(
    conn: sqlite3.Connection,
    image_tag: str,
    tp_size: int = 8,
    mtp_enabled: bool = True,
) -> bool:
    """Check if this image already has a completed benchmark with existing results."""
    row = conn.execute(
        "SELECT status, result_dir FROM benchmark_runs "
        "WHERE image_tag = ? AND model_name = ? AND tp_size = ? AND mtp_enabled = ?",
        (image_tag, _cfg.MODEL_NAME, tp_size, int(mtp_enabled)),
    ).fetchone()
    if row is None or row["status"] != "completed":
        return False
    # Verify the result directory and JSONL actually exist
    result_dir = row["result_dir"]
    if result_dir and Path(result_dir).is_dir() and (Path(result_dir) / "bench_results.jsonl").exists():
        return True
    logger.info("Result dir missing for completed run %s, will re-run", image_tag)
    return False


def create_run(
    conn: sqlite3.Connection,
    image_tag: str,
    sglang_version: str,
    rocm_version: str,
    build_date: str,
    result_dir: str,
    status: str = "running",
    tp_size: int = 8,
    mtp_enabled: bool = True,
) -> int:
    """Create or reset a benchmark_runs record and return its ID.

    If a previous record exists for this image+model+tp+mtp, reset it
    instead of inserting a duplicate.
    """
    existing = conn.execute(
        "SELECT id, status FROM benchmark_runs "
        "WHERE image_tag = ? AND model_name = ? AND tp_size = ? AND mtp_enabled = ?",
        (image_tag, _cfg.MODEL_NAME, tp_size, int(mtp_enabled)),
    ).fetchone()

    now = datetime.now(timezone.utc).isoformat()

    if existing:
        run_id = existing["id"]
        # Clean up old metrics/accuracy from the failed attempt
        conn.execute("DELETE FROM benchmark_metrics WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM accuracy_results WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM regression_alerts WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM version_snapshots WHERE run_id = ?", (run_id,))
        conn.execute(
            """UPDATE benchmark_runs
               SET sglang_version = ?, rocm_version = ?, build_date = ?,
                   run_timestamp = ?, status = ?, result_dir = ?,
                   error_message = NULL, duration_total_sec = NULL
               WHERE id = ?""",
            (sglang_version, rocm_version, build_date, now, status, result_dir, run_id),
        )
        conn.commit()
        logger.info("Reset existing failed run %d for %s", run_id, image_tag)
        return run_id

    cur = conn.execute(
        """INSERT INTO benchmark_runs
           (image_tag, sglang_version, rocm_version, build_date,
            model_name, tp_size, mtp_enabled, run_timestamp, status, result_dir)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            image_tag,
            sglang_version,
            rocm_version,
            build_date,
            _cfg.MODEL_NAME,
            tp_size,
            int(mtp_enabled),
            now,
            status,
            result_dir,
        ),
    )
    conn.commit()
    return cur.lastrowid


def update_run_status(
    conn: sqlite3.Connection,
    run_id: int,
    status: str,
    error_message: Optional[str] = None,
    duration_total_sec: Optional[float] = None,
) -> None:
    """Update the status of a benchmark run."""
    conn.execute(
        """UPDATE benchmark_runs
           SET status = ?, error_message = ?, duration_total_sec = ?
           WHERE id = ?""",
        (status, error_message, duration_total_sec, run_id),
    )
    conn.commit()


def _sanitize_json_line(line: str) -> str:
    """Replace bare Infinity/NaN with null for JSON parsing."""
    # Match Infinity and NaN that appear as JSON values (not inside strings)
    line = re.sub(r':\s*Infinity\b', ': null', line)
    line = re.sub(r':\s*-Infinity\b', ': null', line)
    line = re.sub(r':\s*NaN\b', ': null', line)
    return line


def parse_jsonl(jsonl_path: Path) -> list[dict]:
    """Parse a bench_results.jsonl file into a list of dicts."""
    results = []
    if not jsonl_path.exists():
        logger.warning("JSONL file not found: %s", jsonl_path)
        return results

    for line_num, raw_line in enumerate(
        jsonl_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            sanitized = _sanitize_json_line(line)
            data = json.loads(sanitized)
            results.append(data)
        except json.JSONDecodeError as e:
            logger.warning("Skipping JSONL line %d: %s", line_num, e)

    return results


def ingest_metrics(conn: sqlite3.Connection, run_id: int, jsonl_path: Path) -> int:
    """Parse JSONL and insert benchmark_metrics rows. Returns number of rows inserted."""
    entries = parse_jsonl(jsonl_path)
    count = 0
    for entry in entries:
        concurrency = entry.get("max_concurrency")
        if concurrency is None:
            continue
        try:
            conn.execute(
                """INSERT OR REPLACE INTO benchmark_metrics
                   (run_id, concurrency, completed_requests, benchmark_duration_s,
                    request_throughput, input_throughput, output_throughput, total_throughput,
                    mean_e2e_latency_ms, median_e2e_latency_ms, p90_e2e_latency_ms, p99_e2e_latency_ms,
                    mean_ttft_ms, median_ttft_ms, p99_ttft_ms,
                    mean_itl_ms, median_itl_ms, p95_itl_ms, p99_itl_ms,
                    mean_tpot_ms, median_tpot_ms)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    concurrency,
                    entry.get("completed"),
                    entry.get("duration"),
                    entry.get("request_throughput"),
                    entry.get("input_throughput"),
                    entry.get("output_throughput"),
                    entry.get("total_throughput"),
                    entry.get("mean_e2e_latency_ms"),
                    entry.get("median_e2e_latency_ms"),
                    entry.get("p90_e2e_latency_ms"),
                    entry.get("p99_e2e_latency_ms"),
                    entry.get("mean_ttft_ms"),
                    entry.get("median_ttft_ms"),
                    entry.get("p99_ttft_ms"),
                    entry.get("mean_itl_ms"),
                    entry.get("median_itl_ms"),
                    entry.get("p95_itl_ms"),
                    entry.get("p99_itl_ms"),
                    entry.get("mean_tpot_ms"),
                    entry.get("median_tpot_ms"),
                ),
            )
            count += 1
        except sqlite3.IntegrityError as e:
            logger.warning("Duplicate metric for run %d concurrency %d: %s", run_id, concurrency, e)

    conn.commit()
    logger.info("Ingested %d metric rows for run %d", count, run_id)
    return count


# Accuracy log patterns:
#   Format 1 (decimal):  "Accuracy: 0.943"
#   Format 2 (fraction): "Accuracy: 1850/2000 = 92.50%"
ACCURACY_DECIMAL_RE = re.compile(r"Accuracy:\s*(0?\.\d+|1\.0+)\s*$", re.MULTILINE)
ACCURACY_FRACTION_RE = re.compile(r"Accuracy:\s*(\d+)\s*/\s*(\d+)\s*=\s*([\d.]+)%")
# Also grab num_questions from the progress bar: "1319/1319"
NUM_QUESTIONS_RE = re.compile(r"(\d+)/\1\s*\[")


def parse_accuracy_log(log_path: Path) -> Optional[dict]:
    """Parse the accuracy log and extract the final accuracy line."""
    if not log_path.exists():
        logger.warning("Accuracy log not found: %s", log_path)
        return None

    text = log_path.read_text(encoding="utf-8", errors="replace")

    # Try fraction format first (more specific)
    fraction_matches = ACCURACY_FRACTION_RE.findall(text)
    if fraction_matches:
        correct, total, pct = fraction_matches[-1]
        return {
            "num_correct": int(correct),
            "num_questions": int(total),
            "accuracy_pct": float(pct),
            "raw_output": text[-2000:] if len(text) > 2000 else text,
        }

    # Try decimal format: "Accuracy: 0.943"
    decimal_matches = ACCURACY_DECIMAL_RE.findall(text)
    if decimal_matches:
        accuracy_frac = float(decimal_matches[-1])
        accuracy_pct = accuracy_frac * 100.0

        # Try to find num_questions from progress bar
        num_q_matches = NUM_QUESTIONS_RE.findall(text)
        num_questions = int(num_q_matches[-1]) if num_q_matches else None
        num_correct = round(accuracy_frac * num_questions) if num_questions else None

        return {
            "num_correct": num_correct,
            "num_questions": num_questions,
            "accuracy_pct": round(accuracy_pct, 2),
            "raw_output": text[-2000:] if len(text) > 2000 else text,
        }

    logger.warning("No accuracy line found in %s", log_path)
    return None


def ingest_accuracy(conn: sqlite3.Connection, run_id: int, log_path: Path) -> bool:
    """Parse accuracy log and insert into accuracy_results. Returns True if successful."""
    result = parse_accuracy_log(log_path)
    if result is None:
        return False

    try:
        conn.execute(
            """INSERT OR REPLACE INTO accuracy_results
               (run_id, num_questions, num_correct, accuracy_pct, raw_output)
               VALUES (?, ?, ?, ?, ?)""",
            (
                run_id,
                result["num_questions"],
                result["num_correct"],
                result["accuracy_pct"],
                result["raw_output"],
            ),
        )
        conn.commit()
        logger.info(
            "Ingested accuracy for run %d: %d/%d = %.2f%%",
            run_id,
            result["num_correct"],
            result["num_questions"],
            result["accuracy_pct"],
        )
        return True
    except sqlite3.IntegrityError as e:
        logger.warning("Duplicate accuracy for run %d: %s", run_id, e)
        return False


def ingest_version_snapshot(conn: sqlite3.Connection, run_id: int, snapshot_dict: dict) -> int:
    """Insert version snapshot rows from a snapshot JSON dict. Returns number of rows inserted."""
    libraries = snapshot_dict.get("libraries", [])
    count = 0
    for lib in libraries:
        try:
            conn.execute(
                """INSERT OR REPLACE INTO version_snapshots
                   (run_id, library_name, version, source_type,
                    git_commit, git_commit_date, git_commit_subject)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    run_id,
                    lib.get("name"),
                    lib.get("version"),
                    lib.get("source_type", "pip"),
                    lib.get("git_commit"),
                    lib.get("git_commit_date"),
                    lib.get("git_commit_subject"),
                ),
            )
            count += 1
        except sqlite3.IntegrityError as e:
            logger.warning("Duplicate version snapshot for run %d lib %s: %s", run_id, lib.get("name"), e)
    conn.commit()
    logger.info("Ingested %d version snapshot rows for run %d", count, run_id)
    return count


def get_version_snapshot(conn: sqlite3.Connection, run_id: int) -> list[dict]:
    """Retrieve all version snapshot rows for a given run."""
    rows = conn.execute(
        """SELECT library_name, version, source_type,
                  git_commit, git_commit_date, git_commit_subject
           FROM version_snapshots WHERE run_id = ?
           ORDER BY library_name""",
        (run_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def ingest_run(
    conn: sqlite3.Connection,
    run_id: int,
    result_dir: Path,
) -> None:
    """Full ingestion: parse JSONL metrics and accuracy log for a completed run."""
    jsonl_path = result_dir / "bench_results.jsonl"
    accuracy_path = result_dir / "accuracy_gsm8k.log"

    ingest_metrics(conn, run_id, jsonl_path)
    ingest_accuracy(conn, run_id, accuracy_path)

    version_path = result_dir / "version_snapshot.json"
    if version_path.exists():
        snapshot = json.loads(version_path.read_text())
        ingest_version_snapshot(conn, run_id, snapshot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    init_db()
    print(f"Database ready at {DB_PATH}")
