"""Main orchestrator: discover -> pull -> benchmark -> collect -> detect regressions."""

import argparse
import fcntl
import logging
import os
import re
import signal
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from config import (
    BENCH_SCRIPT,
    MODEL_PATH,
    CONCURRENCIES,
    USE_SUDO_DOCKER,
    MTP_MODE,
    ACCURACY_MODE,
    ACCURACY_NUM_QUESTIONS,
    ACCURACY_PARALLEL,
    ACCURACY_NUM_SHOTS,
    WAIT_TIMEOUT_SEC,
    HOST_HOME_DIR,
    DEFAULT_LOOKBACK_DAYS,
    MAX_LOOKBACK_DAYS,
    MIN_FREE_DISK_GB,
    MAX_IMAGES_RETAINED,
    ROCM_VERSIONS,
    LOCK_FILE,
    BENCHMARK_RUNS_DIR,
    LOG_DIR,
    MODEL_NAME,
)
from collector import get_connection, init_db, is_already_benchmarked, create_run, update_run_status, ingest_run
from discover import discover_images
from regression import detect_regressions

logger = logging.getLogger(__name__)

# Pattern to extract container name from script output
CONTAINER_NAME_RE = re.compile(r"Starting container:\s*(\S+)")

# Global state for signal-based cleanup
_active_process: subprocess.Popen = None
_active_container: str = ""
_active_run_id: int = None
_active_conn = None


def _signal_handler(signum, frame):
    """Handle SIGINT/SIGTERM: kill subprocess, clean up container, exit."""
    sig_name = signal.Signals(signum).name
    logger.warning("Received %s, cleaning up...", sig_name)

    # Kill the benchmark subprocess
    if _active_process and _active_process.poll() is None:
        logger.info("Terminating benchmark subprocess (pid=%d)", _active_process.pid)
        _active_process.terminate()
        try:
            _active_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.info("Force-killing benchmark subprocess")
            _active_process.kill()
            _active_process.wait(timeout=5)

    # Clean up the Docker container
    if _active_container:
        logger.info("Cleaning up container: %s", _active_container)
        cleanup_container(_active_container)

    # Mark the DB record as failed
    if _active_run_id and _active_conn:
        try:
            update_run_status(
                _active_conn, _active_run_id, "failed",
                error_message=f"Interrupted by {sig_name}",
            )
        except Exception:
            pass

    logger.info("Cleanup complete, exiting.")
    sys.exit(1)


def _docker_cmd() -> list[str]:
    """Return the docker command prefix (with or without sudo).

    Always passes --config pointing to the user's docker config so that
    sudo docker uses the user's credentials rather than root's.
    """
    docker_config = f"{HOST_HOME_DIR}/.docker"
    if USE_SUDO_DOCKER:
        return ["sudo", "docker", "--config", docker_config]
    return ["docker", "--config", docker_config]


def check_disk_space() -> bool:
    """Check if there's enough free disk space."""
    stat = shutil.disk_usage("/")
    free_gb = stat.free / (1024 ** 3)
    if free_gb < MIN_FREE_DISK_GB:
        logger.error("Insufficient disk space: %.1f GB free (need %d GB)", free_gb, MIN_FREE_DISK_GB)
        return False
    logger.info("Disk space OK: %.1f GB free", free_gb)
    return True


def pull_image(image: str, retries: int = 3) -> bool:
    """Pull a Docker image with retries and exponential backoff.

    Uses the user's docker config via _docker_cmd() so that
    sudo docker inherits the correct credentials instead of root's.
    """
    cmd = _docker_cmd() + ["pull", image]
    for attempt in range(1, retries + 1):
        logger.info("Pulling image (attempt %d/%d): %s", attempt, retries, image)
        try:
            subprocess.run(cmd, check=True, timeout=1800, capture_output=True, text=True)
            logger.info("Successfully pulled: %s", image)
            return True
        except subprocess.CalledProcessError as e:
            logger.warning("Pull failed (attempt %d): %s", attempt, e.stderr[:500] if e.stderr else str(e))
        except subprocess.TimeoutExpired:
            logger.warning("Pull timed out (attempt %d)", attempt)

        if attempt < retries:
            backoff = 30 * (2 ** (attempt - 1))
            logger.info("Retrying in %d seconds...", backoff)
            time.sleep(backoff)

    logger.error("Failed to pull image after %d attempts: %s", retries, image)
    return False


def run_benchmark(image: str, result_dir: Path) -> tuple[str, str]:
    """Run the benchmark script and return (stdout, container_name).

    Passes all flags to suppress interactive prompts.
    Pipes a newline to stdin to answer the profile mode prompt with 'N'.
    Uses Popen to stream output in real time for container name detection
    and to allow clean interruption via signals.
    """
    global _active_process, _active_container

    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "bash", BENCH_SCRIPT,
        "--image", image,
        "--model-path", MODEL_PATH,
        "--concurrencies", CONCURRENCIES,
        "--home-dir", HOST_HOME_DIR,
        "--keep-container",
        "--result-dir", str(result_dir),
        "--wait-timeout-sec", str(WAIT_TIMEOUT_SEC),
    ]

    if MTP_MODE:
        cmd.append("--mtp")
    else:
        cmd.append("--no-mtp")

    if ACCURACY_MODE:
        cmd.extend([
            "--accuracy",
            "--accuracy-num-questions", str(ACCURACY_NUM_QUESTIONS),
            "--accuracy-parallel", str(ACCURACY_PARALLEL),
            "--accuracy-num-shots", str(ACCURACY_NUM_SHOTS),
        ])

    logger.info("Running benchmark: %s", " ".join(cmd))

    # Use Popen so we can stream stdout, detect the container name early,
    # and handle Ctrl+C cleanup properly.
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    _active_process = proc

    # Send newline to answer the profile mode prompt, then close stdin
    try:
        proc.stdin.write("\n")
        proc.stdin.flush()
        proc.stdin.close()
    except BrokenPipeError:
        pass

    # Stream stdout, capture it, and watch for the container name
    stdout_lines = []
    container_name = ""
    log_file = result_dir / "orchestrator_output.log"

    with open(log_file, "w") as log_fh:
        for line in proc.stdout:
            log_fh.write(line)
            log_fh.flush()
            stdout_lines.append(line)

            if not container_name:
                match = CONTAINER_NAME_RE.search(line)
                if match:
                    container_name = match.group(1)
                    _active_container = container_name
                    logger.info("Detected container name: %s", container_name)

    proc.wait()
    _active_process = None
    stdout = "".join(stdout_lines)

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, cmd, stdout, "")

    return stdout, container_name


def cleanup_container(container_name: str) -> None:
    """Remove the benchmark container."""
    if not container_name:
        return
    cmd = _docker_cmd() + ["rm", "-f", container_name]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=60)
        logger.info("Removed container: %s", container_name)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        logger.warning("Failed to remove container %s: %s", container_name, e)


def prune_old_images() -> None:
    """Remove old Docker images, keeping only MAX_IMAGES_RETAINED most recent per ROCm version."""
    from config import TAG_REGEX, DOCKER_HUB_REPO

    cmd = _docker_cmd() + ["images", "--format", "{{.Repository}}:{{.Tag}}", DOCKER_HUB_REPO]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        lines = [l.strip() for l in result.stdout.splitlines() if l.strip()]
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        logger.warning("Failed to list Docker images for pruning")
        return

    # Group by rocm_version
    images_by_rocm: dict[str, list[tuple[str, str]]] = {}
    for full_image in lines:
        tag = full_image.split(":")[-1] if ":" in full_image else ""
        m = TAG_REGEX.match(tag)
        if not m:
            continue
        _, rocm_ver, build_date = m.groups()
        images_by_rocm.setdefault(rocm_ver, []).append((build_date, full_image))

    for rocm_ver, images in images_by_rocm.items():
        images.sort(key=lambda x: x[0], reverse=True)
        to_remove = images[MAX_IMAGES_RETAINED:]
        for _, image in to_remove:
            logger.info("Pruning old image: %s", image)
            rm_cmd = _docker_cmd() + ["rmi", image]
            try:
                subprocess.run(rm_cmd, capture_output=True, text=True, timeout=60)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                logger.warning("Failed to remove image: %s", image)


def process_image(conn, image_info, dry_run: bool = False) -> bool:
    """Process a single image: pull, benchmark, collect, detect regressions.
    Returns True on success.
    """
    global _active_container, _active_run_id, _active_conn

    tag = image_info.full_tag
    image = image_info.full_image

    if is_already_benchmarked(conn, tag):
        logger.info("Skipping already benchmarked: %s", tag)
        return True

    if dry_run:
        logger.info("[DRY RUN] Would benchmark: %s", image)
        return True

    if not check_disk_space():
        logger.error("Skipping %s due to insufficient disk space", tag)
        return False

    result_dir = BENCHMARK_RUNS_DIR / tag
    run_id = None
    container_name = ""
    _active_conn = conn

    try:
        # Pull image
        if not pull_image(image):
            return False

        # Create DB record
        run_id = create_run(
            conn,
            image_tag=tag,
            sglang_version=image_info.sglang_version,
            rocm_version=image_info.rocm_version,
            build_date=image_info.build_date,
            result_dir=str(result_dir),
            status="running",
        )
        _active_run_id = run_id

        # Run benchmark
        start_time = time.monotonic()
        stdout, container_name = run_benchmark(image, result_dir)
        duration = time.monotonic() - start_time

        # Ingest results
        ingest_run(conn, run_id, result_dir)

        # Detect regressions
        alerts = detect_regressions(conn, run_id)
        if alerts:
            for a in alerts:
                logger.warning(
                    "REGRESSION: %s (c=%s): %.2f vs baseline %.2f (%.1f%%)",
                    a["metric_name"],
                    a.get("concurrency", "N/A"),
                    a["current_value"],
                    a["baseline_value"],
                    a["regression_pct"],
                )

        # Mark completed
        update_run_status(conn, run_id, "completed", duration_total_sec=duration)
        logger.info("Completed benchmark for %s in %.1f sec", tag, duration)
        return True

    except subprocess.TimeoutExpired as e:
        msg = f"Benchmark timed out: {e}"
        logger.error(msg)
        if run_id:
            update_run_status(conn, run_id, "failed", error_message=msg)
        return False

    except subprocess.CalledProcessError as e:
        msg = f"Benchmark failed (rc={e.returncode}): {(e.stderr or '')[:500]}"
        logger.error(msg)
        if run_id:
            update_run_status(conn, run_id, "failed", error_message=msg)
        return False

    except Exception as e:
        msg = f"Unexpected error: {e}"
        logger.error(msg, exc_info=True)
        if run_id:
            update_run_status(conn, run_id, "failed", error_message=msg)
        return False

    finally:
        cleanup_container(container_name)
        _active_container = ""
        _active_run_id = None
        _active_conn = None


def main():
    parser = argparse.ArgumentParser(
        description="ROCm SGLang performance regression orchestrator"
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f"Days to look back for images (default: {DEFAULT_LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--rocm-versions",
        nargs="+",
        default=ROCM_VERSIONS,
        help=f"ROCm versions to test (default: {ROCM_VERSIONS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and filter images but don't run benchmarks",
    )
    parser.add_argument(
        "--force-rerun",
        type=str,
        default=None,
        help="Force re-run a specific image tag (deletes existing record)",
    )
    args = parser.parse_args()

    # Validate rocm versions
    for v in args.rocm_versions:
        if v not in ROCM_VERSIONS:
            parser.error(
                f"Unknown ROCm version '{v}'. Valid values: {ROCM_VERSIONS}"
            )

    # Setup logging
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    BENCHMARK_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / "orchestrator.log"
    handlers = [logging.StreamHandler()]
    try:
        handlers.append(logging.FileHandler(log_file))
    except PermissionError:
        print(f"WARNING: Cannot write to {log_file}, logging to stderr only", file=sys.stderr)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        handlers=handlers,
    )

    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Acquire file lock (non-blocking)
    lock_fd = open(LOCK_FILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.error("Another orchestrator instance is already running. Exiting.")
        sys.exit(1)

    try:
        logger.info(
            "Starting orchestrator: lookback=%d days, rocm=%s, dry_run=%s",
            args.lookback_days,
            args.rocm_versions,
            args.dry_run,
        )

        if not check_disk_space():
            sys.exit(1)

        # Initialize DB
        conn = get_connection()
        init_db(conn)

        # Handle force-rerun
        if args.force_rerun:
            logger.info("Force re-running: %s", args.force_rerun)
            conn.execute(
                "DELETE FROM benchmark_runs WHERE image_tag = ? AND model_name = ?",
                (args.force_rerun, MODEL_NAME),
            )
            conn.commit()

        # Discover images
        images = discover_images(
            lookback_days=args.lookback_days,
            rocm_versions=args.rocm_versions,
        )

        if not images:
            logger.info("No images found. Nothing to do.")
            return

        logger.info("Found %d image(s) to process", len(images))

        # Process each image sequentially
        success_count = 0
        fail_count = 0
        skip_count = 0

        for img in images:
            if is_already_benchmarked(conn, img.full_tag) and not args.force_rerun:
                logger.info("Skipping (already completed): %s", img.full_tag)
                skip_count += 1
                continue

            if process_image(conn, img, dry_run=args.dry_run):
                success_count += 1
            else:
                fail_count += 1

        logger.info(
            "Orchestrator finished: %d succeeded, %d failed, %d skipped",
            success_count,
            fail_count,
            skip_count,
        )

        # Prune old images
        if not args.dry_run:
            prune_old_images()

        conn.close()

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


if __name__ == "__main__":
    main()
