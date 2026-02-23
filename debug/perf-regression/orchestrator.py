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

import config as _cfg
from config import (
    BENCH_SCRIPT,
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
    save_model_config,
    TP_MTP_VARIANTS,
    variant_label,
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

    # Kill the benchmark subprocess and its entire process group
    if _active_process and _active_process.poll() is None:
        logger.info("Terminating benchmark process group (pid=%d)", _active_process.pid)
        _kill_process_group(_active_process)
        try:
            _active_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

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


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill the process and its entire process group.

    When USE_SUDO_DOCKER is enabled, docker child processes are
    root-owned and survive a plain os.killpg from a non-root user.
    os.killpg may *partially* succeed (killing user-owned bash but
    not root-owned docker children) without raising PermissionError,
    so we must always continue to docker-kill and sudo-kill regardless.

    As a final safety net, closes the raw stdout fd to unblock any
    reader waiting on the pipe.  We use os.close() on the raw fd
    instead of proc.stdout.close() because the latter acquires
    TextIOWrapper's internal lock, which deadlocks when the main
    thread is blocked inside readline().
    """
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return

    # Step 1: killpg — kills user-owned processes in the group.
    # On Linux killpg succeeds if it can signal ANY process in the
    # group, so root-owned docker children silently survive.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass

    # Step 2: kill the Docker container so root-owned docker-exec
    # children exit and release the stdout pipe write-end.
    # _docker_cmd() already includes sudo when USE_SUDO_DOCKER=True.
    if _active_container:
        try:
            cmd = _docker_cmd() + ["kill", _active_container]
            subprocess.run(cmd, timeout=10, capture_output=True)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # Step 3: sudo kill for any remaining root-owned children
    try:
        subprocess.run(
            ["sudo", "kill", "-9", f"-{pgid}"],
            timeout=5, capture_output=True,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Step 4: fallback to killing just the direct child
    try:
        proc.kill()
    except (ProcessLookupError, OSError):
        pass

    # Safety net: close the raw stdout fd to unblock the main loop.
    # We bypass proc.stdout.close() because TextIOWrapper.close()
    # acquires an internal lock that deadlocks when the main thread
    # is blocked inside readline() holding the same lock.
    try:
        os.close(proc.stdout.fileno())
    except OSError:
        pass


def _monitor_server_log(server_log: Path, proc: subprocess.Popen) -> None:
    """Background thread: tail the server log for fatal GPU errors.

    If a fatal pattern is found, kills the entire process group so the
    benchmark aborts immediately instead of hanging on a dead server.
    """
    import threading

    FATAL_PATTERNS = [
        "Memory access fault by GPU",
        "Xcdna kernel error",
        "uncorrectable ECC error",
    ]

    def _watcher():
        # Wait for the server log file to appear (up to 5 min)
        for _ in range(300):
            if server_log.exists():
                break
            if proc.poll() is not None:
                return
            time.sleep(1)
        if not server_log.exists():
            return

        # Tail the file, checking new content every second
        with open(server_log, "r") as fh:
            while proc.poll() is None:
                line = fh.readline()
                if not line:
                    time.sleep(1)
                    continue
                for pat in FATAL_PATTERNS:
                    if pat in line:
                        logger.error(
                            "Fatal GPU error in server log: %s", line.rstrip()
                        )
                        logger.error(
                            "Killing benchmark process group (server crashed). "
                            "Full server log: %s", server_log,
                        )
                        _kill_process_group(proc)
                        return

    t = threading.Thread(target=_watcher, daemon=True)
    t.start()
    return t


def run_benchmark(image: str, result_dir: Path, tp_size: int = 8, mtp: bool = True) -> tuple[str, str]:
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
        "--model-path", _cfg.MODEL_PATH,
        "--tp-size", str(tp_size),
        "--concurrencies", CONCURRENCIES,
        "--home-dir", HOST_HOME_DIR,
        "--keep-container",
        "--result-dir", str(result_dir),
        "--wait-timeout-sec", str(WAIT_TIMEOUT_SEC),
    ]

    if mtp:
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

    log_file = result_dir / "orchestrator_output.log"
    server_log = result_dir / "server.log"
    logger.info("Running benchmark: %s", " ".join(cmd))
    logger.info("Output log: %s", log_file)
    logger.info("Server log: %s", server_log)

    # Use Popen so we can stream stdout, detect the container name early,
    # and handle Ctrl+C cleanup properly.
    # start_new_session=True creates a new process group so we can kill
    # bash AND all its child docker processes at once via os.killpg().
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    _active_process = proc

    # Send newline to answer the profile mode prompt, then close stdin
    try:
        proc.stdin.write("\n")
        proc.stdin.flush()
        proc.stdin.close()
    except BrokenPipeError:
        pass

    # Start background thread to monitor server log for GPU faults
    watcher = _monitor_server_log(server_log, proc)

    # Fatal patterns to also check in benchmark stdout/stderr
    FATAL_PATTERNS = [
        "Memory access fault by GPU",
        "Xcdna kernel error",
        "uncorrectable ECC error",
    ]

    # Stream stdout, capture it, and watch for the container name
    stdout_lines = []
    container_name = ""
    fatal_hit = ""

    with open(log_file, "w") as log_fh:
        try:
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

                # Detect fatal GPU errors in stdout as well
                if not fatal_hit:
                    for pat in FATAL_PATTERNS:
                        if pat in line:
                            fatal_hit = pat
                            logger.error("Fatal GPU error in stdout: %s", line.rstrip())
                            logger.error("Killing benchmark process group immediately")
                            _kill_process_group(proc)
                            break
        except (ValueError, OSError):
            # stdout fd was closed by the watcher thread to force-unblock us.
            # os.close(fd) causes OSError (EBADF), while higher-level close
            # causes ValueError.
            pass

    proc.wait()
    _active_process = None
    stdout = "".join(stdout_lines)

    if fatal_hit or proc.returncode == -9:
        reason = fatal_hit or "killed by server-log watcher (GPU fault)"
        raise subprocess.CalledProcessError(
            proc.returncode, cmd, stdout, f"Fatal GPU error: {reason}"
        )

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


def _salvage_version_snapshot(
    conn, run_id: int, container_name: str, result_dir: Path
) -> None:
    """Try to copy version_snapshot.json out of a container on failure.

    The snapshot is collected early in the benchmark script (right after
    the server health check), so it often exists even when the benchmark
    crashes later.  We look for it in the container and, if found, copy
    it to the host result dir and ingest it into the database.
    """
    from collector import ingest_version_snapshot

    host_path = result_dir / "version_snapshot.json"
    if host_path.exists():
        # Already on host (copied before failure)
        try:
            import json
            snapshot = json.loads(host_path.read_text())
            ingest_version_snapshot(conn, run_id, snapshot)
            logger.info("Ingested version snapshot from existing host file")
        except Exception as e:
            logger.warning("Failed to ingest existing version snapshot: %s", e)
        return

    # Try to find and copy from the container.
    # The container may be stopped (e.g. after docker kill on GPU fault),
    # so we first try docker-exec (running container), then fall back to
    # docker-cp with the known path pattern (works on stopped containers).
    result_dir.mkdir(parents=True, exist_ok=True)
    container_path = None

    # Method 1: docker exec find (only works on running containers)
    try:
        find_cmd = _docker_cmd() + [
            "exec", container_name, "bash", "-c",
            "find /tmp -name version_snapshot.json -maxdepth 3 2>/dev/null | head -1",
        ]
        find_result = subprocess.run(
            find_cmd, capture_output=True, text=True, timeout=10
        )
        # Only trust stdout when docker exec succeeded; on stopped
        # containers Docker prints the OCI error to stdout, which would
        # be mistaken for a file path.
        if find_result.returncode == 0:
            container_path = find_result.stdout.strip() or None
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        pass

    # Method 2: derive path from container name (works on stopped containers).
    # Container name: jacky-sglang-bench-<tag>-<YYYYMMDD_HHMMSS>
    # Container result dir: /tmp/sglang-bench-<YYYYMMDD_HHMMSS>
    if not container_path:
        m = re.search(r"-(\d{8}_\d{6})$", container_name)
        if m:
            container_path = f"/tmp/sglang-bench-{m.group(1)}/version_snapshot.json"

    if not container_path:
        logger.info("No version snapshot path found for container %s", container_name)
        return

    try:
        cp_cmd = _docker_cmd() + [
            "cp", f"{container_name}:{container_path}", str(host_path),
        ]
        subprocess.run(cp_cmd, check=True, capture_output=True, timeout=10)
        logger.info("Salvaged version snapshot from container: %s", container_path)

        import json
        snapshot = json.loads(host_path.read_text())
        ingest_version_snapshot(conn, run_id, snapshot)
    except Exception as e:
        logger.warning("Failed to salvage version snapshot: %s", e)


def process_image(conn, image_info, dry_run: bool = False, variants=None) -> bool:
    """Process a single image: pull, benchmark, collect, detect regressions.

    Iterates over all TP/MTP variants (or a filtered subset), tracking
    results independently per variant via tp_size/mtp_enabled columns.
    Returns True if all variants succeed.
    """
    global _active_container, _active_run_id, _active_conn

    tag = image_info.full_tag
    image = image_info.full_image
    all_success = True
    active_variants = variants or TP_MTP_VARIANTS

    # Pull image once for all variants
    if not dry_run:
        if not check_disk_space():
            logger.error("Skipping %s due to insufficient disk space", tag)
            return False
        if not pull_image(image):
            return False

    for tp_size, mtp in active_variants:
        label = variant_label(tp_size, mtp)

        if is_already_benchmarked(conn, tag, tp_size=tp_size, mtp_enabled=mtp):
            logger.info("Skipping %s [%s] (already benchmarked)", tag, label)
            continue

        if dry_run:
            logger.info("[DRY RUN] Would benchmark: %s [%s]", image, label)
            continue

        result_dir = BENCHMARK_RUNS_DIR / f"{tag}_{label}"
        run_id = None
        container_name = ""
        _active_conn = conn

        try:
            # Clean up stale result dir from a previous failed attempt
            if result_dir.exists():
                logger.info("Removing stale result dir: %s", result_dir)
                shutil.rmtree(result_dir)

            # Create DB record
            run_id = create_run(
                conn,
                image_tag=tag,
                sglang_version=image_info.sglang_version,
                rocm_version=image_info.rocm_version,
                build_date=image_info.build_date,
                result_dir=str(result_dir),
                status="running",
                tp_size=tp_size,
                mtp_enabled=mtp,
            )
            _active_run_id = run_id

            # Run benchmark
            start_time = time.monotonic()
            stdout, container_name = run_benchmark(
                image, result_dir, tp_size=tp_size, mtp=mtp
            )
            duration = time.monotonic() - start_time

            # Ingest results
            ingest_run(conn, run_id, result_dir)

            # Detect regressions
            alerts = detect_regressions(conn, run_id)
            if alerts:
                for a in alerts:
                    logger.warning(
                        "REGRESSION [%s]: %s (c=%s): %.2f vs baseline %.2f (%.1f%%)",
                        label,
                        a["metric_name"],
                        a.get("concurrency", "N/A"),
                        a["current_value"],
                        a["baseline_value"],
                        a["regression_pct"],
                    )

            # Mark completed
            update_run_status(conn, run_id, "completed", duration_total_sec=duration)
            logger.info("Completed benchmark for %s [%s] in %.1f sec", tag, label, duration)

        except subprocess.TimeoutExpired as e:
            msg = f"Benchmark timed out [{label}]: {e}"
            logger.error(msg)
            if run_id:
                update_run_status(conn, run_id, "failed", error_message=msg)
            all_success = False

        except subprocess.CalledProcessError as e:
            output_tail = (e.output or e.stderr or '')[-500:]
            log_path = result_dir / "orchestrator_output.log"
            msg = f"Benchmark failed [{label}] (rc={e.returncode}): {output_tail}"
            logger.error(msg)
            if log_path.exists():
                logger.error("Full output log: %s", log_path)
            if run_id:
                update_run_status(conn, run_id, "failed", error_message=msg)
            all_success = False

        except Exception as e:
            msg = f"Unexpected error [{label}]: {e}"
            logger.error(msg, exc_info=True)
            if run_id:
                update_run_status(conn, run_id, "failed", error_message=msg)
            all_success = False

        finally:
            cname = container_name or _active_container
            if cname and run_id:
                _salvage_version_snapshot(conn, run_id, cname, result_dir)
            cleanup_container(cname)
            _active_container = ""
            _active_run_id = None
            _active_conn = None

    return all_success


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
        "--model-path",
        type=str,
        default=None,
        help=f"Model path for sglang.launch_server (default: {_cfg.MODEL_PATH})",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help=f"Short model name for DB records (default: {_cfg.MODEL_NAME})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover and filter images but don't run benchmarks",
    )
    parser.add_argument(
        "--force-rerun",
        type=str,
        nargs="*",
        default=None,
        help="Force re-run image tags (deletes existing records). "
        "If no tags specified, re-runs all images found in --lookback-days.",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Only run specific variants (e.g., TP8 TP8+MTP). Default: all",
    )
    args = parser.parse_args()

    # Interactive model path prompt (same pattern as run-local-benchmark-e2e.sh)
    # Strip ANSI escape sequences to prevent garbage model names from
    # accidental arrow-key input (e.g. \x1b[A from pressing Up).
    def _strip_ansi(s: str) -> str:
        return re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', s).strip()

    if args.model_path:
        _cfg.MODEL_PATH = args.model_path
    else:
        user_input = _strip_ansi(input(f"Model path (default: {_cfg.MODEL_PATH}): "))
        if user_input:
            _cfg.MODEL_PATH = user_input

    # Derive model name from path basename; --model-name overrides
    if args.model_name:
        _cfg.MODEL_NAME = args.model_name
    else:
        _cfg.MODEL_NAME = Path(_cfg.MODEL_PATH).name or Path(_cfg.MODEL_PATH).parent.name

    # Persist for next run
    save_model_config(_cfg.MODEL_PATH, _cfg.MODEL_NAME)

    # Parse --variants filter
    active_variants = None
    if args.variants:
        active_variants = [
            (tp, mtp) for tp, mtp in TP_MTP_VARIANTS
            if variant_label(tp, mtp) in args.variants
        ]
        if not active_variants:
            available = [variant_label(t, m) for t, m in TP_MTP_VARIANTS]
            parser.error(f"No matching variants. Available: {available}")

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

        # Discover images
        images = discover_images(
            lookback_days=args.lookback_days,
            rocm_versions=args.rocm_versions,
        )

        # Handle force-rerun
        # Normalize tags: strip "repo:" prefix if present so they match DB values
        # When --force-rerun is given with no args, re-run all discovered images
        if args.force_rerun is not None:
            if args.force_rerun:
                args.force_rerun = [
                    t.split(":")[-1] if ":" in t else t for t in args.force_rerun
                ]
            else:
                args.force_rerun = [img.full_tag for img in images]
                logger.info(
                    "No tags specified for --force-rerun, re-running all %d discovered image(s)",
                    len(args.force_rerun),
                )
            rerun_variants = active_variants or TP_MTP_VARIANTS
            for tag in args.force_rerun:
                for tp, mtp in rerun_variants:
                    logger.info("Force re-running: %s [%s]", tag, variant_label(tp, mtp))
                    conn.execute(
                        "DELETE FROM benchmark_runs "
                        "WHERE image_tag = ? AND model_name = ? AND tp_size = ? AND mtp_enabled = ?",
                        (tag, _cfg.MODEL_NAME, tp, int(mtp)),
                    )
            conn.commit()

        if not images:
            logger.info("No images found. Nothing to do.")
            return

        logger.info("Found %d image(s) to process", len(images))

        # Process each image sequentially
        success_count = 0
        fail_count = 0

        for img in images:
            if process_image(conn, img, dry_run=args.dry_run, variants=active_variants):
                success_count += 1
            else:
                fail_count += 1

        logger.info(
            "Orchestrator finished: %d succeeded, %d failed",
            success_count,
            fail_count,
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
