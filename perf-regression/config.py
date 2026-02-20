"""Central configuration for the ROCm performance regression testing system."""

import os
import re
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
HOST_HOME_DIR = "/home/yichiche"
BENCH_SCRIPT = "/home/yichiche/agent-box/run-local-benchmark-e2e.sh"
MODEL_PATH = "/raid/models/DeepSeek-R1-MXFP4-Preview/"
MODEL_NAME = "DeepSeek-R1-MXFP4"

# Runtime directories (auto-created)
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
BENCHMARK_RUNS_DIR = Path(HOST_HOME_DIR) / "benchmark_runs"
DB_PATH = DATA_DIR / "perf_regression.db"
LOCK_FILE = BASE_DIR / ".orchestrator.lock"

# ── Docker Hub ─────────────────────────────────────────────────────────────
DOCKER_HUB_REPO = "rocm/sgl-dev"
DOCKER_HUB_API_BASE = "https://hub.docker.com/v2/repositories"
TAG_FILTER_GPU = "mi35x"
ROCM_VERSIONS = ["700", "720"]

# Tag regex: v0.5.7-rocm700-mi35x-20260108
TAG_REGEX = re.compile(
    r"^(v[\d.]+(?:\.post\d+)?(?:rc\d+)?)-rocm(\d+)-mi35x-(\d{8})$"
)

# ── Benchmark settings ─────────────────────────────────────────────────────
USE_SUDO_DOCKER = True
MTP_MODE = True
CONCURRENCIES = "1,2,4"

ACCURACY_MODE = True
ACCURACY_NUM_QUESTIONS = 2000
ACCURACY_PARALLEL = 1000
ACCURACY_NUM_SHOTS = 1

WAIT_TIMEOUT_SEC = 2400

# ── Scheduling & retention ─────────────────────────────────────────────────
DEFAULT_LOOKBACK_DAYS = 3
MAX_LOOKBACK_DAYS = 60
MIN_FREE_DISK_GB = 50
MAX_IMAGES_RETAINED = 5

# ── Regression detection ──────────────────────────────────────────────────
REGRESSION_THRESHOLD_PCT = 5.0
REGRESSION_WINDOW = 5

# Metrics to monitor: (metric_name, direction)
# "higher_better" = throughput-like, "lower_better" = latency-like
MONITORED_METRICS = [
    ("output_throughput", "higher_better"),
    ("total_throughput", "higher_better"),
    ("median_e2e_latency_ms", "lower_better"),
    ("median_ttft_ms", "lower_better"),
    ("median_itl_ms", "lower_better"),
    ("p99_e2e_latency_ms", "lower_better"),
]

# p99 uses a wider threshold
METRIC_THRESHOLDS = {
    "p99_e2e_latency_ms": 10.0,  # 10% for tail latency
}

ACCURACY_REGRESSION_THRESHOLD = 2.0  # absolute percentage points

# ── Dashboard ──────────────────────────────────────────────────────────────
DASHBOARD_PORT = 8080
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
