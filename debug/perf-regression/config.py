"""Central configuration for the ROCm performance regression testing system."""

import os
import re
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
AGENT_BOX_DIR = BASE_DIR.parent.parent
HOST_HOME_DIR = os.getenv("AGENT_BOX_HOST_HOME", str(AGENT_BOX_DIR.parent))
BENCH_SCRIPT = str(AGENT_BOX_DIR / "benchmark" / "run-local-benchmark-e2e.sh")

# ── Model defaults (overridden by saved config or CLI args) ───────────────
_DEFAULT_MODEL_PATH = "/raid/models/DeepSeek-R1-MXFP4-Preview/"
_DEFAULT_MODEL_NAME = Path(_DEFAULT_MODEL_PATH).name or Path(_DEFAULT_MODEL_PATH).parent.name
_CONFIG_FILE = AGENT_BOX_DIR / ".bench_config"


def _load_saved_config() -> dict[str, str]:
    """Load key=value pairs from the shared bench config file."""
    saved = {}
    if _CONFIG_FILE.is_file():
        for line in _CONFIG_FILE.read_text().splitlines():
            if "=" in line:
                key, _, value = line.partition("=")
                saved[key.strip()] = value.strip()
    return saved


def save_model_config(model_path: str, model_name: str) -> None:
    """Persist model settings back to the shared config file.

    Preserves any other keys (e.g. HOST_HOME_DIR) already in the file.
    """
    saved = _load_saved_config()
    saved["MODEL_PATH"] = model_path
    saved["MODEL_NAME"] = model_name
    _CONFIG_FILE.write_text(
        "\n".join(f"{k}={v}" for k, v in saved.items()) + "\n"
    )


_saved = _load_saved_config()
MODEL_PATH = _saved.get("MODEL_PATH", _DEFAULT_MODEL_PATH)
MODEL_NAME = _saved.get("MODEL_NAME", _DEFAULT_MODEL_NAME)

# Runtime directories (auto-created)
DATA_DIR = BASE_DIR / "data"
BENCHMARK_RUNS_DIR = Path(HOST_HOME_DIR) / "benchmark_runs"
LOG_DIR = BENCHMARK_RUNS_DIR / "logs"
DB_PATH = Path(
    os.getenv(
        "PERF_REGRESSION_DB_PATH",
        str(BENCHMARK_RUNS_DIR / "data" / "perf_regression.db"),
    )
)
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
MTP_MODE = True  # Legacy: used by run_daily.sh; orchestrator uses TP_MTP_VARIANTS
CONCURRENCIES = "1,2,4"

# ── TP / MTP variant matrix ──────────────────────────────────────────────
TP_MTP_VARIANTS = [
    (2, False),   # TP2
    (2, True),    # TP2+MTP
    (4, False),   # TP4
    (4, True),    # TP4+MTP
    (8, False),   # TP8
    (8, True),    # TP8+MTP
]


def variant_label(tp_size: int, mtp: bool) -> str:
    """Human-readable variant label."""
    return f"TP{tp_size}{'+MTP' if mtp else ''}"


def variant_model_name(base_name: str, tp_size: int, mtp: bool) -> str:
    """Unique model_name for DB tracking per variant."""
    return f"{base_name}-{variant_label(tp_size, mtp)}"

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

# ── Version tracking ──────────────────────────────────────────────────────
GIT_TRACKED_REPOS = [
    {"name": "sglang",  "path": "/sgl-workspace/sglang"},
    {"name": "aiter",   "path": "/sgl-workspace/aiter", "pip_name": "amd-aiter"},
]

PIP_TRACKED_PACKAGES = [
    "torch",
    "transformers",
    "flashinfer",
    "triton",
]

HIGH_PRIORITY_LIBRARIES = ["sglang", "aiter", "triton", "flashinfer"]
