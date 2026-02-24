"""Docker Hub image discovery for rocm/sgl-dev tags."""

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.request import urlopen, Request
import json
import logging

from config import (
    DOCKER_HUB_API_BASE,
    DOCKER_HUB_REPO,
    TAG_FILTER_GPU,
    TAG_REGEX,
    ROCM_VERSIONS,
    DEFAULT_LOOKBACK_DAYS,
    MAX_LOOKBACK_DAYS,
)

logger = logging.getLogger(__name__)


@dataclass
class ImageInfo:
    full_tag: str
    full_image: str
    sglang_version: str
    rocm_version: str
    build_date: str  # YYYYMMDD
    build_datetime: datetime

    def __repr__(self) -> str:
        return (
            f"ImageInfo(tag={self.full_tag}, sglang={self.sglang_version}, "
            f"rocm={self.rocm_version}, date={self.build_date})"
        )


def _fetch_json(url: str) -> dict:
    """Fetch JSON from a URL."""
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def discover_images(
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    lookback_days_min: int = 0,
    rocm_versions: Optional[list[str]] = None,
    max_pages: int = 20,
) -> list[ImageInfo]:
    """Query Docker Hub for matching rocm/sgl-dev images within the lookback window.

    Args:
        lookback_days: Maximum days to look back (oldest cutoff).
        lookback_days_min: Minimum days to look back (newest cutoff).
            When non-zero, only images *older* than this many days ago are
            included.  For example, lookback_days=10, lookback_days_min=7
            returns images between 7 and 10 days ago.
        rocm_versions: ROCm versions to filter.
        max_pages: Maximum Docker Hub API pages to fetch.
    """
    if rocm_versions is None:
        rocm_versions = ROCM_VERSIONS

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    cutoff_str = cutoff.strftime("%Y%m%d")

    # Near-side cutoff: exclude images newer than lookback_days_min days ago
    cutoff_near_str = None
    if lookback_days_min > 0:
        cutoff_near = datetime.now(timezone.utc) - timedelta(days=lookback_days_min)
        cutoff_near_str = cutoff_near.strftime("%Y%m%d")

    url = (
        f"{DOCKER_HUB_API_BASE}/{DOCKER_HUB_REPO}/tags/"
        f"?page_size=100&ordering=-last_updated&name={TAG_FILTER_GPU}"
    )

    results: list[ImageInfo] = []
    pages_fetched = 0

    while url and pages_fetched < max_pages:
        logger.info("Fetching: %s", url)
        data = _fetch_json(url)
        pages_fetched += 1

        for tag_info in data.get("results", []):
            tag_name = tag_info.get("name", "")
            m = TAG_REGEX.match(tag_name)
            if not m:
                continue

            sglang_ver, rocm_ver, build_date = m.groups()

            if rocm_ver not in rocm_versions:
                continue

            if build_date < cutoff_str:
                continue

            if cutoff_near_str and build_date > cutoff_near_str:
                continue

            build_dt = datetime.strptime(build_date, "%Y%m%d").replace(
                tzinfo=timezone.utc
            )

            results.append(
                ImageInfo(
                    full_tag=tag_name,
                    full_image=f"{DOCKER_HUB_REPO}:{tag_name}",
                    sglang_version=sglang_ver,
                    rocm_version=rocm_ver,
                    build_date=build_date,
                    build_datetime=build_dt,
                )
            )

        url = data.get("next")

    # Sort by build_date ascending
    results.sort(key=lambda x: (x.build_date, x.rocm_version))
    if lookback_days_min:
        logger.info(
            "Discovered %d images within %d-%d day lookback window",
            len(results), lookback_days_min, lookback_days,
        )
    else:
        logger.info("Discovered %d images within %d-day lookback", len(results), lookback_days)
    return results


def _parse_lookback_days(value: str) -> tuple[int, int]:
    """Parse lookback-days value into (max, min). See orchestrator._parse_lookback_days."""
    for sep in ("-", ":"):
        if sep in value:
            parts = value.split(sep, 1)
            a, b = int(parts[0]), int(parts[1])
            lo, hi = min(a, b), max(a, b)
            return hi, lo
    return int(value), 0


def main():
    parser = argparse.ArgumentParser(description="Discover rocm/sgl-dev Docker images")
    parser.add_argument(
        "--lookback-days",
        type=str,
        default=str(DEFAULT_LOOKBACK_DAYS),
        help=f"Days to look back. Supports a single value (e.g. 10) "
        f"or a range MIN-MAX or MIN:MAX (e.g. 7-10). "
        f"(default: {DEFAULT_LOOKBACK_DAYS})",
    )
    parser.add_argument(
        "--rocm-versions",
        nargs="+",
        default=ROCM_VERSIONS,
        help=f"ROCm versions to filter (default: {ROCM_VERSIONS})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    lookback_max, lookback_min = _parse_lookback_days(args.lookback_days)
    images = discover_images(
        lookback_days=lookback_max,
        lookback_days_min=lookback_min,
        rocm_versions=args.rocm_versions,
    )

    if not images:
        print("No images found.")
        sys.exit(0)

    print(f"\nFound {len(images)} image(s):\n")
    for img in images:
        print(f"  {img.full_image}")
        print(f"    SGLang: {img.sglang_version}  ROCm: {img.rocm_version}  Date: {img.build_date}")
    print()


if __name__ == "__main__":
    main()
