from __future__ import annotations

import logging
import threading
import time
import requests
from prometheus_client.parser import text_string_to_metric_families
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# --- metric names as they appear in vLLM /metrics ---
_METRIC_PREFILL = "vllm:request_prefill_time_seconds"
_METRIC_TTFT = "vllm:time_to_first_token_seconds"
_METRIC_DECODE = "vllm:request_decode_time_seconds"

# V1 engine gauge name; fall back to alternative names if missing.
_CACHE_GAUGE_CANDIDATES = [
    "vllm:kv_cache_usage_perc",
    "vllm:gpu_cache_usage_perc",
]


class MetricsSnapshot:
    """Parsed /metrics endpoint at a point in time."""

    def __init__(self, body: str):
        self._histograms: Dict[str, Tuple[float, float]] = {}
        self._gauges: Dict[str, float] = {}

        for family in text_string_to_metric_families(body):
            name = family.name

            if family.type == "histogram":
                s = c = 0.0
                for sample in family.samples:
                    if sample.name.endswith("_sum"):
                        s = sample.value
                    elif sample.name.endswith("_count"):
                        c = sample.value
                self._histograms[name] = (s, c)

            elif family.type == "gauge":
                val = None
                for sample in family.samples:
                    val = sample.value  # last wins
                if val is not None:
                    self._gauges[name] = val

    def get_histogram_sum_count(self, metric_name: str) -> Optional[Tuple[float, float]]:
        return self._histograms.get(metric_name)

    def get_gauge(self, metric_name: str) -> Optional[float]:
        return self._gauges.get(metric_name)

    def get_first_gauge(self, candidates: list[str]) -> Optional[float]:
        """Try each candidate gauge name, return the first found."""
        for name in candidates:
            v = self._gauges.get(name)
            if v is not None:
                return v
        return None

    @classmethod
    def fetch(cls, url: str) -> MetricsSnapshot:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return cls(resp.text)


def compute_differential_average(
    before: MetricsSnapshot,
    after: MetricsSnapshot,
    metric_name: str,
) -> Optional[float]:
    b = before.get_histogram_sum_count(metric_name)
    a = after.get_histogram_sum_count(metric_name)
    if b is None or a is None:
        return None
    d_sum = a[0] - b[0]
    d_count = a[1] - b[1]
    if d_count <= 0:
        return None
    return d_sum / d_count


class PeakKVCacheTracker:
    """Background thread polling /metrics at high frequency to capture peak KV cache usage."""

    def __init__(self, metrics_url: str, interval: float = 0.1):
        self._url = metrics_url
        self._interval = interval
        self.peak_perc: float = 0.0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._stop.clear()
        self.peak_perc = 0.0
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                snap = MetricsSnapshot.fetch(self._url)
                val = snap.get_first_gauge(_CACHE_GAUGE_CANDIDATES)
                if val is not None:
                    self.peak_perc = max(self.peak_perc, val)
            except Exception:
                logger.debug("KV cache poll failed", exc_info=True)
            self._stop.wait(self._interval)
