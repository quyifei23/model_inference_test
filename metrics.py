from __future__ import annotations

import logging
import threading
import time
import requests
from prometheus_client.parser import text_string_to_metric_families
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --- vLLM metric names ---
_METRIC_VLLM_PREFILL = "vllm:request_prefill_time_seconds"
_METRIC_VLLM_TTFT = "vllm:time_to_first_token_seconds"
_METRIC_VLLM_DECODE = "vllm:request_decode_time_seconds"
_CACHE_GAUGE_VLLM = ["vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"]

# --- SGLang metric names ---
_METRIC_SGLANG_PREFILL = "sglang:per_stage_req_latency_seconds"
_SGLANG_PREFILL_LABEL = {"stage": "prefill_forward"}
_METRIC_SGLANG_TTFT = "sglang:time_to_first_token_seconds"
_METRIC_SGLANG_E2E = "sglang:e2e_request_latency_seconds"
_CACHE_GAUGE_SGLANG = ["sglang:token_usage"]


# --- per-backend lookup tables ---

def _prefill_metric(server_type: str) -> str:
    return _METRIC_VLLM_PREFILL if server_type == "vllm" else _METRIC_SGLANG_PREFILL


def _ttft_metric(server_type: str) -> str:
    return _METRIC_VLLM_TTFT if server_type == "vllm" else _METRIC_SGLANG_TTFT


def _decode_metric(server_type: str) -> str | None:
    if server_type == "vllm":
        return _METRIC_VLLM_DECODE
    return None  # SGLang: derive from e2e - ttft


def _e2e_metric(server_type: str) -> str | None:
    if server_type == "sglang":
        return _METRIC_SGLANG_E2E
    return None


def _cache_gauge_candidates(server_type: str) -> list[str]:
    return _CACHE_GAUGE_VLLM if server_type == "vllm" else _CACHE_GAUGE_SGLANG


def _prefill_label_filter(server_type: str) -> dict | None:
    if server_type == "sglang":
        return _SGLANG_PREFILL_LABEL
    return None


# ---------------------------------------------------------------------------
# MetricsSnapshot
# ---------------------------------------------------------------------------

def _labels_match(labels: Dict[str, str], wanted: Dict[str, str]) -> bool:
    for k, v in wanted.items():
        if labels.get(k) != v:
            return False
    return True


def _labels_key(labels: Dict[str, str]) -> str:
    """Stable string key for a label dict."""
    return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))


class MetricsSnapshot:
    """Parsed /metrics endpoint at a point in time.

    Histograms are stored per (family_name, labels_key) so that metrics with
    different label combinations (e.g. SGLang's per-stage latencies) are kept separate.
    """

    def __init__(self, body: str):
        self._histograms: Dict[Tuple[str, str], Tuple[float, float]] = {}
        self._gauges: Dict[str, float] = {}

        for family in text_string_to_metric_families(body):
            fname = family.name

            if family.type == "histogram":
                # Group sum/count pairs by matching label sets.
                pairs: Dict[str, Dict[str, float]] = {}  # labels_key -> {"sum": x, "count": y}
                for sample in family.samples:
                    if sample.name.endswith("_sum"):
                        suffix_pos = -4  # "_sum"
                    elif sample.name.endswith("_count"):
                        suffix_pos = -6  # "_count"
                    else:
                        continue
                    key = _labels_key(sample.labels)
                    if key not in pairs:
                        pairs[key] = {}
                    kind = sample.name[suffix_pos + 1:]  # "sum" or "count"
                    pairs[key][kind] = sample.value

                for lkey, vals in pairs.items():
                    self._histograms[(fname, lkey)] = (
                        vals.get("sum", 0.0),
                        vals.get("count", 0.0),
                    )

            elif family.type == "gauge":
                for sample in family.samples:
                    # Use (name, labels_key) to support gauge candidates with labels.
                    key = _labels_key(sample.labels)
                    val = sample.value
                    self._gauges[sample.name] = val
                    self._gauges[f"{sample.name}|{key}"] = val

    def get_histogram_sum_count(
        self,
        metric_name: str,
        label_filter: Optional[Dict[str, str]] = None,
    ) -> Optional[Tuple[float, float]]:
        """Return (sum, count) for the histogram, optionally filtered by labels.

        When label_filter is given, only entries whose labels match the filter are summed.
        Otherwise all entries under the family are summed.
        """
        if label_filter is None:
            total_sum = 0.0
            total_count = 0.0
            found = False
            for (fname, _lkey), (s, c) in self._histograms.items():
                if fname == metric_name:
                    total_sum += s
                    total_count += c
                    found = True
            return (total_sum, total_count) if found else None

        # Filtered lookup: merge all matching label sets.
        total_sum = 0.0
        total_count = 0.0
        found = False
        for (fname, lkey), (s, c) in self._histograms.items():
            if fname != metric_name:
                continue
            labels = {}
            for pair in lkey.split(","):
                k, v = pair.split("=", 1)
                labels[k] = v
            if _labels_match(labels, label_filter):
                total_sum += s
                total_count += c
                found = True
        return (total_sum, total_count) if found else None

    def get_gauge(self, name: str) -> Optional[float]:
        return self._gauges.get(name)

    def get_first_gauge(self, candidates: list[str]) -> Optional[float]:
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


# ---------------------------------------------------------------------------
# Differential computation
# ---------------------------------------------------------------------------

def compute_differential_average(
    before: MetricsSnapshot,
    after: MetricsSnapshot,
    metric_name: str,
    label_filter: Optional[Dict[str, str]] = None,
) -> Optional[float]:
    b = before.get_histogram_sum_count(metric_name, label_filter)
    a = after.get_histogram_sum_count(metric_name, label_filter)
    if b is None or a is None:
        return None
    d_sum = a[0] - b[0]
    d_count = a[1] - b[1]
    if d_count <= 0:
        return None
    return d_sum / d_count


def compute_differential_two_metric_div(
    before: MetricsSnapshot,
    after: MetricsSnapshot,
    metric_name_a: str,
    metric_name_b: str,
    label_filter: Optional[Dict[str, str]] = None,
) -> Optional[float]:
    """Compute avg(a - b) = avg(a) - avg(b) using differential sums/counts.

    Only valid when both metrics track the same requests (same d_count).
    Returns None if counts don't match or are zero.
    """
    ba = before.get_histogram_sum_count(metric_name_a, label_filter)
    bb = before.get_histogram_sum_count(metric_name_b, label_filter)
    aa = after.get_histogram_sum_count(metric_name_a, label_filter)
    ab = after.get_histogram_sum_count(metric_name_b, label_filter)
    if None in (ba, bb, aa, ab):
        return None
    d_sum_a = aa[0] - ba[0]
    d_count_a = aa[1] - ba[1]
    d_sum_b = ab[0] - bb[0]
    d_count_b = ab[1] - bb[1]
    if d_count_a <= 0 or d_count_a != d_count_b:
        return None
    return (d_sum_a - d_sum_b) / d_count_a


# ---------------------------------------------------------------------------
# Peak KV Cache tracker
# ---------------------------------------------------------------------------

class PeakKVCacheTracker:
    """Background thread polling /metrics at high frequency to capture peak KV cache."""

    def __init__(self, metrics_url: str, candidates: list[str], interval: float = 0.1):
        self._url = metrics_url
        self._candidates = candidates
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
                val = snap.get_first_gauge(self._candidates)
                if val is not None:
                    self.peak_perc = max(self.peak_perc, val)
            except Exception:
                logger.debug("KV cache poll failed", exc_info=True)
            self._stop.wait(self._interval)
