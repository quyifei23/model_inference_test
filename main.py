from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, Any, List

from config import load_config, BenchmarkConfig, ModelConfig, WorkloadConfig
from container import VLLMContainer
from metrics import (
    MetricsSnapshot,
    compute_differential_average,
    compute_differential_two_metric_div,
    PeakKVCacheTracker,
    _prefill_metric,
    _prefill_label_filter,
    _ttft_metric,
    _decode_metric,
    _e2e_metric,
    _cache_gauge_candidates,
)
from benchmark import PromptGenerator, run_workload_burst, warmup

logger = logging.getLogger("benchmark")


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def get_gpu_memory_mib(tensor_parallel_size: int = 1, visible_gpus: str = "all") -> list[int]:
    """Query total GPU memory for the relevant devices (MiB each)."""
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10,
    )
    # Parse: "0, 40960 MiB" -> {0: 40960, 1: 40960, ...}
    all_mem = {}
    for line in r.stdout.strip().split("\n"):
        parts = line.split(",")
        idx = int(parts[0].strip())
        mem = int(parts[1].strip().split()[0])
        all_mem[idx] = mem

    if visible_gpus == "all":
        gpu_ids = sorted(all_mem.keys())
    else:
        gpu_ids = [int(x.strip()) for x in visible_gpus.split(",")]

    if len(gpu_ids) < tensor_parallel_size:
        raise RuntimeError(
            f"Only {len(gpu_ids)} GPUs available (visible_gpus={visible_gpus}), "
            f"but tensor_parallel_size={tensor_parallel_size}"
        )
    return [all_mem[i] for i in gpu_ids[:tensor_parallel_size]]


def get_model_weight_bytes(model_path: str) -> int:
    """Read from model.safetensors.index.json or sum .safetensors files."""
    idx = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            meta = json.load(f).get("metadata", {}) or {}
        total = meta.get("total_size")
        if total:
            return int(total)
    # Fallback: sum weight files
    total = 0
    for fn in os.listdir(model_path):
        if fn.endswith(".safetensors"):
            total += os.path.getsize(os.path.join(model_path, fn))
    return total


def compute_total_kv_cache_bytes(
    model_path: str,
    gpu_memory_utilization: float,
    gpu_memories_mib: list[int],
) -> int:
    """KV Cache = (total_gpu_memory - model_weights) * gpu_memory_utilization."""
    total_gpu_bytes = sum(gpu_memories_mib) * 1024 * 1024
    weight_bytes = get_model_weight_bytes(model_path)
    return max(int((total_gpu_bytes - weight_bytes) * gpu_memory_utilization), 0)


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_orphans() -> None:
    for prefix in ("vllm-bench-", "sglang-bench-"):
        r = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={prefix}", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=10,
        )
        for name in r.stdout.strip().split("\n"):
            if name:
                logger.warning("Cleaning up orphan container: %s", name)
                subprocess.run(["docker", "stop", name], capture_output=True, timeout=30)
                subprocess.run(["docker", "rm", "--force", name], capture_output=True, timeout=15)


# ---------------------------------------------------------------------------
# Workload runner
# ---------------------------------------------------------------------------

def run_single_workload(
    base_url: str,
    model_name: str,
    wl: WorkloadConfig,
    generator: PromptGenerator,
    metrics_url: str,
    request_timeout: int,
    poll_interval: float,
    server_type: str,
) -> Dict[str, Any]:
    prompt = generator.generate(wl.input_tokens)
    actual_tokens = len(generator._tokenizer.encode(prompt).ids)
    logger.info(
        "Workload in=%d out=%d con=%d  actual_prompt_tokens=%d",
        wl.input_tokens, wl.output_tokens, wl.concurrency, actual_tokens,
    )

    before = MetricsSnapshot.fetch(metrics_url)

    candidates = _cache_gauge_candidates(server_type)
    tracker = PeakKVCacheTracker(metrics_url, candidates=candidates, interval=poll_interval)
    tracker.start()

    t0 = time.monotonic()
    result = asyncio.run(
        run_workload_burst(
            base_url, model_name, prompt,
            max_tokens=wl.output_tokens,
            concurrency=wl.concurrency,
            timeout_seconds=request_timeout,
        )
    )
    elapsed = time.monotonic() - t0

    time.sleep(2.0)

    tracker.stop()
    after = MetricsSnapshot.fetch(metrics_url)

    prefill_filter = _prefill_label_filter(server_type)
    avg_prefill = compute_differential_average(
        before, after, _prefill_metric(server_type), label_filter=prefill_filter,
    )
    avg_ttft = compute_differential_average(
        before, after, _ttft_metric(server_type),
    )

    decode_metric = _decode_metric(server_type)
    if decode_metric:
        avg_decode = compute_differential_average(before, after, decode_metric)
    else:
        # SGLang: decode = e2e - ttft
        avg_decode = compute_differential_two_metric_div(
            before, after,
            _e2e_metric(server_type), _ttft_metric(server_type),
        )

    return {
        "input_tokens": wl.input_tokens,
        "output_tokens": wl.output_tokens,
        "concurrency": wl.concurrency,
        "avg_prefill_ms": (avg_prefill * 1000) if avg_prefill is not None else None,
        "avg_ttft_ms": (avg_ttft * 1000) if avg_ttft is not None else None,
        "avg_decode_ms": (avg_decode * 1000) if avg_decode is not None else None,
        "peak_kv_cache_perc": tracker.peak_perc,
        "num_success": result["num_success"],
        "num_failed": result["num_failed"],
        "burst_duration_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_all(cfg: BenchmarkConfig) -> List[Dict[str, Any]]:
    all_results: List[Dict[str, Any]] = []

    for model_cfg in cfg.models:
        logger.info("===== Model: %s =====", model_cfg.name)

        gpu_memories_mib = get_gpu_memory_mib(
            model_cfg.tensor_parallel_size, model_cfg.visible_gpus,
        )
        total_gpu_gb = sum(gpu_memories_mib) / 1024
        logger.info("GPUs: %d × %d MiB = %.1f GB total",
                     len(gpu_memories_mib), gpu_memories_mib[0], total_gpu_gb)

        total_cache_bytes = compute_total_kv_cache_bytes(
            model_cfg.path, model_cfg.gpu_memory_utilization, gpu_memories_mib,
        )
        total_cache_gb = total_cache_bytes / (1024 ** 3)
        weight_gb = get_model_weight_bytes(model_cfg.path) / (1024 ** 3)
        logger.info(
            "weights=%.1f GB  gpu_util=%.2f  kv_cache_budget=%.1f GB",
            weight_gb, model_cfg.gpu_memory_utilization, total_cache_gb,
        )

        if total_cache_bytes <= 0:
            logger.critical("KV cache budget <= 0, skipping model %s", model_cfg.name)
            continue

        os.makedirs(cfg.output_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(cfg.output_dir, f"{model_cfg.name}_{ts}.log")

        container = VLLMContainer(
            model_cfg,
            port=cfg.vllm_port,
            shm_size=cfg.shm_size,
            health_timeout=cfg.health_timeout_seconds,
        )
        container.start(log_file=log_path)
        try:
            container.wait_until_healthy()

            base_url = container.base_url
            metrics_url = f"{base_url}/metrics"
            model_name = model_cfg.path

            generator = PromptGenerator(model_cfg.path)

            # Global warmup
            warmup(base_url, model_name)

            model_results: List[Dict[str, Any]] = []
            total = len(cfg.workloads)
            for i, wl in enumerate(cfg.workloads):
                print(f"\n{'='*60}")
                print(f"[{i+1}/{total}] in={wl.input_tokens} out={wl.output_tokens} concurrency={wl.concurrency}")
                print(f"{'='*60}")
                row = run_single_workload(
                    base_url=base_url,
                    model_name=model_name,
                    wl=wl,
                    generator=generator,
                    metrics_url=metrics_url,
                    request_timeout=cfg.request_timeout_seconds,
                    poll_interval=cfg.metrics_poll_interval,
                    server_type=model_cfg.server_type,
                )
                row["model"] = model_cfg.name
                row["tensor_parallel_size"] = model_cfg.tensor_parallel_size
                row["total_kv_cache_gb"] = total_cache_gb
                row["gpu_memory_gb"] = total_gpu_gb
                row["model_weight_gb"] = weight_gb
                row["gpu_memory_utilization"] = model_cfg.gpu_memory_utilization
                row["max_model_len"] = model_cfg.max_model_len
                row["peak_kv_cache_gb"] = row["peak_kv_cache_perc"] * total_cache_gb

                model_results.append(row)
                _print_row(row)

            all_results.append({
                "model": model_cfg.name,
                "model_path": model_cfg.path,
                "docker_image": model_cfg.image,
                "tensor_parallel_size": model_cfg.tensor_parallel_size,
                "gpu_memory_gb": total_gpu_gb,
                "total_kv_cache_gb": total_cache_gb,
                "model_weight_gb": weight_gb,
                "results": model_results,
            })

        finally:
            container.stop()

    return all_results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _print_row(r: Dict[str, Any]) -> None:
    prefill = f"{r['avg_prefill_ms']:.1f}" if r["avg_prefill_ms"] is not None else "N/A"
    ttft = f"{r['avg_ttft_ms']:.1f}" if r["avg_ttft_ms"] is not None else "N/A"
    decode = f"{r['avg_decode_ms']:.1f}" if r["avg_decode_ms"] is not None else "N/A"
    print(
        f"  {r['model']:20s}"
        f" in={r['input_tokens']:5d} out={r['output_tokens']:5d} con={r['concurrency']:2d}"
        f" | prefill={prefill:>8s} ms  ttft={ttft:>8s} ms  decode={decode:>8s} ms"
        f" | kv_peak={r['peak_kv_cache_gb']:.2f} GB ({r['peak_kv_cache_perc']*100:.1f}%)"
        f" | ok={r['num_success']}/{r['num_success']+r['num_failed']}"
    )


def save_results(all_results: List[Dict[str, Any]], output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    jpath = os.path.join(output_dir, f"results_{ts}.json")
    with open(jpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    cpath = os.path.join(output_dir, f"results_{ts}.csv")
    with open(cpath, "w") as f:
        f.write(
            "model,tensor_parallel_size,input_tokens,output_tokens,concurrency,"
            "avg_prefill_ms,avg_ttft_ms,avg_decode_ms,"
            "peak_kv_cache_perc,peak_kv_cache_gb,total_kv_cache_gb,"
            "gpu_memory_gb,model_weight_gb,gpu_memory_utilization,max_model_len,"
            "num_success,num_failed,burst_duration_seconds\n"
        )
        for entry in all_results:
            for r in entry["results"]:
                f.write(
                    f"{r['model']},{r['tensor_parallel_size']},"
                    f"{r['input_tokens']},{r['output_tokens']},"
                    f"{r['concurrency']},"
                    f"{r['avg_prefill_ms']},{r['avg_ttft_ms']},{r['avg_decode_ms']},"
                    f"{r['peak_kv_cache_perc']},{r['peak_kv_cache_gb']},"
                    f"{r['total_kv_cache_gb']},{r['gpu_memory_gb']},"
                    f"{r['model_weight_gb']},{r['gpu_memory_utilization']},"
                    f"{r['max_model_len']},"
                    f"{r['num_success']},{r['num_failed']},"
                    f"{r['burst_duration_seconds']}\n"
                )

    logger.info("Results: %s | %s", jpath, cpath)
    return jpath


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="vLLM KV Cache & Latency Benchmark")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    _setup_logging(args.verbose)
    cleanup_orphans()

    cfg = load_config(args.config)
    cfg.output_dir = args.output_dir

    all_results = run_all(cfg)
    save_results(all_results, cfg.output_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
