from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    name: str
    path: str
    image: str
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 8192
    tensor_parallel_size: int = 1
    server_type: str = "vllm"   # "vllm" or "sglang"
    extra_args: List[str] = field(default_factory=list)


@dataclass
class WorkloadConfig:
    input_tokens: int
    output_tokens: int
    concurrency: int


@dataclass
class BenchmarkConfig:
    models: List[ModelConfig]
    workloads: List[WorkloadConfig]
    metrics_poll_interval: float = 0.1
    vllm_port: int = 8000
    health_timeout_seconds: int = 600
    request_timeout_seconds: int = 1200
    shm_size: str = "4GB"
    output_dir: str = "output"


def load_config(path: str = "config.yaml") -> BenchmarkConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    models = [ModelConfig(**m) for m in raw["models"]]
    workloads = [WorkloadConfig(**w) for w in raw["workloads"]]

    overrides = {k: v for k, v in raw.items() if k not in ("models", "workloads")}
    return BenchmarkConfig(models=models, workloads=workloads, **overrides)
