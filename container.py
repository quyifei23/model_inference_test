from __future__ import annotations

import logging
import subprocess
import time
import requests
from config import ModelConfig

logger = logging.getLogger(__name__)


class VLLMContainer:
    """Manage a vLLM Docker container via subprocess + docker CLI."""

    def __init__(
        self,
        model_cfg: ModelConfig,
        port: int = 8000,
        shm_size: str = "4GB",
        health_timeout: int = 1200,
    ):
        self._model = model_cfg
        self._port = port
        self._shm_size = shm_size
        self._health_timeout = health_timeout
        self._container_id: str | None = None
        self._name = f"vllm-bench-{model_cfg.name}-{int(time.time())}"

    # --- public API ---

    def start(self) -> None:
        cmd = [
            "docker", "run", "-d", "--rm",
            "--gpus", "all",
            "--name", self._name,
            "--shm-size", self._shm_size,
            "-p", f"{self._port}:{self._port}",
            "-v", "/data/models:/data/models:ro",
            self._model.image,
            "--port", str(self._port),
            "--model", self._model.path,
            "--gpu-memory-utilization", str(self._model.gpu_memory_utilization),
            "--max-model-len", str(self._model.max_model_len),
            "--tensor-parallel-size", str(self._model.tensor_parallel_size),
            "--no-enable-prefix-caching",
        ] + self._model.extra_args

        logger.info("Starting container: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"Docker start failed:\n{result.stderr}")
        self._container_id = result.stdout.strip()
        logger.info("Container started: %s", self._container_id[:12])

    def wait_until_healthy(self) -> None:
        deadline = time.time() + self._health_timeout
        while time.time() < deadline:
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=5)
                if resp.status_code == 200:
                    logger.info("vLLM healthy at %s", self.base_url)
                    return
            except requests.RequestException:
                pass
            time.sleep(2)
        self._raise_health_timeout()

    def stop(self) -> None:
        if self._container_id is None:
            return
        logger.info("Stopping container %s", self._name)
        subprocess.run(["docker", "stop", self._name], capture_output=True, timeout=60)
        subprocess.run(["docker", "rm", "--force", self._name], capture_output=True, timeout=30)
        self._container_id = None

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self._port}"

    # --- context manager ---

    def __enter__(self):
        self.start()
        self.wait_until_healthy()
        return self

    def __exit__(self, *exc):
        self.stop()
        return False

    # --- helpers ---

    def get_logs(self) -> str:
        """Return container logs (stdout + stderr)."""
        r = subprocess.run(
            ["docker", "logs", self._name],
            capture_output=True, text=True, timeout=30,
        )
        return r.stdout + "\n" + r.stderr if r.stderr else r.stdout

    def _raise_health_timeout(self) -> None:
        logs = subprocess.run(
            ["docker", "logs", self._name, "--tail", "40"],
            capture_output=True, text=True, timeout=10,
        )
        raise RuntimeError(
            f"vLLM container {self._name} not healthy within {self._health_timeout}s.\n"
            f"--- container logs (last 40 lines) ---\n{logs.stdout}\n{logs.stderr}"
        )
