from __future__ import annotations

import logging
import os
import subprocess
import threading
import time
import requests
from config import ModelConfig

logger = logging.getLogger(__name__)


def _model_mount_volume(model_path: str) -> str:
    """Derive the Docker volume mount from the model path.

    Example: /data/models/Qwen3.5-9B -> -v /data/models:/data/models:ro
    """
    parent = os.path.dirname(os.path.abspath(model_path))
    return f"{parent}:{parent}:ro"


class VLLMContainer:
    """Manage a vLLM or SGLang Docker container via subprocess + docker CLI."""

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
        self._log_thread: threading.Thread | None = None
        self._log_stop: threading.Event | None = None
        sv_type = model_cfg.server_type
        self._name = f"{sv_type}-bench-{model_cfg.name}-{int(time.time())}"

    # --- public API ---

    def start(self, log_file: str | None = None) -> None:
        volume = _model_mount_volume(self._model.path)
        gpus = self._model.visible_gpus
        gpu_arg = f'"device={gpus}"' if gpus != "all" else "all"
        common = [
            "docker", "run", "-d", "--rm",
            "--gpus", gpu_arg,
            "--name", self._name,
            "--shm-size", self._shm_size,
            "-p", f"{self._port}:{self._port}",
            "-v", volume,
            self._model.image,
        ]

        if self._model.server_type == "sglang":
            server_args = self._build_sglang_args()
        else:
            server_args = self._build_vllm_args()

        cmd = common + server_args + self._model.extra_args
        logger.info("Starting container: %s", " ".join(cmd))
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"Docker start failed:\n{result.stderr}")
        self._container_id = result.stdout.strip()
        logger.info("Container started: %s", self._container_id[:12])

        if log_file:
            self._start_log_stream(log_file)

    def _start_log_stream(self, log_file: str) -> None:
        """Start a background thread that streams docker logs -f to a file."""
        self._log_stop = threading.Event()

        def _stream() -> None:
            with open(log_file, "w") as f:
                proc = subprocess.Popen(
                    ["docker", "logs", "-f", "--since", "0s", self._name],
                    stdout=f, stderr=subprocess.STDOUT,
                )
                # Wait until stop is signaled, then kill the log process
                while not self._log_stop.is_set():
                    if proc.poll() is not None:
                        break
                    self._log_stop.wait(1.0)
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()

        self._log_thread = threading.Thread(target=_stream, daemon=True)
        self._log_thread.start()
        logger.info("Log streaming to %s", log_file)

    def _build_vllm_args(self) -> list[str]:
        return [
            "--port", str(self._port),
            "--model", self._model.path,
            "--gpu-memory-utilization", str(self._model.gpu_memory_utilization),
            "--max-model-len", str(self._model.max_model_len),
            "--tensor-parallel-size", str(self._model.tensor_parallel_size),
            "--no-enable-prefix-caching",
        ]

    def _build_sglang_args(self) -> list[str]:
        return [
            "sglang", "serve",
            "--model-path", self._model.path,
            "--host", "0.0.0.0",
            "--port", str(self._port),
            "--mem-fraction-static", str(self._model.gpu_memory_utilization),
            "--context-length", str(self._model.max_model_len),
            "--tp-size", str(self._model.tensor_parallel_size),
            "--enable-metrics",
        ]

    def wait_until_healthy(self) -> None:
        deadline = time.time() + self._health_timeout
        while time.time() < deadline:
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=5)
                if resp.status_code == 200:
                    logger.info("%s healthy at %s", self._model.server_type, self.base_url)
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

        # Stop log streaming
        if self._log_stop:
            self._log_stop.set()
        if self._log_thread:
            self._log_thread.join(timeout=10)

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
            f"Container {self._name} not healthy within {self._health_timeout}s.\n"
            f"--- container logs (last 40 lines) ---\n{logs.stdout}\n{logs.stderr}"
        )
