from __future__ import annotations

import asyncio
import logging
import os
import random
import time
from typing import Dict, Any

from openai import AsyncOpenAI
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)


class PromptGenerator:
    """Generate random text that tokenises to exactly `num_tokens` tokens."""

    def __init__(self, model_path: str):
        tok_path = self._find_tokenizer(model_path)
        logger.info("Loading tokenizer from %s", tok_path)
        self._tokenizer = Tokenizer.from_file(tok_path)
        self._vocab_size = self._tokenizer.get_vocab_size()
        # Clamp random range to avoid special-token edge cases.
        self._safe_vocab = min(self._vocab_size, 64000)

    @staticmethod
    def _find_tokenizer(model_path: str) -> str:
        candidates = [
            os.path.join(model_path, "tokenizer.json"),
            os.path.join(model_path, "tokenizer", "tokenizer.json"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        raise FileNotFoundError(
            f"No tokenizer.json found at {candidates}. "
            f"Ensure the model directory contains a HuggingFace tokenizer."
        )

    def generate(self, num_tokens: int) -> str:
        target = num_tokens
        tolerance = max(1, int(num_tokens * 0.02))

        for _ in range(5):
            ids = [random.randint(0, self._safe_vocab - 1) for _ in range(target)]
            text = self._tokenizer.decode(ids)
            actual = len(self._tokenizer.encode(text).ids)
            if abs(actual - num_tokens) <= tolerance:
                return text
            # Adjust target for next attempt
            target = target + (num_tokens - actual)
            target = max(1, target)

        return text  # best-effort after retries


async def run_workload_burst(
    base_url: str,
    model_name: str,
    prompt_text: str,
    max_tokens: int,
    concurrency: int,
    timeout_seconds: int = 1200,
) -> Dict[str, Any]:
    """Fire `concurrency` completion requests simultaneously (burst)."""

    client = AsyncOpenAI(
        base_url=f"{base_url}/v1",
        api_key="skip",
        timeout=float(timeout_seconds),
    )

    try:
        async def _one(req_id: int) -> Dict[str, Any]:
            t0 = time.monotonic()
            try:
                resp = await client.completions.create(
                    model=model_name,
                    prompt=prompt_text,
                    max_tokens=max_tokens,
                    temperature=0.0,
                    extra_body={"ignore_eos": True},
                )
                elapsed = time.monotonic() - t0
                u = resp.usage
                return {
                    "request_id": req_id,
                    "success": True,
                    "latency_seconds": elapsed,
                    "prompt_tokens": u.prompt_tokens if u else 0,
                    "completion_tokens": u.completion_tokens if u else 0,
                    "error": None,
                }
            except Exception as exc:
                elapsed = time.monotonic() - t0
                logger.warning("Request %d failed after %.1fs: %s", req_id, elapsed, exc)
                return {
                    "request_id": req_id,
                    "success": False,
                    "latency_seconds": elapsed,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "error": str(exc),
                }

        tasks = [_one(i) for i in range(concurrency)]
        results = await asyncio.gather(*tasks)

        ok = [r for r in results if r["success"]]
        bad = [r for r in results if not r["success"]]
        return {
            "total": concurrency,
            "num_success": len(ok),
            "num_failed": len(bad),
            "results": results,
            "total_duration_seconds": max((r["latency_seconds"] for r in results), default=0.0),
        }
    finally:
        await client.close()


def warmup(base_url: str, model_name: str, prompt: str = "Hello, how are you?") -> None:
    """Send a single small request to warm up CUDA kernels and lazy init."""
    result = asyncio.run(
        run_workload_burst(base_url, model_name, prompt, max_tokens=64, concurrency=1)
    )
    logger.info(
        "Warmup: %d/%d ok, %.1fs",
        result["num_success"],
        result["total"],
        result["total_duration_seconds"],
    )
