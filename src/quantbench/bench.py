"""Core benchmarking: apply quantization configs, measure what changed."""

from __future__ import annotations

import copy
import importlib.util
import io
import statistics
import time
from dataclasses import dataclass

import torch


@dataclass
class BenchResult:
    name: str
    accuracy: float
    size_bytes: int
    latency_ms: float
    error: str | None = None

    def accuracy_delta(self, baseline: BenchResult) -> float:
        return self.accuracy - baseline.accuracy

    def size_ratio(self, baseline: BenchResult) -> float:
        return self.size_bytes / max(baseline.size_bytes, 1)

    def speedup(self, baseline: BenchResult) -> float:
        return baseline.latency_ms / max(self.latency_ms, 1e-9)


def load_entrypoint(path: str):
    """Load a user model file defining `get_model()` and `get_eval_batches()`.

    `get_model()` returns an eval-mode-able `nn.Module`; `get_eval_batches()`
    returns an iterable of `(inputs, targets)` tuples.
    """
    spec = importlib.util.spec_from_file_location("quantbench_entry", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import entrypoint {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for required in ("get_model", "get_eval_batches"):
        if not hasattr(module, required):
            raise ValueError(f"entrypoint {path} must define {required}()")
    return module


@torch.no_grad()
def measure_accuracy(model: torch.nn.Module, batches) -> float:
    """Top-1 accuracy over the eval batches."""
    correct = total = 0
    model.eval()
    for inputs, targets in batches:
        preds = model(inputs).argmax(dim=-1)
        correct += int((preds == targets).sum())
        total += int(targets.numel())
    if total == 0:
        raise ValueError("get_eval_batches() yielded no data")
    return correct / total


def measure_size(model: torch.nn.Module) -> int:
    """Serialized state_dict size in bytes (proxy for on-device weight size)."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module, example_input, warmup: int = 5, runs: int = 20
) -> float:
    """Median single-batch CPU latency in milliseconds."""
    model.eval()
    for _ in range(warmup):
        model(example_input)
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        model(example_input)
        samples.append((time.perf_counter() - start) * 1000)
    return statistics.median(samples)


def run_config(name, quantize_fn, base_model, batches) -> BenchResult:
    """Deep-copy the model, apply a quantization config, measure everything."""
    model = copy.deepcopy(base_model)
    try:
        if quantize_fn is not None:
            quantize_fn(model)
        batches = list(batches)
        example_input = batches[0][0]
        return BenchResult(
            name=name,
            accuracy=measure_accuracy(model, batches),
            size_bytes=measure_size(model),
            latency_ms=measure_latency(model, example_input),
        )
    except Exception as e:  # config not applicable to this model
        return BenchResult(
            name=name, accuracy=0.0, size_bytes=0, latency_ms=0.0, error=f"{type(e).__name__}: {e}"
        )
