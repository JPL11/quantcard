"""Registry of quantization recipes benchmarked by quantbench.

Each entry maps a short name to a callable that quantizes a model in place
using torchao's `quantize_` API — the same configs ExecuTorch consumes.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def _int8_weight_only(model: torch.nn.Module) -> None:
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    quantize_(model, Int8WeightOnlyConfig())


def _int8_dynamic_activation_int8_weight(model: torch.nn.Module) -> None:
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    quantize_(model, Int8DynamicActivationInt8WeightConfig())


CONFIGS: dict[str, Callable[[torch.nn.Module], None] | None] = {
    "fp32": None,  # baseline, no quantization
    "int8wo": _int8_weight_only,
    "int8da-int8w": _int8_dynamic_activation_int8_weight,
}
