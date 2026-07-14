"""Registry of quantization recipes benchmarked by quantcard.

Each entry maps a short name to a callable that quantizes a model in place
using torchao's `quantize_` API — the same configs ExecuTorch consumes.
Configs that need features missing from the installed torchao raise inside
the callable; `run_config` isolates that into the report's error column.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def _int8_weight_only(model: torch.nn.Module, filter_fn=None) -> None:
    from torchao.quantization import Int8WeightOnlyConfig, quantize_

    quantize_(model, Int8WeightOnlyConfig(), filter_fn=filter_fn)


def _int8_dynamic_activation_int8_weight(model: torch.nn.Module, filter_fn=None) -> None:
    from torchao.quantization import Int8DynamicActivationInt8WeightConfig, quantize_

    quantize_(model, Int8DynamicActivationInt8WeightConfig(), filter_fn=filter_fn)


def _int4_weight_only(model: torch.nn.Module, filter_fn=None) -> None:
    """int4 weight-only in the unpacked layout ExecuTorch lowers from."""
    from torchao.quantization import IntxWeightOnlyConfig, quantize_

    quantize_(model, IntxWeightOnlyConfig(weight_dtype=torch.int4), filter_fn=filter_fn)


def _int8_dynamic_activation_int4_weight(model: torch.nn.Module, filter_fn=None) -> None:
    """The classic ExecuTorch 8da4w recipe."""
    from torchao.quantization import Int8DynamicActivationIntxWeightConfig, quantize_

    quantize_(model, Int8DynamicActivationIntxWeightConfig(weight_dtype=torch.int4), filter_fn=filter_fn)


def _qat_prepare(base_config_factory: Callable[[], object]) -> Callable[[torch.nn.Module], None]:
    """QAT prepare step: swaps in fake-quantized modules.

    Measures what the fake-quant numerics cost *before* any training —
    the accuracy delta a QAT run starts from.
    """

    def apply(model: torch.nn.Module) -> None:
        from torchao.quantization import quantize_
        from torchao.quantization.qat import QATConfig

        quantize_(model, QATConfig(base_config_factory(), step="prepare"))

    return apply


def _intx_base(weight_dtype):
    def factory():
        from torchao.quantization import Int8DynamicActivationIntxWeightConfig

        return Int8DynamicActivationIntxWeightConfig(weight_dtype=weight_dtype)

    return factory


CONFIGS: dict[str, Callable[[torch.nn.Module], None] | None] = {
    "fp32": None,  # baseline, no quantization
    "int8wo": _int8_weight_only,
    "int8da-int8w": _int8_dynamic_activation_int8_weight,
    "int4wo": _int4_weight_only,
    "int8da-int4w": _int8_dynamic_activation_int4_weight,
    "qat-int8da-int4w": _qat_prepare(_intx_base(torch.int4)),
}
