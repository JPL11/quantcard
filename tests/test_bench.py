import os

import torch
from click.testing import CliRunner

from quantbench.bench import BenchResult, load_entrypoint, run_config
from quantbench.cli import main
from quantbench.configs import CONFIGS
from quantbench.report import render_markdown

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples", "tiny_mlp.py")


def _tiny_model():
    # Feature dims are multiples of 32 so per-group int4 configs apply,
    # and large enough that quantization actually shrinks the weights.
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2))


def _batches():
    torch.manual_seed(1)
    x = torch.randn(64, 64)
    y = (x.sum(dim=-1) > 0).long()
    return [(x, y)]


def test_run_config_baseline():
    result = run_config("fp32", None, _tiny_model(), _batches())
    assert result.error is None
    assert 0.0 <= result.accuracy <= 1.0
    assert result.size_bytes > 0
    assert result.latency_ms > 0


def test_run_config_int8wo_close_to_baseline():
    model = _tiny_model()
    batches = _batches()
    base = run_config("fp32", None, model, batches)
    quant = run_config("int8wo", CONFIGS["int8wo"], model, batches)
    assert quant.error is None
    # int8 weight-only on a tiny model should barely move accuracy.
    assert abs(quant.accuracy_delta(base)) <= 0.05


def test_run_config_survives_failure():
    def broken(model):
        raise RuntimeError("nope")

    result = run_config("broken", broken, _tiny_model(), _batches())
    assert result.error is not None and "nope" in result.error


def test_render_markdown():
    base = BenchResult("fp32", 0.9, 1000, 2.0)
    quant = BenchResult("int8wo", 0.89, 300, 1.0)
    text = render_markdown([base, quant], "model.py")
    assert "| int8wo | 89.00% | -1.00% | 0 KiB | 0.30x | 1.00 | 2.00x |" in text


def test_load_entrypoint_validates(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("x = 1\n")
    try:
        load_entrypoint(str(bad))
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "get_model" in str(e)


def test_cli_report_runs_end_to_end(tmp_path):
    out = tmp_path / "report.md"
    runner = CliRunner()
    result = runner.invoke(main, ["report", EXAMPLE, "--configs", "fp32,int8wo", "-o", str(out)])
    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert "| fp32 |" in text and "| int8wo |" in text


def test_int4_configs_quantize():
    model = _tiny_model()
    batches = _batches()
    base = run_config("fp32", None, model, batches)
    for name in ("int4wo", "int8da-int4w"):
        result = run_config(name, CONFIGS[name], model, batches)
        assert result.error is None, result.error
        assert result.size_bytes < base.size_bytes  # weights actually shrank
        assert abs(result.accuracy_delta(base)) <= 0.15


def test_qat_prepare_config():
    model = _tiny_model()
    batches = _batches()
    base = run_config("fp32", None, model, batches)
    result = run_config("qat-int8da-int4w", CONFIGS["qat-int8da-int4w"], model, batches)
    assert result.error is None, result.error
    # Fake quantization keeps high-precision weights: size stays ~baseline
    # (small fake-quantizer metadata overhead is fine, shrinking is not).
    assert result.size_bytes >= 0.95 * base.size_bytes
    assert abs(result.accuracy_delta(base)) <= 0.15


def test_pte_column_reports_error_or_size():
    result = run_config("fp32", None, _tiny_model(), _batches(), export_pte=True)
    assert result.error is None
    # With executorch installed we get a size; without it, a clear error.
    assert (result.pte_bytes is not None) != (result.pte_error is not None)
    if result.pte_error is not None:
        assert "executorch" in result.pte_error


def test_render_markdown_with_pte_column():
    base = BenchResult("fp32", 0.9, 1000, 2.0, pte_bytes=2048)
    quant = BenchResult("int8wo", 0.89, 300, 1.0, pte_error="ImportError: nope")
    text = render_markdown([base, quant], "model.py")
    header = next(line for line in text.splitlines() if line.startswith("| config"))
    assert header.endswith("| .pte |")
    assert "2 KiB" in text
    assert "export failed (ImportError)" in text
