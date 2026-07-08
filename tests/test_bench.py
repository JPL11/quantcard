import os

import torch
from click.testing import CliRunner

from quantbench.bench import BenchResult, load_entrypoint, run_config
from quantbench.cli import main
from quantbench.configs import CONFIGS
from quantbench.report import render_markdown

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples", "tiny_mlp.py")


def _tiny_model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))


def _batches():
    torch.manual_seed(1)
    x = torch.randn(64, 8)
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
