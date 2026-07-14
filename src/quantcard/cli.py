"""quantcard command line interface."""

from __future__ import annotations

import click

from .bench import load_entrypoint, run_config
from .configs import CONFIGS
from .report import render_markdown


@click.group()
@click.version_option(package_name="quantcard")
def main():
    """Quantization report cards for PyTorch models headed to the edge."""


@main.command()
def configs():
    """List available quantization configs."""
    for name in CONFIGS:
        click.echo(name)


@main.command()
@click.argument("entrypoint", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--configs",
    "config_names",
    default="fp32,int8wo,int8da-int8w",
    show_default=True,
    help="Comma-separated config names (first = baseline).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write the markdown report here instead of stdout.",
)
@click.option(
    "--pte",
    is_flag=True,
    help="Add an ExecuTorch .pte export-size column (requires the executorch package).",
)
def report(entrypoint, config_names, output, pte):
    """Benchmark quantization configs against a model ENTRYPOINT.

    ENTRYPOINT is a python file defining get_model() and get_eval_batches().
    """
    if config_names.strip() == "all":
        names = list(CONFIGS)
    else:
        names = [n.strip() for n in config_names.split(",") if n.strip()]
    unknown = [n for n in names if n not in CONFIGS]
    if unknown:
        raise click.BadParameter(f"unknown configs {unknown}; available: {sorted(CONFIGS)}")

    entry = load_entrypoint(entrypoint)
    base_model = entry.get_model()
    batches = list(entry.get_eval_batches())

    results = []
    for name in names:
        click.echo(f"benchmarking {name} ...", err=True)
        results.append(run_config(name, CONFIGS[name], base_model, batches, export_pte=pte))

    text = render_markdown(results, entrypoint)
    if output:
        with open(output, "w") as f:
            f.write(text)
        click.echo(f"wrote {output}", err=True)
    else:
        click.echo(text)


@main.command()
@click.argument("entrypoint", type=click.Path(exists=True, dir_okay=False))
@click.option("--config", "config_name", default="int8wo", show_default=True)
@click.option("--max-acc-drop", type=float, default=1.0, show_default=True,
              help="Fail if accuracy drops more than this many percentage points.")
@click.option("--max-size-ratio", type=float, default=None,
              help="Fail if quantized/baseline size exceeds this ratio.")
def check(entrypoint, config_name, max_acc_drop, max_size_ratio):
    """CI gate: fail (exit 1) if a config regresses accuracy or size.

    Benchmarks fp32 and CONFIG, then applies the thresholds. Intended for
    model-repo CI: `quantcard check model.py --max-acc-drop 0.5`.
    """
    if config_name not in CONFIGS or config_name == "fp32":
        raise click.BadParameter(f"choose a non-baseline config from {sorted(set(CONFIGS) - {'fp32'})}")
    entry = load_entrypoint(entrypoint)
    base_model = entry.get_model()
    batches = list(entry.get_eval_batches())
    base = run_config("fp32", None, base_model, batches)
    result = run_config(config_name, CONFIGS[config_name], base_model, batches)
    if result.error:
        raise click.ClickException(f"{config_name} failed to apply: {result.error}")
    drop_pct = (base.accuracy - result.accuracy) * 100
    ratio = result.size_ratio(base)
    click.echo(f"{config_name}: acc {result.accuracy:.2%} (drop {drop_pct:.2f} pts), size ratio {ratio:.2f}x")
    failures = []
    if drop_pct > max_acc_drop:
        failures.append(f"accuracy drop {drop_pct:.2f} > {max_acc_drop} pts")
    if max_size_ratio is not None and ratio > max_size_ratio:
        failures.append(f"size ratio {ratio:.2f} > {max_size_ratio}")
    if failures:
        raise click.ClickException("; ".join(failures))
    click.echo("ok")


@main.command()
@click.argument("entrypoint", type=click.Path(exists=True, dir_okay=False))
@click.option("--config", "config_name", default="int8wo", show_default=True)
def layers(entrypoint, config_name):
    """Per-layer sensitivity: quantize one layer at a time, rank by accuracy.

    Shows which layers are safe to quantize and which to keep in fp32.
    """
    from .bench import layer_sensitivity

    fn = CONFIGS.get(config_name)
    if fn is None:
        raise click.BadParameter(f"choose a non-baseline config from {sorted(set(CONFIGS) - {'fp32'})}")
    entry = load_entrypoint(entrypoint)
    base_acc, rows = layer_sensitivity(fn, entry.get_model(), entry.get_eval_batches())
    click.echo(f"baseline accuracy: {base_acc:.2%}   (config: {config_name})")
    click.echo(f"{'layer':<32} {'acc delta':>10}")
    for fqn, delta, err in sorted(rows, key=lambda r: r[1]):
        click.echo(f"{fqn:<32} {delta:>+10.2%}" + (f"  [{err}]" if err else ""))


if __name__ == "__main__":
    main()
