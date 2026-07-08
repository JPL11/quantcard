"""quantbench command line interface."""

from __future__ import annotations

import click

from .bench import load_entrypoint, run_config
from .configs import CONFIGS
from .report import render_markdown


@click.group()
@click.version_option(package_name="quantbench")
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


if __name__ == "__main__":
    main()
