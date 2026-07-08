"""Markdown report rendering."""

from __future__ import annotations

from .bench import BenchResult


def render_markdown(results: list[BenchResult], entry_path: str) -> str:
    baseline = results[0]
    lines = [
        "# quantbench report",
        "",
        f"Model entrypoint: `{entry_path}`",
        "",
        "| config | top-1 acc | acc delta | weights | size ratio | latency (ms) | speedup |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.error:
            lines.append(f"| {r.name} | — | — | — | — | — | failed: {r.error} |")
            continue
        lines.append(
            f"| {r.name} | {r.accuracy:.2%} | {r.accuracy_delta(baseline):+.2%} "
            f"| {r.size_bytes / 1024:.0f} KiB | {r.size_ratio(baseline):.2f}x "
            f"| {r.latency_ms:.2f} | {r.speedup(baseline):.2f}x |"
        )
    lines.append("")
    lines.append(
        "Baseline is the first row. Sizes are serialized state_dicts; "
        "latency is median single-batch CPU inference."
    )
    return "\n".join(lines) + "\n"
