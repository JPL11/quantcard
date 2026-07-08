"""Markdown report rendering."""

from __future__ import annotations

from .bench import BenchResult


def _fmt_pte(r: BenchResult) -> str:
    if r.pte_bytes is not None:
        return f"{r.pte_bytes / 1024:.0f} KiB"
    if r.pte_error is not None:
        return f"export failed ({r.pte_error.split(':')[0]})"
    return "—"


def render_markdown(results: list[BenchResult], entry_path: str) -> str:
    baseline = results[0]
    with_pte = any(r.pte_bytes is not None or r.pte_error is not None for r in results)

    header = "| config | top-1 acc | acc delta | weights | size ratio | latency (ms) | speedup |"
    rule = "|---|---|---|---|---|---|---|"
    if with_pte:
        header += " .pte |"
        rule += "---|"

    lines = [
        "# quantbench report",
        "",
        f"Model entrypoint: `{entry_path}`",
        "",
        header,
        rule,
    ]
    for r in results:
        if r.error:
            cells = f"| {r.name} | — | — | — | — | — | failed: {r.error} |"
            lines.append(cells + (" — |" if with_pte else ""))
            continue
        row = (
            f"| {r.name} | {r.accuracy:.2%} | {r.accuracy_delta(baseline):+.2%} "
            f"| {r.size_bytes / 1024:.0f} KiB | {r.size_ratio(baseline):.2f}x "
            f"| {r.latency_ms:.2f} | {r.speedup(baseline):.2f}x |"
        )
        if with_pte:
            row += f" {_fmt_pte(r)} |"
        lines.append(row)
    lines.append("")
    lines.append(
        "Baseline is the first row. Sizes are serialized state_dicts; "
        "latency is median single-batch CPU inference."
    )
    if with_pte:
        lines.append(".pte is the ExecuTorch-exported program size.")
    return "\n".join(lines) + "\n"
