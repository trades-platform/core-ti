"""Generate Markdown benchmark comparison report from pytest-benchmark JSON."""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def load_results(path: str) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    return data.get("benchmarks", [])


def parse_result(bm: dict) -> tuple[str, str, str, float]:
    name = bm["name"]
    # name format: "test_bench_sma[1K-pandas]"
    if "[" not in name:
        return bm.get("group", "unknown"), "unknown", "?", 0.0
    parts = name.split("[")[1].rstrip("]").split("-")
    size = parts[0]
    backend = parts[1] if len(parts) > 1 else "?"
    indicator = bm.get("group", name.split("_")[-1].split("[")[0])
    median_s = bm["stats"]["median"]
    median_ms = median_s * 1_000
    return indicator, backend, size, median_ms


def fmt_time(ms: float) -> str:
    if ms < 0.001:
        return f"{ms * 1_000_000:.1f}ns"
    if ms < 1:
        return f"{ms * 1000:.1f}us"
    if ms < 1000:
        return f"{ms:.2f}ms"
    return f"{ms / 1000:.2f}s"


def _speed_badge(ratio: float) -> str:
    if ratio < 1.2:
        return "🟢"
    if ratio < 3:
        return "🟡"
    if ratio < 10:
        return "🟠"
    return "🔴"


def fmt_cell(ms: float, best_ms: float) -> str:
    t = fmt_time(ms)
    if ms == best_ms:
        return f"🟢 **{t}**"
    ratio = ms / best_ms
    badge = _speed_badge(ratio)
    return f"{badge} {t} ({ratio:.1f}x)"


def generate_report(results: list[dict]) -> str:
    # group: indicator -> size -> [(backend, median_ms)]
    groups: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))

    for bm in results:
        indicator, backend, size, median_ms = parse_result(bm)
        groups[indicator][size].append((backend, median_ms))

    backends = sorted({b for entries in groups.values() for size_entries in entries.values() for b, _ in size_entries})
    lines = ["## Benchmark Results\n"]
    lines.append("Median times per indicator, grouped by data size. **Bold** = fastest.\n")

    for indicator in sorted(groups):
        lines.append(f"### {indicator.upper()}\n")
        header = "| Size | " + " | ".join(b.title() for b in backends) + " |"
        sep = "|------|" + "|".join(["--------" for _ in backends]) + "|"
        lines.append(header)
        lines.append(sep)

        for size in sorted(groups[indicator], key=lambda s: int(s.replace("K", "000"))):
            entries = groups[indicator][size]
            backend_map = {b: ms for b, ms in entries}
            best_ms = min(backend_map.values()) if backend_map else 1.0
            cells = []
            for b in backends:
                if b in backend_map:
                    cells.append(fmt_cell(backend_map[b], best_ms))
                else:
                    cells.append("-")
            lines.append(f"| {size} | " + " | ".join(cells) + " |")

        lines.append("")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("Usage: python report.py <benchmark.json>", file=sys.stderr)
        sys.exit(1)

    report = generate_report(load_results(sys.argv[1]))
    print(report)

    import os
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
