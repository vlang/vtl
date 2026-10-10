#!/usr/bin/env python3
"""Fail the PR benchmark job when optimized VTL GEMM is over 3x slower."""

from __future__ import annotations

import re
import sys
from pathlib import Path


MAX_SLOWDOWN = 3.0
CASES = (
    (
        "f64",
        re.compile(r"^\s*gemm\s*\|\s*512x512\s*\|[^|]*\|\s*([0-9.eE+-]+)\s*$"),
        re.compile(r"^numpy gemm 512x512\s*\|[^|]*\|\s*([0-9.eE+-]+)\s+GFLOPS\b"),
    ),
    (
        "f32",
        re.compile(r"^\s*gemm_f32\s*\|\s*512x512\s*\|[^|]*\|\s*([0-9.eE+-]+)\s*$"),
        re.compile(r"^numpy gemm_f32 512x512\s*\|[^|]*\|\s*([0-9.eE+-]+)\s+GFLOPS\b"),
    ),
)


def find_gflops(report: str, pattern: re.Pattern[str], label: str) -> float:
    matches = [pattern.search(line) for line in report.splitlines()]
    values = [float(match.group(1)) for match in matches if match]
    if len(values) != 1:
        raise ValueError(f"expected one {label} 512x512 result, found {len(values)}")
    if values[0] <= 0:
        raise ValueError(f"{label} 512x512 GFLOPS must be positive, got {values[0]}")
    return values[0]


def check_report(path: Path, annotate: bool = False) -> int:
    try:
        report = path.read_text(encoding="utf-8")
        failures: list[str] = []
        results: list[tuple[str, float, float, float]] = []
        for dtype, vtl_pattern, numpy_pattern in CASES:
            vtl_gflops = find_gflops(report, vtl_pattern, f"VTL {dtype}")
            numpy_gflops = find_gflops(report, numpy_pattern, f"NumPy {dtype}")
            slowdown = numpy_gflops / vtl_gflops
            results.append((dtype, vtl_gflops, numpy_gflops, slowdown))
            print(
                f"{dtype} 512x512: VTL {vtl_gflops:.2f} GFLOPS, "
                f"NumPy {numpy_gflops:.2f} GFLOPS, slowdown {slowdown:.2f}x"
            )
            if slowdown > MAX_SLOWDOWN:
                failures.append(f"{dtype} slowdown {slowdown:.2f}x exceeds {MAX_SLOWDOWN:.1f}x")

        if annotate:
            summary = [
                "## CPU GEMM performance budget",
                "",
                "| dtype | shape | VTL CBLAS GFLOPS | NumPy GFLOPS | NumPy / VTL | budget |",
                "| --- | ---: | ---: | ---: | ---: | --- |",
            ]
            for dtype, vtl_gflops, numpy_gflops, slowdown in results:
                status = "FAIL" if slowdown > MAX_SLOWDOWN else "PASS"
                summary.append(
                    f"| {dtype} | 512×512 | {vtl_gflops:.2f} | {numpy_gflops:.2f} "
                    f"| {slowdown:.2f}× | {status} |"
                )
            path.write_text(report.rstrip() + "\n\n" + "\n".join(summary) + "\n", encoding="utf-8")
            return 0

        if failures:
            message = "CPU GEMM performance budget failed: " + "; ".join(failures)
            print(message, file=sys.stderr)
            print(
                "A maintainer may apply the performance-exempt label with a rationale.",
                file=sys.stderr,
            )
            return 1
        return 0
    except (OSError, ValueError) as error:
        print(f"cannot evaluate CPU GEMM performance budget: {error}", file=sys.stderr)
        return 2


def main() -> int:
    arguments = sys.argv[1:]
    annotate = arguments[:1] == ["--annotate"]
    report_arguments = arguments[1:] if annotate else arguments
    if len(report_arguments) != 1:
        print(f"usage: {Path(sys.argv[0]).name} [--annotate] REPORT.md", file=sys.stderr)
        return 2
    return check_report(Path(report_arguments[0]), annotate=annotate)


if __name__ == "__main__":
    raise SystemExit(main())
