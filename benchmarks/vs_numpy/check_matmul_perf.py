#!/usr/bin/env python3
"""Fail the PR benchmark job when optimized VTL GEMM is over 3x slower."""

from __future__ import annotations

import json
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


def check_report(path: Path, annotate: bool = False, json_path: Path | None = None) -> int:
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

        if json_path is not None:
            payload = {
                "schema_version": 1,
                "benchmark": "cpu-gemm",
                "max_slowdown": MAX_SLOWDOWN,
                "results": [
                    {
                        "dtype": dtype,
                        "shape": [512, 512],
                        "vtl_gflops": vtl_gflops,
                        "numpy_gflops": numpy_gflops,
                        "slowdown": slowdown,
                        "status": "fail" if slowdown > MAX_SLOWDOWN else "pass",
                    }
                    for dtype, vtl_gflops, numpy_gflops, slowdown in results
                ],
            }
            json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

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
    annotate = False
    json_path: Path | None = None
    report_path: Path | None = None
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--annotate" and not annotate:
            annotate = True
            index += 1
        elif argument == "--json-out" and json_path is None and index + 1 < len(arguments):
            json_path = Path(arguments[index + 1])
            index += 2
        elif report_path is None:
            report_path = Path(argument)
            index += 1
        else:
            report_path = None
            break
    if report_path is None or index != len(arguments):
        print(
            f"usage: {Path(sys.argv[0]).name} [--annotate] [--json-out RESULTS.json] REPORT.md",
            file=sys.stderr,
        )
        return 2
    return check_report(report_path, annotate=annotate, json_path=json_path)


if __name__ == "__main__":
    raise SystemExit(main())
