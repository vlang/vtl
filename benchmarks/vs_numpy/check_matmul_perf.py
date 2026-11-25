#!/usr/bin/env python3
"""Fail the PR benchmark job when optimized VTL GEMM is over 3x slower."""

from __future__ import annotations

import json
import math
import os
import platform
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
SVD_SIZES = (16, 32, 64, 128)
SVD_ROW = re.compile(
    r"^\s*(\d+)x(\d+)\s*\|\s*([0-9.eE+-]+)\s*\|\s*([0-9.eE+-]+)\s*$"
)
STAT_ROW = re.compile(
    r"^\s*(vtl|numpy)_(sum|mean)\s*\|\s*(\d+)\s*\|\s*([0-9.eE+-]+)\s*\|\s*checksum=([0-9.eE+-]+)\s*$"
)
STAT_SIZES = (100_000, 1_000_000)


def find_gflops(report: str, pattern: re.Pattern[str], label: str) -> float:
    matches = [pattern.search(line) for line in report.splitlines()]
    values = [float(match.group(1)) for match in matches if match]
    if len(values) != 1:
        raise ValueError(f"expected one {label} 512x512 result, found {len(values)}")
    if not math.isfinite(values[0]) or values[0] <= 0:
        raise ValueError(f"{label} 512x512 GFLOPS must be finite and positive, got {values[0]}")
    return values[0]


def find_svd_rows(report: str, heading: str, label: str) -> dict[int, tuple[float, float]]:
    marker = f"### {heading}"
    if report.count(marker) != 1:
        raise ValueError(f"expected one {label} SVD section, found {report.count(marker)}")

    section = report.split(marker, maxsplit=1)[1]
    next_heading = re.search(r"^#{2,3} ", section, flags=re.MULTILINE)
    if next_heading is not None:
        section = section[: next_heading.start()]

    results: dict[int, tuple[float, float]] = {}
    for line in section.splitlines():
        match = SVD_ROW.match(line)
        if match is None:
            continue
        rows, columns = int(match.group(1)), int(match.group(2))
        mean_ms, checksum = float(match.group(3)), float(match.group(4))
        if rows != columns or rows not in SVD_SIZES:
            raise ValueError(f"unexpected {label} SVD shape {rows}x{columns}")
        if rows in results:
            raise ValueError(f"duplicate {label} SVD result for {rows}x{rows}")
        if not math.isfinite(mean_ms) or mean_ms <= 0.0:
            raise ValueError(f"{label} SVD time for {rows}x{rows} must be finite and positive")
        if not math.isfinite(checksum):
            raise ValueError(f"{label} SVD checksum for {rows}x{rows} must be finite")
        results[rows] = (mean_ms, checksum)

    if tuple(sorted(results)) != SVD_SIZES:
        raise ValueError(
            f"expected {label} SVD results for sizes {SVD_SIZES}, found {tuple(sorted(results))}"
        )
    return results


def svd_benchmark_group(report: str, backend: str, heading: str) -> dict[str, object]:
    vtl_rows = find_svd_rows(report, heading, f"VTL {backend}")
    numpy_rows = find_svd_rows(report, "NumPy reduced SVD", "NumPy")
    results = []
    for size in SVD_SIZES:
        vtl_ms, vtl_checksum = vtl_rows[size]
        numpy_ms, numpy_checksum = numpy_rows[size]
        slowdown = numpy_ms / vtl_ms
        if not math.isfinite(slowdown):
            raise ValueError(f"SVD slowdown for {backend} at {size}x{size} is not finite")
        results.append(
            {
                "backend": backend,
                "dtype": "f64",
                "shape": [size, size],
                "vtl_mean_ms": vtl_ms,
                "numpy_mean_ms": numpy_ms,
                "numpy_over_vtl": slowdown,
                "vtl_checksum": vtl_checksum,
                "numpy_checksum": numpy_checksum,
            }
        )
    return {
        "benchmark": "reduced-svd",
        "unit": "ms",
        "baseline": "numpy",
        "results": results,
    }


def find_stat_rows(
    report: str, heading: str, source: str
) -> dict[tuple[str, int], tuple[float, float]]:
    marker = f"### {heading}"
    if report.count(marker) != 1:
        raise ValueError(f"expected one {source} statistics section, found {report.count(marker)}")

    section = report.split(marker, maxsplit=1)[1]
    next_heading = re.search(r"^#{2,3} ", section, flags=re.MULTILINE)
    if next_heading is not None:
        section = section[: next_heading.start()]

    results: dict[tuple[str, int], tuple[float, float]] = {}
    for line in section.splitlines():
        match = STAT_ROW.match(line)
        if match is None or match.group(1) != source:
            continue
        operation, size = match.group(2), int(match.group(3))
        mean_ms, checksum = float(match.group(4)), float(match.group(5))
        key = (operation, size)
        if size not in STAT_SIZES:
            raise ValueError(f"unexpected {source} {operation} size {size}")
        if key in results:
            raise ValueError(f"duplicate {source} {operation} result for size {size}")
        if not math.isfinite(mean_ms) or mean_ms <= 0.0:
            raise ValueError(f"{source} {operation} time for size {size} must be finite and positive")
        if not math.isfinite(checksum):
            raise ValueError(f"{source} {operation} checksum for size {size} must be finite")
        results[key] = (mean_ms, checksum)

    expected = {(operation, size) for operation in ("sum", "mean") for size in STAT_SIZES}
    if results.keys() != expected:
        raise ValueError(
            f"expected {source} sum/mean results for sizes {STAT_SIZES}, "
            f"found {tuple(sorted(results))}"
        )
    return results


def statistics_benchmark_group(report: str) -> dict[str, object]:
    vtl_rows = find_stat_rows(report, "VTL contiguous sum/mean", "vtl")
    numpy_rows = find_stat_rows(report, "NumPy contiguous sum/mean", "numpy")
    results = []
    for operation in ("sum", "mean"):
        for size in STAT_SIZES:
            vtl_ms, vtl_checksum = vtl_rows[(operation, size)]
            numpy_ms, numpy_checksum = numpy_rows[(operation, size)]
            checksum_error = abs(vtl_checksum - numpy_checksum)
            checksum_tolerance = 1e-8 * max(1.0, abs(numpy_checksum))
            if checksum_error > checksum_tolerance:
                raise ValueError(
                    f"{operation} checksum mismatch at size {size}: "
                    f"VTL {vtl_checksum}, NumPy {numpy_checksum}"
                )
            slowdown = numpy_ms / vtl_ms
            if not math.isfinite(slowdown):
                raise ValueError(f"{operation} slowdown at size {size} is not finite")
            results.append(
                {
                    "operation": operation,
                    "dtype": "f64",
                    "shape": [size],
                    "vtl_mean_ms": vtl_ms,
                    "numpy_mean_ms": numpy_ms,
                    "numpy_over_vtl": slowdown,
                    "vtl_checksum": vtl_checksum,
                    "numpy_checksum": numpy_checksum,
                    "checksum_abs_error": checksum_error,
                    "correctness_status": "pass",
                }
            )
    return {
        "benchmark": "contiguous-statistics",
        "unit": "ms",
        "baseline": "numpy",
        "results": results,
    }


def environment_metadata() -> dict[str, object]:
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "v_version": os.environ.get("VTL_V_VERSION"),
        "c_compiler": os.environ.get("VTL_C_COMPILER"),
        "numpy_version": os.environ.get("VTL_NUMPY_VERSION"),
    }


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
                "schema_version": 2,
                "environment": environment_metadata(),
                "benchmarks": [
                    {
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
                    },
                    svd_benchmark_group(report, "pure-v", "VTL reduced SVD pure-V"),
                    svd_benchmark_group(report, "lapacke", "VTL reduced SVD LAPACKE"),
                    statistics_benchmark_group(report),
                ],
            }
            json_path.write_text(
                json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
            )

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
