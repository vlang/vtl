"""Tests for the PR CPU GEMM performance budget parser."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from check_matmul_perf import check_report, main


def make_report(vtl_f64: float, vtl_f32: float) -> str:
    return f"""gemm | 512x512 | 3.0 | {vtl_f64}
gemm_f32 | 512x512 | 2.0 | {vtl_f32}
numpy gemm 512x512 | 2.5 ms | 100.0 GFLOPS | 1.0
numpy gemm_f32 512x512 | 1.8 ms | 210.0 GFLOPS | 1.0
### VTL reduced SVD pure-V
16x16    | 1.0 | 10.0
32x32    | 2.0 | 20.0
64x64    | 3.0 | 30.0
128x128  | 4.0 | 40.0
### VTL reduced SVD LAPACKE
16x16    | 0.5 | 10.0
32x32    | 1.0 | 20.0
64x64    | 1.5 | 30.0
128x128  | 2.0 | 40.0
### NumPy reduced SVD
16x16    | 0.25 | 10.0
32x32    | 0.5 | 20.0
64x64    | 0.75 | 30.0
128x128  | 1.0 | 40.0
"""


class MatmulPerformanceBudgetTest(unittest.TestCase):
    def run_report(self, contents: str) -> int:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            path.write_text(contents, encoding="utf-8")
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                return check_report(path)

    def test_results_within_budget_pass(self) -> None:
        self.assertEqual(self.run_report(make_report(120.0, 200.0)), 0)

    def test_slow_f64_result_fails(self) -> None:
        self.assertEqual(self.run_report(make_report(20.0, 200.0)), 1)

    def test_missing_benchmark_result_fails_closed(self) -> None:
        self.assertEqual(self.run_report("gemm | 512x512 | 3.0 | 120.0\n"), 2)

    def test_non_finite_matmul_result_fails_closed(self) -> None:
        report = make_report(120.0, 200.0).replace(
            "gemm | 512x512 | 3.0 | 120.0", "gemm | 512x512 | 3.0 | 1e309"
        )
        self.assertEqual(self.run_report(report), 2)

    def test_annotation_records_ratios_before_enforcement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            path.write_text(make_report(20.0, 200.0), encoding="utf-8")
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(check_report(path, annotate=True), 0)
            annotated = path.read_text(encoding="utf-8")
            self.assertIn("| f64 | 512×512 | 20.00 | 100.00 | 5.00× | FAIL |", annotated)

    def test_cli_writes_machine_readable_gemm_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.md"
            json_path = Path(directory) / "results.json"
            report_path.write_text(make_report(20.0, 200.0), encoding="utf-8")
            arguments = [
                "check_matmul_perf.py",
                "--annotate",
                "--json-out",
                str(json_path),
                str(report_path),
            ]

            with patch("sys.argv", arguments), redirect_stdout(StringIO()):
                self.assertEqual(main(), 0)

            results = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(results["schema_version"], 2)
            benchmarks = results["benchmarks"]
            self.assertEqual(benchmarks[0]["benchmark"], "cpu-gemm")
            self.assertEqual(benchmarks[0]["max_slowdown"], 3.0)
            self.assertEqual(
                benchmarks[0]["results"],
                [
                    {
                        "dtype": "f64",
                        "shape": [512, 512],
                        "vtl_gflops": 20.0,
                        "numpy_gflops": 100.0,
                        "slowdown": 5.0,
                        "status": "fail",
                    },
                    {
                        "dtype": "f32",
                        "shape": [512, 512],
                        "vtl_gflops": 200.0,
                        "numpy_gflops": 210.0,
                        "slowdown": 1.05,
                        "status": "pass",
                    },
                ],
            )
            self.assertEqual(len(benchmarks), 3)
            self.assertEqual(benchmarks[1]["benchmark"], "reduced-svd")
            self.assertEqual(benchmarks[1]["baseline"], "numpy")
            self.assertEqual(benchmarks[1]["results"][0]["backend"], "pure-v")
            self.assertEqual(benchmarks[1]["results"][0]["shape"], [16, 16])
            self.assertEqual(benchmarks[1]["results"][0]["numpy_over_vtl"], 0.25)
            self.assertEqual(benchmarks[2]["results"][0]["backend"], "lapacke")
            self.assertEqual(benchmarks[2]["results"][0]["numpy_over_vtl"], 0.5)

    def test_json_requires_complete_svd_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.md"
            json_path = Path(directory) / "results.json"
            report_path.write_text(make_report(120.0, 200.0).replace("128x128  | 4.0 | 40.0\n", ""), encoding="utf-8")
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(check_report(report_path, json_path=json_path), 2)
            self.assertFalse(json_path.exists())

    def test_json_rejects_non_finite_svd_timings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.md"
            json_path = Path(directory) / "results.json"
            report = make_report(120.0, 200.0).replace(
                "16x16    | 1.0 | 10.0\n", "16x16    | 1e309 | 10.0\n"
            )
            report_path.write_text(report, encoding="utf-8")
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(check_report(report_path, json_path=json_path), 2)
            self.assertFalse(json_path.exists())


if __name__ == "__main__":
    unittest.main()
