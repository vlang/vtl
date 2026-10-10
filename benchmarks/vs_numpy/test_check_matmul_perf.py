"""Tests for the PR CPU GEMM performance budget parser."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest

from check_matmul_perf import check_report


def make_report(vtl_f64: float, vtl_f32: float) -> str:
    return f"""gemm | 512x512 | 3.0 | {vtl_f64}
gemm_f32 | 512x512 | 2.0 | {vtl_f32}
numpy gemm 512x512 | 2.5 ms | 100.0 GFLOPS | 1.0
numpy gemm_f32 512x512 | 1.8 ms | 210.0 GFLOPS | 1.0
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

    def test_annotation_records_ratios_before_enforcement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            path.write_text(make_report(20.0, 200.0), encoding="utf-8")
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(check_report(path, annotate=True), 0)
            annotated = path.read_text(encoding="utf-8")
            self.assertIn("| f64 | 512×512 | 20.00 | 100.00 | 5.00× | FAIL |", annotated)


if __name__ == "__main__":
    unittest.main()
