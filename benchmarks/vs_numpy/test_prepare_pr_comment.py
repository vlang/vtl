"""Tests for preparing a size-limited GitHub benchmark comment."""

import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from prepare_pr_comment import MAX_COMMENT_CHARS, bounded_comment, main


class PreparePrCommentTest(unittest.TestCase):
    def test_short_report_is_unchanged(self) -> None:
        report = "# Benchmark results\n\nAll done.\n"
        self.assertEqual(bounded_comment(report, "https://example.test/run/1"), report)

    def test_long_report_is_truncated_at_a_line_boundary_with_run_link(self) -> None:
        report = "benchmark output\n" * (MAX_COMMENT_CHARS // 10)
        workflow_url = "https://github.com/vlang/vtl/actions/runs/123"

        comment = bounded_comment(report, workflow_url)

        self.assertLessEqual(len(comment), MAX_COMMENT_CHARS)
        self.assertIn("truncated", comment)
        self.assertIn(workflow_url, comment)
        self.assertTrue(comment.endswith("\n"))

    def test_cli_writes_report_with_github_run_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.md"
            comment_path = Path(directory) / "comment.md"
            report = "benchmark output\n" * (MAX_COMMENT_CHARS // 10)
            report_path.write_text(report, encoding="utf-8")
            environment = {
                "GITHUB_SERVER_URL": "https://github.com",
                "GITHUB_REPOSITORY": "vlang/vtl",
                "GITHUB_RUN_ID": "123",
            }

            with patch.dict(os.environ, environment), patch.object(
                sys, "argv", ["prepare_pr_comment.py", str(report_path), str(comment_path)]
            ):
                self.assertEqual(main(), 0)

            comment = comment_path.read_text(encoding="utf-8")
            self.assertLessEqual(len(comment), MAX_COMMENT_CHARS)
            self.assertIn("https://github.com/vlang/vtl/actions/runs/123", comment)
            self.assertEqual(report_path.read_text(encoding="utf-8"), report)


if __name__ == "__main__":
    unittest.main()
