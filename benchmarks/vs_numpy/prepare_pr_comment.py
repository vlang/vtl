#!/usr/bin/env python3
"""Prepare a bounded copy of the benchmark report for a GitHub PR comment."""

from __future__ import annotations

import os
import sys
from pathlib import Path


MAX_COMMENT_CHARS = 60_000


def bounded_comment(report: str, workflow_url: str) -> str:
    notice = (
        "\n\n> The benchmark report was truncated to fit GitHub's comment size limit. "
        f"[View the complete report and logs]({workflow_url}).\n"
    )
    if len(report) <= MAX_COMMENT_CHARS:
        return report
    if len(notice) >= MAX_COMMENT_CHARS:
        raise ValueError("workflow URL is too long to prepare a benchmark comment")

    prefix = report[: MAX_COMMENT_CHARS - len(notice)]
    last_newline = prefix.rfind("\n")
    if last_newline >= 0:
        prefix = prefix[: last_newline + 1]
    return prefix + notice


def main() -> int:
    if len(sys.argv) != 3:
        print(
            f"usage: {Path(sys.argv[0]).name} REPORT.md COMMENT.md",
            file=sys.stderr,
        )
        return 2
    try:
        workflow_url = (
            f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
            f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        )
        report = Path(sys.argv[1]).read_text(encoding="utf-8")
        comment = bounded_comment(report, workflow_url)
        Path(sys.argv[2]).write_text(comment, encoding="utf-8")
    except (KeyError, OSError, ValueError) as error:
        print(f"cannot prepare benchmark PR comment: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
