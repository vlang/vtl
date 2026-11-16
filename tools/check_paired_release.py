#!/usr/bin/env python3
"""Validate that a VTL candidate and its exact VSL dependency form a release pair."""

import argparse
import re
import sys
from pathlib import Path


VERSION_PATTERN = re.compile(
    r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$"
)


def parse_manifest(contents, label):
    versions = re.findall(
        r"""(?m)^\s*version\s*:\s*(['"])(.*?)\1\s*$""", contents
    )
    if len(versions) != 1:
        raise ValueError(f"{label}: expected exactly one quoted version field")
    version = versions[0][1]
    if not VERSION_PATTERN.fullmatch(version):
        raise ValueError(f"{label}: invalid module version {version!r}")

    dependencies = re.findall(
        r"(?ms)^\s*dependencies\s*:\s*\[(.*?)\]", contents
    )
    if len(dependencies) > 1:
        raise ValueError(f"{label}: expected at most one dependencies field")
    entries = []
    if dependencies:
        entries = re.findall(r"""['"]([^'"]+)['"]""", dependencies[0])
    return version, entries


def validate_pair(vtl_contents, vtl_tag, vsl_contents=None):
    vtl_version, vtl_dependencies = parse_manifest(vtl_contents, "VTL manifest")
    expected_tag = f"v{vtl_version}"
    if vtl_tag != expected_tag:
        raise ValueError(
            f"VTL tag {vtl_tag!r} must match its manifest version ({expected_tag})"
        )

    vsl_dependencies = [
        dependency for dependency in vtl_dependencies
        if dependency.startswith("vsl@")
    ]
    if len(vsl_dependencies) != 1:
        raise ValueError("VTL manifest must declare exactly one versioned VSL dependency")
    expected_dependency = f"vsl@{vtl_tag}"
    legacy_dependency = f"vsl@{vtl_version}"
    if vsl_dependencies[0] not in (expected_dependency, legacy_dependency):
        raise ValueError(
            f"VTL dependency {vsl_dependencies[0]!r} must be {expected_dependency!r} "
            f"(legacy manifests may use {legacy_dependency!r})"
        )

    if vsl_contents is not None:
        vsl_version, _ = parse_manifest(vsl_contents, "VSL manifest")
        if vsl_version != vtl_version:
            raise ValueError(
                f"VSL manifest version {vsl_version!r} must match VTL version "
                f"{vtl_version!r}"
            )
    return expected_tag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vtl-tag", required=True, help="VTL candidate tag, e.g. v1.2.3")
    parser.add_argument("--vtl-manifest", type=Path, default=Path("v.mod"))
    parser.add_argument("--vsl-manifest", type=Path, help="Validate the checked-out VSL manifest")
    parser.add_argument("--github-output", type=Path, help="Write the resolved tag as a workflow output")
    args = parser.parse_args()

    try:
        vtl_contents = args.vtl_manifest.read_text(encoding="utf-8")
        vsl_contents = (
            args.vsl_manifest.read_text(encoding="utf-8")
            if args.vsl_manifest
            else None
        )
        vsl_tag = validate_pair(vtl_contents, args.vtl_tag, vsl_contents)
    except (OSError, ValueError) as error:
        print(f"paired release validation failed: {error}", file=sys.stderr)
        return 1

    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as output:
            output.write(f"vsl_tag={vsl_tag}\n")
    print(f"VTL {args.vtl_tag} pairs with VSL {vsl_tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
