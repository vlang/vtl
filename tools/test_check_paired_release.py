import unittest

from check_paired_release import validate_pair


VTL_MANIFEST = """Module {
    name: 'vtl'
    version: '0.2.0-beta.1'
    repo_url: 'https://github.com/vlang/vtl'
    dependencies: ['vsl@v0.2.0-beta.1']
}
"""
VSL_MANIFEST = """Module {
    name: 'vsl'
    version: '0.2.0-beta.1'
}
"""


class PairedReleaseValidationTests(unittest.TestCase):
    def test_accepts_matching_candidate_pair(self):
        self.assertEqual(
            validate_pair(VTL_MANIFEST, "v0.2.0-beta.1", VSL_MANIFEST),
            "v0.2.0-beta.1",
        )

    def test_accepts_legacy_unprefixed_vsl_dependency_version(self):
        manifest = VTL_MANIFEST.replace(
            "vsl@v0.2.0-beta.1", "vsl@0.2.0-beta.1"
        )
        self.assertEqual(
            validate_pair(manifest, "v0.2.0-beta.1", VSL_MANIFEST),
            "v0.2.0-beta.1",
        )

    def test_rejects_vsl_dependency_only_in_line_comment(self):
        manifest = VTL_MANIFEST.replace(
            "dependencies: ['vsl@v0.2.0-beta.1']",
            "dependencies: [\n        // 'vsl@v0.2.0-beta.1'\n    ]",
        )
        with self.assertRaisesRegex(ValueError, "exactly one versioned VSL"):
            validate_pair(manifest, "v0.2.0-beta.1")

    def test_rejects_vsl_dependency_only_in_block_comment(self):
        for comments in (
            "/* 'vsl@v0.2.0-beta.1' */",
            "/* outer /* nested */ 'vsl@v0.2.0-beta.1' */",
        ):
            manifest = VTL_MANIFEST.replace(
                "dependencies: ['vsl@v0.2.0-beta.1']",
                f"dependencies: [{comments}]",
            )
            with self.subTest(comments=comments):
                with self.assertRaisesRegex(ValueError, "exactly one versioned VSL"):
                    validate_pair(manifest, "v0.2.0-beta.1")

    def test_rejects_version_field_only_in_comment(self):
        manifest = VTL_MANIFEST.replace(
            "version: '0.2.0-beta.1'",
            "// version: '0.2.0-beta.1'",
        )
        with self.assertRaisesRegex(ValueError, "exactly one quoted version field"):
            validate_pair(manifest, "v0.2.0-beta.1")

    def test_rejects_vtl_tag_that_differs_from_manifest(self):
        with self.assertRaisesRegex(ValueError, "must match its manifest version"):
            validate_pair(VTL_MANIFEST, "v0.2.0", VSL_MANIFEST)

    def test_rejects_branch_or_unqualified_version_as_candidate_tag(self):
        for candidate in ("feature/release", "0.2.0-beta.1", "refs/heads/release"):
            with self.subTest(candidate=candidate):
                with self.assertRaisesRegex(ValueError, "must match its manifest version"):
                    validate_pair(VTL_MANIFEST, candidate, VSL_MANIFEST)

    def test_rejects_dependency_tag_that_differs_from_candidate(self):
        manifest = VTL_MANIFEST.replace("v0.2.0-beta.1']", "v0.2.0']")
        with self.assertRaisesRegex(ValueError, "must be 'vsl@v0.2.0-beta.1'"):
            validate_pair(manifest, "v0.2.0-beta.1", VSL_MANIFEST)

    def test_rejects_vsl_manifest_version_mismatch(self):
        manifest = VSL_MANIFEST.replace("0.2.0-beta.1", "0.2.0")
        with self.assertRaisesRegex(ValueError, "must match VTL version"):
            validate_pair(VTL_MANIFEST, "v0.2.0-beta.1", manifest)

    def test_rejects_missing_or_ambiguous_vsl_dependency(self):
        for manifest in (
            VTL_MANIFEST.replace("dependencies: ['vsl@v0.2.0-beta.1']", "dependencies: []"),
            VTL_MANIFEST.replace(
                "['vsl@v0.2.0-beta.1']",
                "['vsl@v0.2.0-beta.1', 'vsl@v0.2.0-beta.1']",
            ),
        ):
            with self.subTest(manifest=manifest):
                with self.assertRaisesRegex(ValueError, "exactly one versioned VSL"):
                    validate_pair(manifest, "v0.2.0-beta.1", VSL_MANIFEST)


if __name__ == "__main__":
    unittest.main()
