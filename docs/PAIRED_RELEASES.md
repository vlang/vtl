# Paired VTL/VSL candidate releases

VTL and VSL candidate releases use the same `v<version>` tag. The
`Paired VTL/VSL release` workflow runs on VTL version tags, or can be dispatched
with an existing VTL candidate tag. It checks that the VTL tag, VTL manifest
version, versioned VSL dependency, and checked-out VSL manifest version all
agree, then runs VTL's scoped test suite against that VSL tag.
The validator and its tests are checked out from the workflow revision, separately
from the candidate source, so dispatch can validate tags created before these
tools existed. Legacy candidate manifests with `vsl@<version>` are accepted and
resolved to the matching `v<version>` VSL tag.

For a candidate pair, prepare and tag the VSL candidate first. Set VTL's
`version` and `vsl@...` dependency in `v.mod` to that exact version, then create
the matching VTL tag. The workflow never falls back to VSL main: a missing tag,
manifest mismatch, or test failure fails the gate. Use workflow dispatch to
re-run validation for an existing VTL tag.

Pull-request and main-branch development CI remains unchanged and continues to
test against VSL main; the paired gate is only for candidate tags or explicit
dispatch.
