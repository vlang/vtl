# ADR-0001: Keep heterogeneous structured data outside the tensor core

**Status:** Proposed

## Context

VTL tensors store one element type per tensor. The NumPy reader requires the
requested V type to match the file dtype and rejects structured, object, and
string dtypes. Adding heterogeneous records directly to `Tensor[T]` would
complicate its storage, views, and numerical APIs.

## Options Considered

### Option A: Add record tensors to VTL

**Pros:**
- Structured NumPy arrays could be represented directly.
- Field access could live beside tensor operations.

**Cons:**
- Heterogeneous fields do not fit the existing homogeneous `Tensor[T]` model.
- Record layout, alignment, views, and serialization would expand the core API.

### Option B: Keep VTL homogeneous and use a companion table package

**Pros:**
- Preserves VTL's typed tensor and numerical-kernel contracts.
- Gives heterogeneous records a natural home without implicit numeric coercion.

**Cons:**
- Requires a separate package/API for table operations.
- Structured-file ingestion is not available in VTL today.

### Option C: Add an external dataframe dependency to VTL

**Pros:**
- Could reuse an established table representation and file ecosystem.

**Cons:**
- Adds a dependency and couples the numerical core to a broader data model.
- Conversion and ownership behavior would still need explicit contracts.

## Decision

Propose Option B: VTL remains a homogeneous tensor library; structured and
record arrays belong in a companion table package, not in `Tensor[T]` or the
core `.npy` reader.

The conversion boundary is explicit and column-oriented: a table package may
select a named field and convert it to a typed VTL tensor, preserving row
order. Each tensor contains one supported dtype; unsupported fields, layouts,
or conversions must return errors rather than silently coercing values. No
companion package or structured-array conversion API is selected by this
proposal.

## Consequences

- VTL can continue rejecting structured, object, and string `.npy` dtypes
  without weakening tensor type guarantees.
- Users needing heterogeneous records still need a separate table package.
- Any future table integration must define field selection, dtype conversion,
  row ordering, missing-value handling, and ownership before implementation.
- Maintainer review is required before this proposal is treated as accepted.

## References

- [Issue #728](https://github.com/vlang/vtl/issues/728)
- [NumPy parity tracker](../NUMPY_PARITY.md#structured-and-record-arrays)
- [NumPy I/O tutorial](../TUTORIAL_NUMPY_IO.md)
- [`a_tensor.v`](../../a_tensor.v): homogeneous `Tensor[T]` storage
- [`npy/npy.v`](../../npy/npy.v): dtype validation and supported `.npy` element types
