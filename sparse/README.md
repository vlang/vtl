# Sparse tensors

`vtl.sparse` is an incremental sparse-tensor implementation for
[#746](https://github.com/vlang/vtl/issues/746). The first slice covers
two-dimensional coordinate (COO) tensors on CPU.

## Current API

- `sparse.from_coo[T](shape, rows, cols, values)` validates a 2-D shape and
  every coordinate, then copies the arrays into a `CooTensor[T]`.
- `sparse.from_dense[T](dense)` extracts non-zero coordinates from a 2-D dense
  tensor.
- `CooTensor.nnz()` reports the stored coordinate count.
- `CooTensor.to_dense()` expands to a dense row-major tensor; values at the
  same coordinate are summed in insertion order.
- `CooTensor.coalesce()` sorts coordinates by row/column and sums duplicates.

Explicit zeros are allowed and are retained by `coalesce`, so `nnz()` counts
stored entries rather than mathematical non-zeros. Empty and zero-sized
dimensions are supported. Duplicate coordinates are legal and are defined to
sum in insertion order.

Run the focused tests with:

```sh
v test tests/sparse/coo_test.v
```

## Not yet supported

CSR/CSC formats, sparse-dense and sparse-sparse arithmetic, reductions,
autograd participation, GPU kernels, and SciPy/PyTorch reference fixtures are
tracked in [#746](https://github.com/vlang/vtl/issues/746). Unsupported
operations must not be assumed to work through an implicit dense conversion.
