# Array interoperability

`vtl.interop` is an in-process CPU buffer interop slice for
[#748](https://github.com/vlang/vtl/issues/748). It describes borrowed buffers
explicitly instead of silently copying or silently aliasing them.

```v
backing := [f64(1), 2, 3, 4]
view := interop.contiguous(backing, [2, 2])!
owned := view.to_tensor()!                 // always copies
borrowed := view.as_borrowed_tensor()!     // zero-copy, guarded
```

## Contract

- `borrow(data, shape, strides, offset, mutable)` validates rank, non-negative
  dimensions and offset, positive strides, buffer bounds, and non-overlapping
  layouts. Overlapping strides are rejected rather than aliased.
- `contiguous(data, shape)` builds a row-major view over the whole buffer.
- `values()` gathers logical elements in row-major order; `to_tensor()` uses it
  as a safe copy fallback for any layout.
- `as_borrowed_tensor()` is allowed only for views that cover the entire
  buffer, start at offset zero, and are row-major contiguous. Other layouts
  must use `to_tensor()`.
- `metadata()` exposes shape, strides, offset, element size, and mutability as
  a plain struct suitable for an ABI boundary.
- A borrowed tensor shares the backing array. The caller must keep that array
  alive and must not mutate it while the tensor is in use. Use `to_tensor()`
  when an owned copy is required.

This module is about in-process buffer exchange. File and model interchange
(`.npy`, `.npz`, checkpoints) is separate and tracked in
[#743](https://github.com/vlang/vtl/issues/743) and
[#726](https://github.com/vlang/vtl/issues/726).

Run the focused tests with:

```sh
v test tests/interop/buffer_test.v
```

Backend/device transfer, C ABI examples, and optional Python/DLPack adapters
remain tracked in [#748](https://github.com/vlang/vtl/issues/748).
