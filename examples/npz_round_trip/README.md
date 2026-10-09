# Mixed-dtype NPZ round trip

Write named tensors with different element types to one `.npz` archive, read them back, and print
member names and values.

## Run

From `~/.vmodules`:

```sh
v run ./vtl/examples/npz_round_trip/main.v
```

## Notes

Pass an existing archive path as an optional argument to read its `weights`, `labels`, and `mask`
members instead. See [NumPy I/O tutorial](../../docs/TUTORIAL_NUMPY_IO.md).
