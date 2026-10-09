# Scatter updates along an axis

Apply repeated-index updates with `scatter_add`, compare replacement writes
with `put_along_axis`, and update row-major flat positions with `put`.

## Run

From `~/.vmodules`:

```sh
v run ./vtl/examples/scatter/main.v
```

## Notes

`scatter_add` accumulates all updates to the same destination. Both
`put_along_axis` and `put` use last-write-wins semantics. `put` repeats a short
values tensor and supports NumPy-style index modes through `put_with_mode`.
