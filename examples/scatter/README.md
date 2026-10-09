# Scatter updates along an axis

Apply repeated-index updates with `scatter_add`, then compare them with `put_along_axis` on a separate tensor.

## Run

From `~/.vmodules`:

```sh
v run ./vtl/examples/scatter/main.v
```

## Notes

`scatter_add` accumulates all updates to the same destination. `put_along_axis` uses last-write-wins semantics.
