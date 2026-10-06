# Find Non-Zero Coordinates

This example shows how `vtl.argwhere` groups the coordinates of non-zero
elements into rows. For an input of rank `r`, the result has shape
`[number of matches, r]`.

Run it from `~/.vmodules`:

```sh
systemd-run --user --scope -p MemoryMax=768M -p MemorySwapMax=0 -- \
  env VJOBS=2 v -prod run ./vtl/examples/argwhere/main.v
```

The behavior follows NumPy's [`argwhere`](https://numpy.org/doc/stable/reference/generated/numpy.argwhere.html)
coordinate grouping. For indexing by axis, VTL also provides `take_along_axis`.
