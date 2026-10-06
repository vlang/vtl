# N-Dimensional Coordinate Grids

Build dense coordinate tensors from any number of one-dimensional inputs.
Choose `.xy` to swap the first two dimensions, matching NumPy's default, or
`.ij` to preserve the input axis order.

Run from `~/.vmodules`:

```sh
systemd-run --user --scope -p MemoryMax=768M -p MemorySwapMax=0 -- \
  env VJOBS=2 v -prod run ./vtl/examples/meshgrid_n/main.v
```
