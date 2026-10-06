# Trapezoidal Integration

Integrate sampled data with uniform spacing or explicit sample coordinates.
VTL reduces the selected axis and returns `f64` results. Explicit coordinates
are used in their original order.

Run from `~/.vmodules`:

```sh
systemd-run --user --scope -p MemoryMax=768M -p MemorySwapMax=0 -- \
  env VJOBS=2 v -prod run ./vtl/examples/trapezoid/main.v
```
