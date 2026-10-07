# Complex tensors

This example creates a `complex128` tensor and applies elementwise arithmetic.
It uses `math.complex.Complex`, V's standard f64 complex type.

From `~/.vmodules`, run it under a memory limit:

```sh
systemd-run --user --scope --quiet --property=MemoryMax=768M \
  --property=MemorySwapMax=0 --setenv=VJOBS=2 -- \
  v run ./vtl/examples/complex_tensors/main.v
```
