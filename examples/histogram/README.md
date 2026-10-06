# Histograms

This example computes evenly spaced and data-driven histograms, then a weighted
probability density using custom, non-uniform bin edges. The last bin includes
its right edge, following NumPy's histogram convention.

Run it from the V module workspace:

```sh
cd ~/.vmodules
systemd-run --user --scope -p MemoryMax=768M -p MemorySwapMax=0 \
  env VJOBS=2 v -prod run ./vtl/examples/histogram/main.v
```

The weighted density is normalized so the sum of each bin's density times its
width equals one. See [`NUMPY_PARITY.md`](../../docs/NUMPY_PARITY.md) for the
current scope and remaining gaps in VTL's NumPy compatibility.
