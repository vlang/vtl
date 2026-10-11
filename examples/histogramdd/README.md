# N-dimensional histograms

This example bins rows of a sample tensor across three dimensions. It shows
explicit edge vectors, unweighted density, and weighted density. Counts are
stored in row-major bin order, and the last bin on each axis includes its
right edge.

Compile and run from `~/.vmodules`:

```sh
systemd-run --user --scope -p MemoryMax=768M -p MemorySwapMax=0 \
  env VJOBS=2 v -prod -o /tmp/vtl-histogramdd ./vtl/examples/histogramdd/main.v
systemd-run --user --scope -p MemoryMax=768M -p MemorySwapMax=0 \
  env VJOBS=2 /tmp/vtl-histogramdd
systemd-run --user --scope -p MemoryMax=2G -p MemorySwapMax=0 \
  env VJOBS=2 uv run --no-project --with numpy python \
  ./vtl/examples/histogramdd/numpy_baseline.py
```

The V and NumPy examples produce the same counts and density values for this
input. VTL stores unweighted counts as `int`; NumPy returns `float64` counts.

For the API details and NumPy compatibility notes, see
[`vtl.stats`](../../stats/README.md) and the
[NumPy parity tracker](../../docs/NUMPY_PARITY.md).
