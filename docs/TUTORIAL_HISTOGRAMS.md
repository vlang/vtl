# Histograms and binning

VTL's `vtl.stats` module provides one-dimensional and paired two-dimensional
histograms. The 2D result uses `[x_bin, y_bin]` order, the same orientation as
NumPy's `histogram2d`.

## Paired samples and explicit edges

`histogram2d(x, y, bins_x, bins_y)` infers a uniform range from each vector.
Use `histogram2d_edges` when bin boundaries are part of the analysis:

```v
import vtl
import vtl.stats

x := vtl.from_1d([0.2, 0.7, 1.2, 1.8])!
y := vtl.from_1d([0.3, 1.3, 0.8, 1.8])!
grid := stats.histogram2d_edges[f64, f64](x, y, [0.0, 1.0, 2.0], [0.0, 1.0, 2.0])!
assert grid.counts.shape == [2, 2]
assert grid.counts.to_array() == [1, 1, 1, 1]
```

Values outside either edge range are ignored. The last bin includes its right
edge. Inputs must be one-dimensional and contain the same number of paired
samples.

## Probability density and weights

`histogram2d_density` and `histogram2d_density_edges` normalize the in-range
counts so that summing each density multiplied by its bin area gives one.
Weighted versions accept one numeric weight per sample; set `density` to
`true` to normalize weighted counts by their total and bin areas:

```v
import vtl
import vtl.stats

x := vtl.from_1d([0.5, 0.5, 1.5])!
y := vtl.from_1d([0.5, 1.5, 0.5])!
weights := vtl.from_1d([1.0, 2.0, 3.0])!
weighted := stats.histogram2d_weighted_edges[f64, f64, f64](x, y, weights,
	[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], true)!
assert weighted.counts.shape == [2, 2]
```

For runnable output, see the [2D histogram example](../examples/histogram2d).
The [NumPy parity tracker](./NUMPY_PARITY.md) records current coverage and gaps.
