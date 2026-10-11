import vtl
import vtl.stats

fn main() {
	sample := vtl.from_array[f64]([0.1, 0.2, 0.3, 0.8, 0.7, 0.9, 1.0, 1.0, 1.0], [3, 3])!
	edges := [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]]

	histogram := stats.histogramdd_edges[f64](sample, edges)!
	println('Counts [x_bin, y_bin, z_bin]: ${histogram.counts}')

	density := stats.histogramdd_density_edges[f64](sample, edges)!
	println('Density [x_bin, y_bin, z_bin]: ${density.counts}')

	weights := vtl.from_1d([1.0, 2.0, 3.0])!
	weighted := stats.histogramdd_weighted_edges[f64, f64](sample, weights, edges, true)!
	println('Weighted density [x_bin, y_bin, z_bin]: ${weighted.counts}')
}
