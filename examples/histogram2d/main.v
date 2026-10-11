import vtl
import vtl.stats

fn main() {
	x := vtl.from_1d([0.2, 0.7, 1.2, 1.8])!
	y := vtl.from_1d([0.3, 1.3, 0.8, 1.8])!
	x_edges := [0.0, 1.0, 2.0]
	y_edges := [0.0, 1.0, 2.0]

	counts := stats.histogram2d_edges[f64, f64](x, y, x_edges, y_edges)!
	eprintln('Counts [x_bin, y_bin]: ${counts.counts}')

	weights := vtl.from_1d([1.0, 2.0, 1.0, 2.0])!
	density := stats.histogram2d_weighted_edges[f64, f64, f64](x, y, weights, x_edges,
		y_edges, true)!
	eprintln('Weighted density [x_bin, y_bin]: ${density.counts}')
}
