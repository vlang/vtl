import vtl
import vtl.stats

fn main() {
	observations := vtl.from_1d([1.0, 2.0, 3.0, 4.0])!
	weights := vtl.from_1d([1.0, 1.0, 1.0, 7.0])!
	median := stats.quantile_weighted(observations, weights, 0.5)!
	eprintln('weighted median: ${median}')

	measurements := vtl.from_array([10.0, 7.0, 4.0, 3.0, 2.0, 1.0], [2, 3])!
	axis_weights := vtl.from_1d([1.0, 1.0, 7.0])!
	medians := stats.quantile_weighted_axis(measurements, axis_weights, 0.5, 1, false)!
	eprintln('weighted row medians: ${medians}')
}
