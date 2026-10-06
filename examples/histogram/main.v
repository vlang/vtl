module main

import vtl
import vtl.stats

fn main() {
	measurements := vtl.from_1d([150, 151, 153, 160, 162, 170, 171, 180])!
	result := stats.histogram[int](measurements, 3)!
	println('Counts: ${result.counts.to_array()}')
	println('Bin edges: ${result.bin_edges.to_array()}')
	automatic := stats.histogram_auto[int](measurements, .automatic)!
	println('Automatic bin count: ${automatic.counts.size}')
	stone := stats.histogram_auto[int](measurements, .stone)!
	println('Stone bin count: ${stone.counts.size}')

	weighted := stats.histogram_weighted_edges[int, f64](measurements,
		vtl.from_1d([1.0, 1, 2, 1, 3, 2, 1, 4])!, [150.0, 160.0, 175.0, 180.0], true)!
	println('Weighted density: ${weighted.counts.to_array()}')
}
