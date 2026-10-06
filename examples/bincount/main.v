module main

import vtl
import vtl.stats

fn main() {
	labels := vtl.from_1d([0, 1, 1, 2, 2, 2])!
	weights := vtl.from_1d([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])!
	println('Counts: ${stats.bincount[int](labels, 4)!.to_array()}')
	println('Weighted counts: ${stats.bincount_weighted[int, f64](labels, weights, 4)!.to_array()}')
}
