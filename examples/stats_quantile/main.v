module main

import vtl
import vtl.stats
import math

fn main() {
	values := vtl.from_1d([32.0, 10.0, 20.0, 40.0, 30.0])!
	for q in [0.0, 0.25, 0.5, 0.75, 1.0] {
		println('q=${q}: ${stats.quantile_linear(values, q)!}')
	}

	measurements := vtl.from_array([1.0, math.nan(), 3.0, 5.0, math.nan(), 7.0], [3, 2])!
	println('NaN-aware mean: ${stats.nanmean(measurements)}')
	println('NaN-aware mean by column: ${stats.nanmean_axis(measurements, 0)!.to_array()}')
	println('NaN-aware sample standard deviation: ${stats.nanstd(measurements, 1)!}')
}
