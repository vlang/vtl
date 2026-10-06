module main

import vtl
import vtl.stats

fn main() {
	measurements := vtl.from_2d([[4.0, 5.0, 4.5], [3.0, 4.0, 5.0]])!
	reliability := vtl.from_1d([1.0, 2.0, 1.0])!
	global_weights := vtl.from_2d([[1.0, 2.0, 1.0], [1.0, 2.0, 1.0]])!
	println('Weighted mean: ${stats.average(measurements, global_weights)!}')
	println('Weighted means by row: ${stats.average_axis(measurements, reliability, 1)!.to_array()}')
}
