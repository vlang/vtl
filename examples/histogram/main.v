module main

import vtl
import vtl.stats

fn main() {
	measurements := vtl.from_1d([150, 151, 153, 160, 162, 170, 171, 180])!
	result := stats.histogram[int](measurements, 3)!
	println('Counts: ${result.counts.to_array()}')
	println('Bin edges: ${result.bin_edges.to_array()}')
}
