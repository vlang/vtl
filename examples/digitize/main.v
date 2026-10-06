module main

import vtl

fn main() {
	measurements := vtl.from_1d([0.2, 1.5, 2.0, 3.8, 5.1])!
	bins := vtl.from_1d([0.0, 2.0, 4.0, 6.0])!
	println('Bin indices: ${vtl.digitize(measurements, bins, false)!.to_array()}')
	println('Insertion positions: ${vtl.searchsorted(bins, measurements, .right)!.to_array()}')
}
