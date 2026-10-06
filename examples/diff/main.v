module main

import vtl

fn main() {
	series := vtl.from_1d([1.0, 2.0, 4.0, 7.0])!
	println('First difference: ${vtl.diff[f64](series, 1, -1)!.to_array()}')
	println('Second difference: ${vtl.diff[f64](series, 2, -1)!.to_array()}')
}
