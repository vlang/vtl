module main

import vtl

fn main() {
	values := vtl.from_1d[f64]([0.0, 0.5, 1.0, -0.5]) or { panic(err) }
	result := values.sinc() or { panic(err) }

	println('values: ${values.to_array()}')
	println('sinc:   ${result.to_array()}')
}
