module main

import vtl

fn main() {
	values := vtl.from_1d[f64]([0.5, -2.0, 4.0]) or { panic(err) }
	inverse := values.reciprocal() or { panic(err) }

	println('values:     ${values.to_array()}')
	println('reciprocal: ${inverse.to_array()}')
}
