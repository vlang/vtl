module main

import vtl

fn main() {
	bases := vtl.from_2d[i16]([[2], [3]]) or { panic(err) }
	exponents := vtl.from_1d[f32]([2.0, 3.0, 0.0]) or { panic(err) }
	result := vtl.pow_promoted[f32, i16, f32](bases, exponents) or { panic(err) }

	println('shape: ${result.shape}')
	println('dtype: ${result.dtype()}')
	println('powers: ${result.to_array()}')
}
