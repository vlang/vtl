module main

import vtl

fn main() {
	vtl.random_seed(42)
	first := vtl.random[f64](0.0, 1.0, [4], vtl.TensorData{})
	println('First sequence: ${first.to_array()}')

	vtl.random_seed(42)
	repeated := vtl.random[f64](0.0, 1.0, [4], vtl.TensorData{})
	println('Repeated sequence: ${repeated.to_array()}')
	assert first.array_equal(repeated)
}
