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

	vtl.random_seed(2026)
	arrival_counts := vtl.poisson(3.5, [4], vtl.TensorData{})!
	lifetimes := vtl.weibull(1.5, [4], vtl.TensorData{})!
	vtl.random_seed(2026)
	assert arrival_counts.array_equal(vtl.poisson(3.5, [4], vtl.TensorData{})!)
	assert lifetimes.array_equal(vtl.weibull(1.5, [4], vtl.TensorData{})!)
	println('Poisson event counts: ${arrival_counts.to_array()}')
	println('Weibull lifetimes: ${lifetimes.to_array()}')
}
