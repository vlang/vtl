module main

import vtl

fn main() {
	mean := vtl.from_array[f64]([0.0, 2.0], [2]) or { panic(err) }
	covariance := vtl.from_array[f64]([1.0, 0.8, 0.8, 1.5], [2, 2]) or { panic(err) }
	mut rng := vtl.new_random_generator(2026)
	samples := rng.multivariate_normal(mean, covariance, [5, 3]) or { panic(err) }
	println('Sample tensor shape: ${samples.shape}')
	println('Samples: ${samples.to_array()}')
	rng.free()
}
