import vtl

fn main() {
	mut rng := vtl.new_random_generator(42)
	concentration := vtl.from_1d([0.5, 1.5, 3.0])!
	samples := rng.dirichlet[f64](concentration, [4])!
	for sample in 0 .. samples.shape[0] {
		mut total := 0.0
		for category in 0 .. samples.shape[1] {
			total += samples.get([sample, category])
		}
		assert total > 0.999999999999 && total < 1.000000000001
	}
	println('Dirichlet sample shape: ${samples.shape}')
	println('Samples: ${samples.to_array()}')
	rng.free()
}
