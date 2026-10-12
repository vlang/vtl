module vtl

import rand
import math

fn test_random_generators_have_independent_reproducible_state() ! {
	mut first := new_random_generator(42)
	mut second := new_random_generator(7)
	first_values := first.uniform(0.0, 1.0, [8])!
	_ := second.uniform(0.0, 1.0, [32])!
	mut replay := new_random_generator(42)
	replayed_values := replay.uniform(0.0, 1.0, [8])!
	assert first_values.array_equal(replayed_values)
	first.free()
	second.free()
	replay.free()
}

fn test_random_generator_supports_normal_and_bernoulli_distributions() ! {
	mut generator := new_random_generator(123)
	normal_values := generator.normal([4], NormalTensorData{ mu: 5.0, sigma: 1.0 })!
	assert normal_values.shape == [4]
	for i in 0 .. normal_values.size {
		assert normal_values.get_nth(i) == normal_values.get_nth(i)
	}
	true_values := generator.bernoulli(1.0, [3])!
	false_values := generator.bernoulli(0.0, [3])!
	for i in 0 .. true_values.size {
		assert true_values.get_nth(i)
	}
	for i in 0 .. false_values.size {
		assert !false_values.get_nth(i)
	}
	generator.free()
}

fn test_random_generator_multivariate_normal_shapes_statistics_and_singular_covariance() ! {
	mean := from_array[f64]([2.0, -1.0], [2])!
	covariance := from_array[f64]([4.0, 1.5, 1.5, 1.0], [2, 2])!
	mut generator := new_random_generator(904)
	samples := generator.multivariate_normal(mean, covariance, [4096])!
	assert samples.shape == [4096, 2]
	mut sum_x := 0.0
	mut sum_y := 0.0
	for i in 0 .. samples.shape[0] {
		sum_x += samples.get_nth(i * 2)
		sum_y += samples.get_nth(i * 2 + 1)
	}
	mean_x := sum_x / f64(samples.shape[0])
	mean_y := sum_y / f64(samples.shape[0])
	mut variance_x := 0.0
	mut variance_y := 0.0
	mut covariance_xy := 0.0
	for i in 0 .. samples.shape[0] {
		dx := samples.get_nth(i * 2) - mean_x
		dy := samples.get_nth(i * 2 + 1) - mean_y
		variance_x += dx * dx
		variance_y += dy * dy
		covariance_xy += dx * dy
	}
	assert math.abs(mean_x - 2.0) < 0.12
	assert math.abs(mean_y + 1.0) < 0.08
	assert math.abs(variance_x / 4096 - 4.0) < 0.2
	assert math.abs(variance_y / 4096 - 1.0) < 0.08
	assert math.abs(covariance_xy / 4096 - 1.5) < 0.12

	mut replay := new_random_generator(904)
	assert samples.array_equal(replay.multivariate_normal(mean, covariance, [4096])!)

	singular := from_array[f64]([1.0, 1.0, 1.0, 1.0], [2, 2])!
	mut singular_rng := new_random_generator(905)
	singular_samples := singular_rng.multivariate_normal(mean, singular, [64])!
	for i in 0 .. singular_samples.shape[0] {
		assert math.abs((singular_samples.get_nth(i * 2) - 2.0) - (singular_samples.get_nth(i * 2 + 1) + 1.0)) < 1e-10
	}
	assert singular_rng.multivariate_normal(mean, covariance, [])!.shape == [2]
	if _ := singular_rng.multivariate_normal(mean, covariance, [-1]) {
		assert false, 'multivariate_normal must reject negative sample dimensions'
	}
	mut asymmetric_rng := new_random_generator(906)
	asymmetric := from_array[f64]([1.0, 0.5, 0.0, 1.0], [2, 2])!
	if _ := asymmetric_rng.multivariate_normal(mean, asymmetric, [1]) {
		assert false, 'multivariate_normal must reject asymmetric covariance'
	}
	mut indefinite_rng := new_random_generator(907)
	indefinite := from_array[f64]([1.0, 2.0, 2.0, 1.0], [2, 2])!
	if _ := indefinite_rng.multivariate_normal(mean, indefinite, [1]) {
		assert false, 'multivariate_normal must reject non-positive-semidefinite covariance'
	}
	generator.free()
	replay.free()
	singular_rng.free()
	asymmetric_rng.free()
	indefinite_rng.free()
}

fn test_random_generator_integers_are_seeded_and_obey_endpoint() ! {
	mut generator := new_random_generator(421)
	values := generator.integers(-3, 7, [256])!
	mut replay := new_random_generator(421)
	assert values.array_equal(replay.integers(-3, 7, [256])!)
	for value in values.to_array() {
		assert value >= -3 && value < 7
	}
	inclusive := generator.integers_with_endpoint(4, 4, [32], true)!
	assert inclusive.to_array() == []int{len: 32, init: 4}
	mut empty_rng := new_random_generator(422)
	empty := empty_rng.integers(0, 5, [0])!
	assert empty.size == 0
	if _ := generator.integers(4, 4, [1]) {
		assert false, 'integers must reject an empty half-open range'
	}
	exclusive := generator.integers_with_endpoint(0, 2, [128], false)!
	for value in exclusive.to_array() {
		assert value >= 0 && value < 2
	}
	if _ := generator.integers_with_endpoint(0, max_int, [1], true) {
		assert false, 'integers must reject an inclusive max_int upper bound'
	}
	if _ := generator.integers(-max_int, max_int, [1]) {
		assert false, 'integers must reject ranges too wide for the RNG'
	}
	generator.free()
	replay.free()
	empty_rng.free()
}

fn test_random_generator_supports_seeded_lognormal() ! {
	mut generator := new_random_generator(258)
	values := generator.lognormal(0.25, 0.5, [64])!
	for value in values.to_array() {
		assert value > 0
	}
	mut replay := new_random_generator(258)
	assert values.array_equal(replay.lognormal(0.25, 0.5, [64])!)
	constant := generator.lognormal(2.0, 0.0, [3])!
	for value in constant.to_array() {
		assert value == math.exp(2.0)
	}
	if _ := generator.lognormal(0.0, -1.0, [1]) {
		assert false, 'lognormal must reject a negative sigma'
	}
	generator.free()
	replay.free()
}
