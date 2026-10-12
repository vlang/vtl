module vtl

import rand
import math

fn test_random_generator_supports_geometric_sampling() ! {
	mut generator := new_random_generator(987)
	values := generator.geometric(0.25, [64])!
	assert values.shape == [64]
	for i in 0 .. values.size {
		assert values.get_nth(i) > 0
	}
	one_trial := generator.geometric(1.0, [4])!
	for i in 0 .. one_trial.size {
		assert one_trial.get_nth(i) == 1
	}
	mut replay := new_random_generator(987)
	replayed := replay.geometric(0.25, [64])!
	assert values.array_equal(replayed)
	generator.free()
	replay.free()
}

fn test_random_generator_supports_seeded_binomial_and_exponential() ! {
	mut generator := new_random_generator(654)
	binomial_values := generator.binomial(12, 0.25, [64])!
	exponential_values := generator.exponential(2.0, [64])!
	for value in binomial_values.to_array() {
		assert value >= 0 && value <= 12
	}
	for value in exponential_values.to_array() {
		assert value >= 0 && !math.is_inf(value, 0) && !math.is_nan(value)
	}
	mut replay := new_random_generator(654)
	assert binomial_values.array_equal(replay.binomial(12, 0.25, [64])!)
	assert exponential_values.array_equal(replay.exponential(2.0, [64])!)
	assert generator.binomial(0, 0.5, [4])!.to_array() == [0, 0, 0, 0]
	assert generator.binomial(5, 1.0, [4])!.to_array() == [5, 5, 5, 5]
	if _ := generator.binomial(-1, 0.5, [1]) {
		assert false, 'binomial must reject negative trial counts'
	}
	if _ := generator.binomial(2, math.nan(), [1]) {
		assert false, 'binomial must reject a NaN probability'
	}
	if _ := generator.exponential(0.0, [1]) {
		assert false, 'exponential must reject a non-positive rate'
	}
	generator.free()
	replay.free()
}

fn test_random_generator_negative_binomial_is_seeded_and_matches_moments() ! {
	mut generator := new_random_generator(8721)
	values := generator.negative_binomial(4.0, 0.5, [4096])!
	assert values.shape == [4096]
	mut total := 0
	for value in values.to_array() {
		assert value >= 0
		total += value
	}
	mean := f64(total) / 4096
	mut squared_deviation := 0.0
	for value in values.to_array() {
		delta := f64(value) - mean
		squared_deviation += delta * delta
	}
	variance := squared_deviation / 4096
	assert math.abs(mean - 4.0) < 0.2
	assert math.abs(variance - 8.0) < 0.8
	mut replay := new_random_generator(8721)
	assert values.array_equal(replay.negative_binomial(4.0, 0.5, [4096])!)
	fractional_shape_samples := generator.negative_binomial(2.5, 0.4, [64])!
	for value in fractional_shape_samples.to_array() {
		assert value >= 0
	}
	assert generator.negative_binomial(2.5, 1.0, [3])!.to_array() == [0, 0, 0]
	if _ := generator.negative_binomial(0, 0.5, [1]) {
		assert false, 'negative_binomial must reject non-positive n'
	}
	if _ := generator.negative_binomial(2, 0, [1]) {
		assert false, 'negative_binomial must reject zero probability'
	}
	if _ := generator.negative_binomial(2, math.nan(), [1]) {
		assert false, 'negative_binomial must reject non-finite probability'
	}
	generator.free()
	replay.free()
}

fn test_random_generator_supports_hypergeometric_sampling() ! {
	mut generator := new_random_generator(751)
	values := generator.hypergeometric(5, 5, 4, [4096])!
	mut replay := new_random_generator(751)
	assert values.array_equal(replay.hypergeometric(5, 5, 4, [4096])!)
	mut total := 0
	for value in values.to_array() {
		assert value >= 0 && value <= 4
		total += value
	}
	assert math.abs(f64(total) / f64(values.size) - 2.0) < 0.06
	complement_draw := generator.hypergeometric(5, 5, 8, [512])!
	mut complement_total := 0
	for value in complement_draw.to_array() {
		assert value >= 3 && value <= 5
		complement_total += value
	}
	assert math.abs(f64(complement_total) / f64(complement_draw.size) - 4.0) < 0.1
	hrua_draw := generator.hypergeometric(50, 50, 80, [1024])!
	mut hrua_total := 0
	for value in hrua_draw.to_array() {
		assert value >= 30 && value <= 50
		hrua_total += value
	}
	assert math.abs(f64(hrua_total) / f64(hrua_draw.size) - 40.0) < 0.25
	reflected_draw := generator.hypergeometric(80, 20, 60, [512])!
	mut reflected_total := 0
	for value in reflected_draw.to_array() {
		assert value >= 40 && value <= 60
		reflected_total += value
	}
	assert math.abs(f64(reflected_total) / f64(reflected_draw.size) - 48.0) < 0.4
	assert generator.hypergeometric(7, 0, 3, [4])!.to_array() == [3, 3, 3, 3]
	assert generator.hypergeometric(7, 5, 0, [2])!.to_array() == [0, 0]
	assert generator.hypergeometric(0, 0, 0, [1])!.to_array() == [0]
	if _ := generator.hypergeometric(-1, 2, 1, [1]) {
		assert false, 'hypergeometric must reject negative population counts'
	}
	if _ := generator.hypergeometric(1, 2, 4, [1]) {
		assert false, 'hypergeometric must reject samples larger than the population'
	}
	if _ := generator.hypergeometric(max_int, 1, 0, [1]) {
		assert false, 'hypergeometric must reject population-size overflow'
	}
	generator.free()
	replay.free()
}
