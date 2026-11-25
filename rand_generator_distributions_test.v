module vtl

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

fn test_random_generator_supports_multivariate_hypergeometric_sampling() ! {
	mut generator := new_random_generator(752)
	colors := from_1d([7, 11, 13])!
	values := generator.multivariate_hypergeometric(colors, 12, [4096])!
	assert values.shape == [4096, 3]
	mut replay := new_random_generator(752)
	assert values.array_equal(replay.multivariate_hypergeometric(colors, 12, [4096])!)
	mut totals := [0, 0, 0]
	for sample_index in 0 .. values.shape[0] {
		mut sample_total := 0
		for category in 0 .. colors.size {
			value := values.get_nth(sample_index * colors.size + category)
			assert value >= 0 && value <= colors.get_nth(category)
			sample_total += value
			totals[category] += value
		}
		assert sample_total == 12
	}
	assert math.abs(f64(totals[0]) / 4096.0 - 12.0 * 7.0 / 31.0) < 0.1
	assert math.abs(f64(totals[1]) / 4096.0 - 12.0 * 11.0 / 31.0) < 0.1
	assert math.abs(f64(totals[2]) / 4096.0 - 12.0 * 13.0 / 31.0) < 0.1
	assert generator.multivariate_hypergeometric(colors, 0, [2])!.to_array() == [0, 0, 0, 0, 0,
		0]
	assert generator.multivariate_hypergeometric(colors, 31, [2])!.to_array() == [7, 11, 13, 7,
		11, 13]
	assert generator.multivariate_hypergeometric(colors, 4, [])!.shape == [3]
	assert generator.multivariate_hypergeometric(colors, 4, [0])!.shape == [0, 3]
	if _ := generator.multivariate_hypergeometric(from_1d([-1, 2])!, 1, [1]) {
		assert false, 'multivariate hypergeometric must reject negative color counts'
	}
	if _ := generator.multivariate_hypergeometric(colors, 32, [1]) {
		assert false, 'multivariate hypergeometric must reject draws larger than the population'
	}
	if _ := generator.multivariate_hypergeometric(from_array[int]([1, 2], [1, 2])!, 1, [1]) {
		assert false, 'multivariate hypergeometric must require one-dimensional colors'
	}
	if _ := generator.multivariate_hypergeometric(from_1d([max_int, 1])!, 0, [1]) {
		assert false, 'multivariate hypergeometric must reject total population overflow'
	}
	if _ := generator.multivariate_hypergeometric(colors, 1, [-1]) {
		assert false, 'multivariate hypergeometric must reject negative sample dimensions'
	}
	generator.free()
	replay.free()
}

fn test_random_generator_zipf_is_seeded_and_matches_moments() ! {
	mut generator := new_random_generator(753)
	values := generator.zipf(4.0, [8192])!
	mut total := 0
	mut ones := 0
	for value in values.to_array() {
		assert value >= 1
		total += value
		if value == 1 {
			ones++
		}
	}
	assert math.abs(f64(total) / f64(values.size) - 1.1106265353261488) < 0.03
	assert math.abs(f64(ones) / f64(values.size) - 0.9239384029215902) < 0.015
	mut replay := new_random_generator(753)
	assert values.array_equal(replay.zipf(4.0, [8192])!)
	assert generator.zipf(1025.0, [4])!.to_array() == [1, 1, 1, 1]
	assert generator.zipf(4.0, [0])!.shape == [0]
	if _ := generator.zipf(1.0, [1]) {
		assert false, 'zipf must reject an exponent at or below one'
	}
	if _ := generator.zipf(math.nan(), [1]) {
		assert false, 'zipf must reject a NaN exponent'
	}
	if _ := generator.zipf(math.inf(1), [1]) {
		assert false, 'zipf must reject an infinite exponent'
	}
	generator.free()
	replay.free()
}

fn test_random_generator_logseries_is_seeded_and_matches_moments() ! {
	mut generator := new_random_generator(754)
	values := generator.logseries(0.5, [8192])!
	mut total := 0
	mut ones := 0
	for value in values.to_array() {
		assert value >= 1
		total += value
		if value == 1 {
			ones++
		}
	}
	assert math.abs(f64(total) / f64(values.size) - 1.4426950408889634) < 0.04
	assert math.abs(f64(ones) / f64(values.size) - 0.7213475204444817) < 0.02
	mut replay := new_random_generator(754)
	assert values.array_equal(replay.logseries(0.5, [8192])!)
	small_probability := generator.logseries(0.01, [4096])!
	mut small_total := 0
	for value in small_probability.to_array() {
		assert value >= 1
		small_total += value
	}
	assert math.abs(f64(small_total) / f64(small_probability.size) -
		(-0.01 / (0.99 * math.log(0.99)))) < 0.02
	assert generator.logseries(0.5, [0])!.shape == [0]
	if _ := generator.logseries(0.0, [1]) {
		assert false, 'logseries must reject zero probability'
	}
	if _ := generator.logseries(1.0, [1]) {
		assert false, 'logseries must reject probability one'
	}
	if _ := generator.logseries(math.nan(), [1]) {
		assert false, 'logseries must reject a NaN probability'
	}
	if _ := generator.logseries(math.inf(1), [1]) {
		assert false, 'logseries must reject an infinite probability'
	}
	generator.free()
	replay.free()
}

fn test_random_generator_noncentral_distributions_are_seeded_and_match_moments() ! {
	mut generator := new_random_generator(755)
	chi_samples := generator.noncentral_chi_square(4.0, 3.0, [8192])!
	mut chi_total := 0.0
	for value in chi_samples.to_array() {
		assert value >= 0 && !math.is_nan(value) && !math.is_inf(value, 0)
		chi_total += value
	}
	assert math.abs(chi_total / 8192.0 - 7.0) < 0.2
	mut replay := new_random_generator(755)
	assert chi_samples.array_equal(replay.noncentral_chi_square(4.0, 3.0, [8192])!)

	mut small_df_rng := new_random_generator(756)
	small_df := small_df_rng.noncentral_chi_square(0.5, 2.0, [8192])!
	mut small_df_total := 0.0
	for value in small_df.to_array() {
		assert value >= 0 && !math.is_nan(value) && !math.is_inf(value, 0)
		small_df_total += value
	}
	assert math.abs(small_df_total / 8192.0 - 2.5) < 0.16

	mut central_rng := new_random_generator(757)
	central := central_rng.noncentral_chi_square(4.0, 0.0, [64])!
	mut chi_rng := new_random_generator(757)
	assert central.array_equal(chi_rng.chi_square(4.0, [64])!)

	mut f_rng := new_random_generator(758)
	f_samples := f_rng.noncentral_f(5.0, 30.0, 4.0, [8192])!
	mut f_total := 0.0
	for value in f_samples.to_array() {
		assert value >= 0 && !math.is_nan(value) && !math.is_inf(value, 0)
		f_total += value
	}
	assert math.abs(f_total / 8192.0 - (30.0 * 9.0 / (5.0 * 28.0))) < 0.06
	mut f_replay := new_random_generator(758)
	assert f_samples.array_equal(f_replay.noncentral_f(5.0, 30.0, 4.0, [8192])!)
	mut central_f_rng := new_random_generator(759)
	central_f := central_f_rng.noncentral_f(5.0, 30.0, 0.0, [64])!
	mut f_distribution_rng := new_random_generator(759)
	assert central_f.array_equal(f_distribution_rng.f_distribution(5.0, 30.0, [64])!)

	if _ := generator.noncentral_chi_square(0.0, 1.0, [1]) {
		assert false, 'noncentral_chi_square must reject non-positive degrees of freedom'
	}
	if _ := generator.noncentral_chi_square(2.0, -1.0, [1]) {
		assert false, 'noncentral_chi_square must reject negative noncentrality'
	}
	if _ := generator.noncentral_chi_square(2.0, math.nan(), [1]) {
		assert false, 'noncentral_chi_square must reject NaN noncentrality'
	}
	if _ := generator.noncentral_f(5.0, 0.0, 1.0, [1]) {
		assert false, 'noncentral_f must reject non-positive denominator degrees of freedom'
	}
	if _ := generator.noncentral_f(5.0, 30.0, -1.0, [1]) {
		assert false, 'noncentral_f must reject negative noncentrality'
	}
	generator.free()
	replay.free()
	small_df_rng.free()
	central_rng.free()
	chi_rng.free()
	f_rng.free()
	f_replay.free()
	central_f_rng.free()
	f_distribution_rng.free()
}

fn test_random_generator_distribution_calls_do_not_interfere_across_streams() ! {
	mut affected := new_random_generator(760)
	mut control := new_random_generator(760)
	mut other := new_random_generator(761)

	expected_zipf := control.zipf(4.0, [16])!
	_ := other.zipf(3.0, [32])!
	random_seed(762)
	_ := zipf(4.0, [32], TensorData{})!
	assert expected_zipf.array_equal(affected.zipf(4.0, [16])!)

	expected_logseries := control.logseries(0.5, [16])!
	_ := other.logseries(0.2, [32])!
	_ := logseries(0.5, [32], TensorData{})!
	assert expected_logseries.array_equal(affected.logseries(0.5, [16])!)

	expected_chi_square := control.noncentral_chi_square(4.0, 3.0, [16])!
	_ := other.noncentral_chi_square(4.0, 2.0, [32])!
	_ := noncentral_chi_square(4.0, 3.0, [32], TensorData{})!
	assert expected_chi_square.array_equal(affected.noncentral_chi_square(4.0, 3.0,
		[16])!)

	expected_f := control.noncentral_f(5.0, 30.0, 4.0, [16])!
	_ := other.noncentral_f(5.0, 20.0, 3.0, [32])!
	_ := noncentral_f(5.0, 30.0, 4.0, [32], TensorData{})!
	assert expected_f.array_equal(affected.noncentral_f(5.0, 30.0, 4.0, [16])!)

	affected.free()
	control.free()
	other.free()
}

fn test_random_generator_cauchy_is_seeded_and_matches_quartiles() ! {
	mut generator := new_random_generator(4242)
	values := generator.cauchy(1.0, 2.0, [4096])!
	assert values.shape == [4096]
	mut below_first_quartile := 0
	mut above_third_quartile := 0
	mut sorted := values.to_array()
	sorted.sort()
	median := sorted[sorted.len / 2]
	for value in values.to_array() {
		assert !math.is_nan(value) && !math.is_inf(value, 0)
		if value < -1.0 {
			below_first_quartile++
		}
		if value > 3.0 {
			above_third_quartile++
		}
	}
	// Cauchy(1, 2) has median 1 and quartiles at -1 and 3
	assert math.abs(median - 1.0) < 0.35
	assert math.abs(f64(below_first_quartile) / 4096.0 - 0.25) < 0.045
	assert math.abs(f64(above_third_quartile) / 4096.0 - 0.25) < 0.045

	mut replay := new_random_generator(4242)
	assert values.array_equal(replay.cauchy(1.0, 2.0, [4096])!)
	if _ := generator.cauchy(0.0, 0.0, [4]) {
		assert false, 'cauchy must reject a non-positive scale'
	}
	generator.free()
	replay.free()
}

fn test_random_generator_wald_is_seeded_and_matches_moments() ! {
	mut generator := new_random_generator(777)
	values := generator.wald(2.0, 3.0, [4096])!
	assert values.shape == [4096]
	mut total := 0.0
	for value in values.to_array() {
		assert value > 0 && !math.is_nan(value) && !math.is_inf(value, 0)
		total += value
	}
	mean := total / 4096.0
	mut squared_deviations := 0.0
	for value in values.to_array() {
		squared_deviations += (value - mean) * (value - mean)
	}
	variance := squared_deviations / 4096.0
	// Wald(2, 3) has mean 2 and variance mu^3 / lambda = 8/3
	assert math.abs(mean - 2.0) < 0.12
	assert math.abs(variance - 8.0 / 3.0) < 0.35

	mut replay := new_random_generator(777)
	assert values.array_equal(replay.wald(2.0, 3.0, [4096])!)
	if _ := generator.wald(0.0, 3.0, [4]) {
		assert false, 'wald must reject a non-positive mean'
	}
	if _ := generator.wald(2.0, -1.0, [4]) {
		assert false, 'wald must reject a non-positive scale'
	}
	generator.free()
	replay.free()
}

fn test_random_generator_vonmises_is_seeded_and_matches_circular_moments() ! {
	mut generator := new_random_generator(914)
	values := generator.vonmises(1.0, 2.0, [8192])!
	assert values.shape == [8192]
	mut cosine_total := 0.0
	mut sine_total := 0.0
	for value in values.to_array() {
		assert value >= -math.pi && value < math.pi
		cosine_total += math.cos(value - 1.0)
		sine_total += math.sin(value - 1.0)
	}
	assert math.abs(cosine_total / 8192.0 - 0.697774657964008) < 0.03
	assert math.abs(sine_total / 8192.0) < 0.03
	mut replay := new_random_generator(914)
	assert values.array_equal(replay.vonmises(1.0, 2.0, [8192])!)

	uniform := generator.vonmises(0.0, 0.0, [8192])!
	mut uniform_cosine_total := 0.0
	mut uniform_sine_total := 0.0
	for value in uniform.to_array() {
		uniform_cosine_total += math.cos(value)
		uniform_sine_total += math.sin(value)
	}
	assert math.abs(uniform_cosine_total / 8192.0) < 0.03
	assert math.abs(uniform_sine_total / 8192.0) < 0.03
	assert generator.vonmises(0.5, 1.0, [0])!.shape == [0]
	if _ := generator.vonmises(0.0, -1.0, [1]) {
		assert false, 'vonmises must reject negative concentration'
	}
	if _ := generator.vonmises(0.0, math.nan(), [1]) {
		assert false, 'vonmises must reject NaN concentration'
	}
	if _ := generator.vonmises(math.inf(1), 1.0, [1]) {
		assert false, 'vonmises must reject non-finite location'
	}
	if _ := generator.vonmises(0.0, 1e17, [1]) {
		assert false, 'vonmises must reject concentration beyond its stable range'
	}
	generator.free()
	replay.free()
}

fn test_random_generator_power_is_seeded_and_matches_moments() ! {
	mut generator := new_random_generator(2027)
	values := generator.power(2.0, [8192])!
	assert values.shape == [8192]
	mut total := 0.0
	for value in values.to_array() {
		assert value >= 0 && value <= 1
		total += value
	}
	mean := total / 8192.0
	mut squared_deviations := 0.0
	for value in values.to_array() {
		squared_deviations += (value - mean) * (value - mean)
	}
	variance := squared_deviations / 8192.0
	// Power(2) has mean 2/3 and variance 1/18.
	assert math.abs(mean - 2.0 / 3.0) < 0.012
	assert math.abs(variance - 1.0 / 18.0) < 0.004

	mut replay := new_random_generator(2027)
	assert values.array_equal(replay.power(2.0, [8192])!)
	assert generator.power(1.0, [0])!.shape == [0]
	if _ := generator.power(0.0, [1]) {
		assert false, 'power must reject a non-positive shape parameter'
	}
	if _ := generator.power(math.nan(), [1]) {
		assert false, 'power must reject a NaN shape parameter'
	}
	if _ := generator.power(math.inf(1), [1]) {
		assert false, 'power must reject an infinite shape parameter'
	}
	generator.free()
	replay.free()
}
