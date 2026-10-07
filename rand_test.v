module vtl

import math

fn test_random_seed_repeats_random_tensor_values() {
	random_seed(42)
	first := random[f64](0.0, 1.0, [16], TensorData{})
	random_seed(42)
	second := random[f64](0.0, 1.0, [16], TensorData{})
	assert first.array_equal(second)

	random_seed(43)
	different_seed := random[f64](0.0, 1.0, [16], TensorData{})
	assert !first.array_equal(different_seed)
}

fn test_random_i32_values_stay_within_requested_range() {
	random_seed(42)
	values := random[i32](-20, 20, [32], TensorData{})
	for value in values.to_array() {
		assert value >= -20 && value < 20
	}
}

fn test_random_i16_and_u8_values_stay_within_requested_ranges() {
	random_seed(43)
	signed := random[i16](-200, 201, [64], TensorData{})
	for value in signed.to_array() {
		assert value >= -200 && value < 201
	}

	unsigned := random[u8](10, 200, [64], TensorData{})
	for value in unsigned.to_array() {
		assert value >= 10 && value < 200
	}
}

fn test_global_poisson_and_weibull_are_seeded_and_validated() ! {
	assert poisson(0.0, [4], TensorData{})!.to_array() == [0, 0, 0, 0]
	random_seed(620)
	small_rate := poisson(4.0, [4096], TensorData{})!
	large_rate := poisson(80.0, [4096], TensorData{})!
	random_seed(620)
	assert small_rate.array_equal(poisson(4.0, [4096], TensorData{})!)
	assert large_rate.array_equal(poisson(80.0, [4096], TensorData{})!)
	mut small_total := 0
	mut large_total := 0
	for value in small_rate.to_array() {
		assert value >= 0
		small_total += value
	}
	for value in large_rate.to_array() {
		assert value >= 0
		large_total += value
	}
	assert math.abs(f64(small_total) / 4096 - 4.0) < 0.2
	assert math.abs(f64(large_total) / 4096 - 80.0) < 1.0
	random_seed(621)
	weibull_values := weibull(1.5, [128], TensorData{})!
	random_seed(621)
	assert weibull_values.array_equal(weibull(1.5, [128], TensorData{})!)
	for value in weibull_values.to_array() {
		assert value >= 0 && !math.is_nan(value) && !math.is_inf(value, 0)
	}
	if _ := poisson(-1.0, [1], TensorData{}) {
		assert false, 'poisson must reject a negative rate'
	}
	if _ := poisson(math.nan(), [1], TensorData{}) {
		assert false, 'poisson must reject a NaN rate'
	}
	if _ := poisson(math.inf(1), [1], TensorData{}) {
		assert false, 'poisson must reject an infinite rate'
	}
	if _ := weibull(0.0, [1], TensorData{}) {
		assert false, 'weibull must reject a non-positive shape parameter'
	}
	if _ := weibull(math.inf(1), [1], TensorData{}) {
		assert false, 'weibull must reject an infinite shape parameter'
	}
}

fn test_global_statistical_distributions_are_seeded_and_validated() ! {
	random_seed(702)
	gamma_samples := gamma(2.0, 3.0, [4096], TensorData{})!
	beta_samples := beta(2.0, 5.0, [4096], TensorData{})!
	chi_samples := chi_square(4.0, [4096], TensorData{})!
	t_samples := student_t(12.0, [4096], TensorData{})!
	f_samples := f_distribution(5.0, 20.0, [4096], TensorData{})!
	random_seed(702)
	assert gamma_samples.array_equal(gamma(2.0, 3.0, [4096], TensorData{})!)
	assert beta_samples.array_equal(beta(2.0, 5.0, [4096], TensorData{})!)
	assert chi_samples.array_equal(chi_square(4.0, [4096], TensorData{})!)
	assert t_samples.array_equal(student_t(12.0, [4096], TensorData{})!)
	assert f_samples.array_equal(f_distribution(5.0, 20.0, [4096], TensorData{})!)
	for value in beta_samples.to_array() {
		assert value >= 0 && value <= 1 && !math.is_nan(value) && !math.is_inf(value, 0)
	}
	for value in gamma_samples.to_array() {
		assert value > 0 && !math.is_nan(value) && !math.is_inf(value, 0)
	}
	for value in chi_samples.to_array() {
		assert value > 0 && !math.is_nan(value) && !math.is_inf(value, 0)
	}
	for value in t_samples.to_array() {
		assert !math.is_nan(value) && !math.is_inf(value, 0)
	}
	for value in f_samples.to_array() {
		assert value > 0 && !math.is_nan(value) && !math.is_inf(value, 0)
	}
	assert math.abs(sample_mean_for_test(gamma_samples.to_array()) - 6.0) < 0.3
	assert math.abs(sample_mean_for_test(beta_samples.to_array()) - (2.0 / 7.0)) < 0.02
	assert math.abs(sample_mean_for_test(chi_samples.to_array()) - 4.0) < 0.2
	assert math.abs(sample_mean_for_test(t_samples.to_array())) < 0.1
	assert math.abs(sample_mean_for_test(f_samples.to_array()) - (20.0 / 18.0)) < 0.12
	if _ := gamma(0.0, 1.0, [1], TensorData{}) {
		assert false, 'gamma must reject a non-positive shape parameter'
	}
	if _ := beta(1.0, math.inf(1), [1], TensorData{}) {
		assert false, 'beta must reject a non-finite shape parameter'
	}
	if _ := chi_square(0.0, [1], TensorData{}) {
		assert false, 'chi_square must reject non-positive degrees of freedom'
	}
	if _ := student_t(math.nan(), [1], TensorData{}) {
		assert false, 'student_t must reject non-finite degrees of freedom'
	}
	if _ := f_distribution(1.0, 0.0, [1], TensorData{}) {
		assert false, 'f_distribution must reject non-positive denominator degrees of freedom'
	}
}

fn sample_mean_for_test(values []f64) f64 {
	mut total := 0.0
	for value in values {
		total += value
	}
	return total / values.len
}
