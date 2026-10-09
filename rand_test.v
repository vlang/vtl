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

fn test_global_hypergeometric_is_seeded_and_validated() ! {
	assert hypergeometric(8, 0, 3, [4], TensorData{})!.to_array() == [3, 3, 3, 3]
	assert hypergeometric(8, 5, 0, [2], TensorData{})!.to_array() == [0, 0]
	random_seed(512)
	values := hypergeometric(5, 5, 4, [2048], TensorData{})!
	random_seed(512)
	assert values.array_equal(hypergeometric(5, 5, 4, [2048], TensorData{})!)
	for value in values.to_array() {
		assert value >= 0 && value <= 4
	}
	large_sample := hypergeometric(50, 50, 80, [512], TensorData{})!
	mut large_sample_total := 0
	for value in large_sample.to_array() {
		assert value >= 30 && value <= 50
		large_sample_total += value
	}
	assert math.abs(f64(large_sample_total) / f64(large_sample.size) - 40.0) < 0.35
	reflected_sample := hypergeometric(80, 20, 60, [512], TensorData{})!
	mut reflected_total := 0
	for value in reflected_sample.to_array() {
		assert value >= 40 && value <= 60
		reflected_total += value
	}
	assert math.abs(f64(reflected_total) / f64(reflected_sample.size) - 48.0) < 0.4
	if _ := hypergeometric(1, -1, 0, [1], TensorData{}) {
		assert false, 'hypergeometric must reject negative population counts'
	}
	if _ := hypergeometric(1, 2, 4, [1], TensorData{}) {
		assert false, 'hypergeometric must reject samples larger than the population'
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

fn test_global_location_scale_distributions_are_seeded_and_validated() ! {
	random_seed(9402)
	gumbel_values := gumbel(2.0, 1.5, [128], TensorData{})!
	laplace_values := laplace(-1.0, 0.75, [128], TensorData{})!
	logistic_values := logistic(0.5, 2.0, [128], TensorData{})!
	random_seed(9402)
	assert gumbel_values.array_equal(gumbel(2.0, 1.5, [128], TensorData{})!)
	assert laplace_values.array_equal(laplace(-1.0, 0.75, [128], TensorData{})!)
	assert logistic_values.array_equal(logistic(0.5, 2.0, [128], TensorData{})!)
	for values in [gumbel_values.to_array(), laplace_values.to_array(), logistic_values.to_array()] {
		for value in values {
			assert !math.is_nan(value) && !math.is_inf(value, 0)
		}
	}
	assert gumbel(3.0, 0.0, [2], TensorData{})!.to_array() == [3.0, 3.0]
	assert laplace(-2.0, 0.0, [2], TensorData{})!.to_array() == [-2.0, -2.0]
	assert logistic(4.0, 0.0, [2], TensorData{})!.to_array() == [4.0, 4.0]
	if _ := gumbel(0.0, -1.0, [1], TensorData{}) {
		assert false, 'gumbel must reject negative scale'
	}
	if _ := laplace(math.inf(1), 1.0, [1], TensorData{}) {
		assert false, 'laplace must reject non-finite location'
	}
	if _ := logistic(0.0, math.nan(), [1], TensorData{}) {
		assert false, 'logistic must reject NaN scale'
	}
}

fn test_global_pareto_rayleigh_and_triangular_are_seeded_and_validated() ! {
	random_seed(9404)
	pareto_values := pareto(3.0, [4096], TensorData{})!
	rayleigh_values := rayleigh(2.0, [4096], TensorData{})!
	triangular_values := triangular(0.0, 1.0, 3.0, [4096], TensorData{})!
	random_seed(9404)
	assert pareto_values.array_equal(pareto(3.0, [4096], TensorData{})!)
	assert rayleigh_values.array_equal(rayleigh(2.0, [4096], TensorData{})!)
	assert triangular_values.array_equal(triangular(0.0, 1.0, 3.0, [4096], TensorData{})!)
	for value in pareto_values.to_array() {
		assert value >= 0 && !math.is_nan(value) && !math.is_inf(value, 0)
	}
	for value in rayleigh_values.to_array() {
		assert value >= 0 && !math.is_nan(value) && !math.is_inf(value, 0)
	}
	for value in triangular_values.to_array() {
		assert value >= 0 && value <= 3 && !math.is_nan(value) && !math.is_inf(value, 0)
	}
	assert math.abs(sample_mean_for_test(pareto_values.to_array()) - 0.5) < 0.1
	assert math.abs(sample_mean_for_test(rayleigh_values.to_array()) - 2 * math.sqrt(math.pi / 2)) < 0.08
	assert math.abs(sample_mean_for_test(triangular_values.to_array()) - 4.0 / 3.0) < 0.05
	assert rayleigh(0.0, [2], TensorData{})!.to_array() == [0.0, 0.0]
	if _ := pareto(math.inf(1), [1], TensorData{}) {
		assert false, 'pareto must reject non-finite shape parameters'
	}
	if _ := rayleigh(math.nan(), [1], TensorData{}) {
		assert false, 'rayleigh must reject NaN scale'
	}
	if _ := triangular(2.0, 1.0, 3.0, [1], TensorData{}) {
		assert false, 'triangular must require left <= mode <= right'
	}
}

fn sample_mean_for_test(values []f64) f64 {
	mut total := 0.0
	for value in values {
		total += value
	}
	return total / values.len
}

fn test_global_lognormal_and_dirichlet_are_seeded_and_validated() ! {
	random_seed(703)
	lognormal_samples := lognormal(0.5, 0.75, [2048], TensorData{})!
	constant_samples := lognormal(1.25, 0.0, [4], TensorData{})!
	concentrations := from_array[f64]([1.0, 2.0, 3.0], [3])!
	dirichlet_samples := dirichlet(concentrations, [1024], TensorData{})!
	random_seed(703)
	assert lognormal_samples.array_equal(lognormal(0.5, 0.75, [2048], TensorData{})!)
	assert dirichlet_samples.array_equal(dirichlet(concentrations, [1024], TensorData{})!)
	for value in constant_samples.to_array() {
		assert value == math.exp(1.25)
	}
	for value in lognormal_samples.to_array() {
		assert value > 0 && !math.is_nan(value) && !math.is_inf(value, 0)
	}
	assert math.abs(sample_mean_for_test(lognormal_samples.to_array()) - math.exp(0.5 + 0.75 * 0.75 / 2)) < 0.15
	assert dirichlet_samples.shape == [1024, 3]
	values := dirichlet_samples.to_array()
	mut category_totals := [3]f64{}
	for sample in 0 .. 1024 {
		mut total := 0.0
		for category in 0 .. 3 {
			value := values[sample * 3 + category]
			assert value >= 0 && value <= 1 && !math.is_nan(value) && !math.is_inf(value, 0)
			category_totals[category] += value
			total += value
		}
		assert math.abs(total - 1.0) < 1e-12
	}
	assert math.abs(category_totals[0] / 1024 - (1.0 / 6.0)) < 0.02
	assert math.abs(category_totals[1] / 1024 - (2.0 / 6.0)) < 0.02
	assert math.abs(category_totals[2] / 1024 - (3.0 / 6.0)) < 0.02
	if _ := lognormal(0.0, -1.0, [1], TensorData{}) {
		assert false, 'lognormal must reject negative sigma'
	}
	invalid_concentrations := from_array[f64]([1.0, 0.0], [2])!
	if _ := dirichlet(invalid_concentrations, [1], TensorData{}) {
		assert false, 'dirichlet must reject non-positive concentrations'
	}
	rank_two_concentrations := from_array[f64]([1.0], [1, 1])!
	if _ := dirichlet(rank_two_concentrations, [1], TensorData{}) {
		assert false, 'dirichlet must require a vector of concentrations'
	}
}
