module vtl

import rand
import math

fn test_random_generator_supports_location_scale_distributions() ! {
	mut generator := new_random_generator(9401)
	gumbel_values := generator.gumbel(2.0, 1.5, [128])!
	laplace_values := generator.laplace(-1.0, 0.75, [128])!
	logistic_values := generator.logistic(0.5, 2.0, [128])!
	for values in [gumbel_values.to_array(), laplace_values.to_array(), logistic_values.to_array()] {
		for value in values {
			assert !math.is_nan(value) && !math.is_inf(value, 0)
		}
	}
	mut replay := new_random_generator(9401)
	assert gumbel_values.array_equal(replay.gumbel(2.0, 1.5, [128])!)
	assert laplace_values.array_equal(replay.laplace(-1.0, 0.75, [128])!)
	assert logistic_values.array_equal(replay.logistic(0.5, 2.0, [128])!)
	assert generator.gumbel(3.0, 0.0, [3])!.to_array() == [3.0, 3.0, 3.0]
	assert generator.laplace(-2.0, 0.0, [3])!.to_array() == [-2.0, -2.0, -2.0]
	assert generator.logistic(4.0, 0.0, [3])!.to_array() == [4.0, 4.0, 4.0]
	large_gumbel_sample := generator.gumbel(2.0, 1.5, [4096])!
	large_laplace_sample := generator.laplace(-1.0, 0.75, [4096])!
	large_logistic_sample := generator.logistic(0.5, 2.0, [4096])!
	mut gumbel_mean := 0.0
	mut laplace_mean := 0.0
	mut logistic_mean := 0.0
	for value in large_gumbel_sample.to_array() {
		gumbel_mean += value
	}
	for value in large_laplace_sample.to_array() {
		laplace_mean += value
	}
	for value in large_logistic_sample.to_array() {
		logistic_mean += value
	}
	assert math.abs(gumbel_mean / 4096 - (2.0 + 0.5772156649015329 * 1.5)) < 0.2
	assert math.abs(laplace_mean / 4096 + 1.0) < 0.1
	assert math.abs(logistic_mean / 4096 - 0.5) < 0.2
	if _ := generator.gumbel(math.nan(), 1.0, [1]) {
		assert false, 'gumbel must reject non-finite location'
	}
	if _ := generator.laplace(0.0, -1.0, [1]) {
		assert false, 'laplace must reject negative scale'
	}
	if _ := generator.logistic(0.0, math.inf(1), [1]) {
		assert false, 'logistic must reject non-finite scale'
	}
	generator.free()
	replay.free()
}

fn test_random_generator_supports_pareto_rayleigh_and_triangular() ! {
	mut generator := new_random_generator(9403)
	pareto_values := generator.pareto(3.0, [4096])!
	rayleigh_values := generator.rayleigh(2.0, [4096])!
	triangular_values := generator.triangular(0.0, 1.0, 3.0, [4096])!
	for value in pareto_values.to_array() {
		assert value >= 0 && !math.is_nan(value) && !math.is_inf(value, 0)
	}
	for value in rayleigh_values.to_array() {
		assert value >= 0 && !math.is_nan(value) && !math.is_inf(value, 0)
	}
	for value in triangular_values.to_array() {
		assert value >= 0 && value <= 3 && !math.is_nan(value) && !math.is_inf(value, 0)
	}
	mut pareto_mean := 0.0
	mut rayleigh_mean := 0.0
	mut triangular_mean := 0.0
	for value in pareto_values.to_array() {
		pareto_mean += value
	}
	for value in rayleigh_values.to_array() {
		rayleigh_mean += value
	}
	for value in triangular_values.to_array() {
		triangular_mean += value
	}
	assert math.abs(pareto_mean / 4096 - 0.5) < 0.1
	assert math.abs(rayleigh_mean / 4096 - 2.0 * math.sqrt(math.pi / 2)) < 0.08
	assert math.abs(triangular_mean / 4096 - 4.0 / 3.0) < 0.05
	mut replay := new_random_generator(9403)
	assert pareto_values.array_equal(replay.pareto(3.0, [4096])!)
	assert rayleigh_values.array_equal(replay.rayleigh(2.0, [4096])!)
	assert triangular_values.array_equal(replay.triangular(0.0, 1.0, 3.0, [4096])!)
	assert generator.rayleigh(0.0, [3])!.to_array() == [0.0, 0.0, 0.0]
	if _ := generator.pareto(0.0, [1]) {
		assert false, 'pareto must reject a non-positive shape parameter'
	}
	if _ := generator.rayleigh(-1.0, [1]) {
		assert false, 'rayleigh must reject negative scale'
	}
	if _ := generator.triangular(0.0, 4.0, 3.0, [1]) {
		assert false, 'triangular must reject a mode outside its endpoints'
	}
	generator.free()
	replay.free()
}

fn test_random_generator_poisson_is_seeded_and_handles_rate_edges() ! {
	mut generator := new_random_generator(146)
	mut replay := new_random_generator(146)
	zero_rate := generator.poisson(0.0, [4])!
	assert zero_rate.to_array() == [0, 0, 0, 0]
	small_rate := generator.poisson(4.0, [4096])!
	large_rate := generator.poisson(80.0, [4096])!
	assert zero_rate.array_equal(replay.poisson(0.0, [4])!)
	assert small_rate.array_equal(replay.poisson(4.0, [4096])!)
	assert large_rate.array_equal(replay.poisson(80.0, [4096])!)
	assert small_rate.shape == [4096]
	assert large_rate.shape == [4096]
	mut small_total := 0
	mut large_total := 0
	for sample in small_rate.to_array() {
		assert sample >= 0
		small_total += sample
	}
	for sample in large_rate.to_array() {
		assert sample >= 0
		large_total += sample
	}
	assert math.abs(f64(small_total) / 4096 - 4.0) < 0.2
	assert math.abs(f64(large_total) / 4096 - 80.0) < 1.0
	if _ := generator.poisson(-1.0, [1]) {
		assert false, 'poisson must reject a negative rate'
	}
	if _ := generator.poisson(math.nan(), [1]) {
		assert false, 'poisson must reject a NaN rate'
	}
	if _ := generator.poisson(math.inf(1), [1]) {
		assert false, 'poisson must reject an infinite rate'
	}
	generator.free()
	replay.free()
}
