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

fn test_random_generator_supports_weibull_and_statistical_distributions() ! {
	mut generator := new_random_generator(264)
	mut replay := new_random_generator(264)
	weibull_values := generator.weibull(1.5, [4096])!
	chi_square_values := generator.chi_square(5.0, [4096])!
	student_t_values := generator.student_t(7.0, [4096])!
	f_values := generator.f_distribution(5.0, 10.0, [4096])!
	assert weibull_values.array_equal(replay.weibull(1.5, [4096])!)
	assert chi_square_values.array_equal(replay.chi_square(5.0, [4096])!)
	assert student_t_values.array_equal(replay.student_t(7.0, [4096])!)
	assert f_values.array_equal(replay.f_distribution(5.0, 10.0, [4096])!)
	mut chi_square_total := 0.0
	mut student_t_total := 0.0
	mut f_total := 0.0
	for value in weibull_values.to_array() {
		assert value >= 0 && !math.is_nan(value) && !math.is_inf(value, 0)
	}
	for value in chi_square_values.to_array() {
		assert value > 0 && !math.is_nan(value) && !math.is_inf(value, 0)
		chi_square_total += value
	}
	for value in student_t_values.to_array() {
		assert !math.is_nan(value) && !math.is_inf(value, 0)
		student_t_total += value
	}
	for value in f_values.to_array() {
		assert value > 0 && !math.is_nan(value) && !math.is_inf(value, 0)
		f_total += value
	}
	assert math.abs(chi_square_total / 4096 - 5.0) < 0.25
	assert math.abs(student_t_total / 4096) < 0.08
	assert math.abs(f_total / 4096 - 1.25) < 0.12
	if _ := generator.weibull(0.0, [1]) {
		assert false, 'weibull must reject a non-positive shape parameter'
	}
	if _ := generator.chi_square(math.inf(1), [1]) {
		assert false, 'chi_square must reject infinite degrees of freedom'
	}
	if _ := generator.student_t(-1.0, [1]) {
		assert false, 'student_t must reject non-positive degrees of freedom'
	}
	if _ := generator.f_distribution(2.0, math.nan(), [1]) {
		assert false, 'f_distribution must reject NaN degrees of freedom'
	}
	generator.free()
	replay.free()
}

fn test_random_generator_supports_seeded_multinomial_counts() ! {
	probabilities := from_1d([0.2, 0.3, 0.5])!
	mut generator := new_random_generator(357)
	counts := generator.multinomial(20, probabilities, [64])!
	assert counts.shape == [64, 3]
	for sample in 0 .. counts.shape[0] {
		mut total := 0
		for category in 0 .. counts.shape[1] {
			count := counts.get([sample, category])
			assert count >= 0
			total += count
		}
		assert total == 20
	}
	mut replay := new_random_generator(357)
	assert counts.array_equal(replay.multinomial(20, probabilities, [64])!)
	assert generator.multinomial(5, from_1d([1.0, 0.0])!, [2])!.to_array() == [5, 0, 5, 0]
	if _ := generator.multinomial(5, from_1d([0.2, 0.2])!, [1]) {
		assert false, 'multinomial must reject probabilities whose sum is not one'
	}
	generator.free()
	replay.free()
}

fn test_random_generator_choice_supports_replacement_modes() ! {
	population := from_1d([10, 20, 30, 40, 50])!
	mut generator := new_random_generator(456)
	without_replacement := generator.choice[int](population, 5, false)!
	assert without_replacement.shape == [5]
	for i in 0 .. without_replacement.size {
		for j in i + 1 .. without_replacement.size {
			assert without_replacement.get_nth(i) != without_replacement.get_nth(j)
		}
	}
	single_value := from_1d([7])!
	with_replacement := generator.choice[int](single_value, 8, true)!
	for i in 0 .. with_replacement.size {
		assert with_replacement.get_nth(i) == 7
	}
	mut replay := new_random_generator(456)
	replayed := replay.choice[int](population, 5, false)!
	assert without_replacement.array_equal(replayed)
	generator.free()
	replay.free()
}

fn test_random_generator_weighted_choice_respects_zero_weights_and_seed() ! {
	population := from_1d(['excluded', 'selected', 'also_excluded'])!
	weights := from_1d([0.0, 3.0, 0.0])!
	mut generator := new_random_generator(852)
	with_replacement := generator.choice_weighted[string](population, weights, 8, true)!
	for value in with_replacement.to_array() {
		assert value == 'selected'
	}
	without_replacement := generator.choice_weighted[string](population, weights, 1, false)!
	assert without_replacement.get_nth(0) == 'selected'
	mut replay := new_random_generator(852)
	assert with_replacement.array_equal(replay.choice_weighted[string](population, weights, 8, true)!)
	if _ := generator.choice_weighted[string](population, weights, 2, false) {
		assert false, 'weighted sampling without replacement must require positive weights'
	}
	if _ := generator.choice_weighted[string](population, from_1d([1.0, -1.0, 1.0])!, 1, true) {
		assert false, 'weighted choice must reject negative weights'
	}
	if _ := generator.choice_weighted[string](population, from_1d([0.0, 0.0, 0.0])!, 1, true) {
		assert false, 'weighted choice must reject zero total weight'
	}
	generator.free()
	replay.free()
}

fn test_random_generator_permutation_is_seeded_and_complete() ! {
	mut generator := new_random_generator(753)
	permuted := generator.permutation(32)!
	assert permuted.shape == [32]
	mut sorted := permuted.to_array()
	sorted.sort()
	assert sorted == irange(0, 32)
	mut replay := new_random_generator(753)
	assert permuted.array_equal(replay.permutation(32)!)
	assert generator.permutation(0)!.size == 0
	if _ := generator.permutation(-1) {
		assert false, 'permutation must reject a negative size'
	}
	generator.free()
	replay.free()
}

fn test_random_generator_permutation_tensor_shuffles_complete_rows() ! {
	input := from_array([10, 11, 20, 21, 30, 31], [3, 2])!
	mut generator := new_random_generator(951)
	permuted := generator.permutation_tensor[int](input)!
	assert permuted.shape == input.shape
	mut first_column := []int{len: permuted.shape[0]}
	for row in 0 .. permuted.shape[0] {
		first := permuted.get([row, 0])
		assert permuted.get([row, 1]) == first + 1
		first_column[row] = first
	}
	first_column.sort()
	assert first_column == [10, 20, 30]
	mut replay := new_random_generator(951)
	assert permuted.array_equal(replay.permutation_tensor[int](input)!)
	if _ := generator.permutation_tensor[int](from_array[int]([1], []int{})!) {
		assert false, 'permutation_tensor must reject scalar tensors'
	}
	generator.free()
	replay.free()
}

fn test_random_generator_permutation_axis_shuffles_complete_slices() ! {
	mut values := []int{}
	for i in 0 .. 2 {
		for j in 0 .. 3 {
			for k in 0 .. 2 {
				values << i * 100 + j * 10 + k
			}
		}
	}
	input := from_array(values, [2, 3, 2])!
	mut generator := new_random_generator(952)
	permuted := generator.permutation_axis[int](input, -2)!
	assert permuted.shape == input.shape
	for i in 0 .. 2 {
		for k in 0 .. 2 {
			mut slice_ids := []int{len: 3}
			for j in 0 .. 3 {
				slice_ids[j] = permuted.get([i, j, k]) / 10
			}
			slice_ids.sort()
			assert slice_ids == [i * 10, i * 10 + 1, i * 10 + 2]
		}
	}
	mut replay := new_random_generator(952)
	assert permuted.array_equal(replay.permutation_axis[int](input, 1)!)
	if _ := generator.permutation_axis[int](input, 3) {
		assert false, 'permutation_axis must reject axes outside the input rank'
	}
	if _ := generator.permutation_axis[int](from_array[int]([1], []int{})!, 0) {
		assert false, 'permutation_axis must reject scalar tensors'
	}
	generator.free()
	replay.free()
}

fn test_random_generator_choice_axis_samples_complete_slices() ! {
	mut values := []int{}
	for i in 0 .. 2 {
		for j in 0 .. 3 {
			for k in 0 .. 2 {
				values << i * 100 + j * 10 + k
			}
		}
	}
	input := from_array(values, [2, 3, 2])!
	mut generator := new_random_generator(953)
	sampled := generator.choice_axis[int](input, 2, -2, false)!
	assert sampled.shape == [2, 2, 2]
	for i in 0 .. 2 {
		for k in 0 .. 2 {
			mut selected_ids := []int{len: 2}
			for j in 0 .. 2 {
				selected_ids[j] = sampled.get([i, j, k]) / 10
			}
			selected_ids.sort()
			assert selected_ids[0] != selected_ids[1]
		}
	}
	mut replay := new_random_generator(953)
	assert sampled.array_equal(replay.choice_axis[int](input, 2, 1, false)!)
	weights := from_1d([1.0, 0.0, 0.0])!
	weighted := generator.choice_weighted_axis[int](input, weights, 1, 1, false)!
	assert weighted.to_array() == [0, 1, 100, 101]
	if _ := generator.choice_axis[int](input, 4, 1, false) {
		assert false, 'choice_axis must reject oversized samples without replacement'
	}
	if _ := generator.choice_weighted_axis[int](input, from_1d([1.0, 1.0])!, 1, 1, true) {
		assert false, 'choice_weighted_axis must reject weights with the wrong length'
	}
	generator.free()
	replay.free()
}

fn test_random_generator_gamma_is_seeded_and_positive() ! {
	mut generator := new_random_generator(789)
	values := generator.gamma(2.0, 3.0, [64])!
	assert values.shape == [64]
	for i in 0 .. values.size {
		assert values.get_nth(i) > 0
	}
	mut replay := new_random_generator(789)
	assert values.array_equal(replay.gamma(2.0, 3.0, [64])!)
	if _ := generator.gamma(0.0, 1.0, [2]) {
		assert false, 'gamma must reject a non-positive shape parameter'
	}
	generator.free()
	replay.free()
}

fn test_random_generator_beta_is_seeded_and_bounded() ! {
	mut generator := new_random_generator(321)
	values := generator.beta(2.0, 5.0, [64])!
	assert values.shape == [64]
	for i in 0 .. values.size {
		assert values.get_nth(i) >= 0 && values.get_nth(i) <= 1
	}
	mut replay := new_random_generator(321)
	assert values.array_equal(replay.beta(2.0, 5.0, [64])!)
	if _ := generator.beta(0.0, 1.0, [2]) {
		assert false, 'beta must reject a non-positive shape parameter'
	}
	generator.free()
	replay.free()
}

fn test_random_generator_rejects_invalid_distribution_parameters() {
	mut generator := new_random_generator(1)
	if _ := generator.uniform(1.0, 0.0, [2]) {
		assert false, 'uniform must reject a reversed range'
	}
	if _ := generator.normal([2], NormalTensorData{ sigma: -1.0 }) {
		assert false, 'normal must reject a negative standard deviation'
	}
	if _ := generator.bernoulli(1.1, [2]) {
		assert false, 'bernoulli must reject probabilities above one'
	}
	if _ := generator.geometric(0.0, [2]) {
		assert false, 'geometric must reject a zero probability'
	}
	if _ := generator.choice[int](from_1d([1, 2])!, 3, false) {
		assert false, 'choice must reject oversampling without replacement'
	}
	generator.free()
}

fn test_random_generator_does_not_change_v_global_random_stream() ! {
	random_seed(31415)
	expected_global_value := rand.f64()
	random_seed(31415)
	mut independent := new_random_generator(2718)
	_ := independent.uniform(0.0, 1.0, [16])!
	observed_global_value := rand.f64()
	assert observed_global_value == expected_global_value
	independent.free()
}
