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
