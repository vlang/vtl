module vtl

import rand
import math

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
