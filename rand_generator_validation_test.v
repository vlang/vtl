module vtl

import rand
import math

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
