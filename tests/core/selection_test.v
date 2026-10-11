module core

import vtl

fn test_choose_broadcasts_indices_and_choice_tensors() ! {
	indices := vtl.from_2d([[0], [1]])!
	first := vtl.from_2d([[10, 20, 30]])!
	second := vtl.from_2d([[100], [200]])!
	got := vtl.choose(indices, [first, second])!
	assert got.shape == [2, 3]
	assert got.to_array() == [10, 20, 30, 200, 200, 200]
}

fn test_choose_uses_flat_choice_indices() ! {
	indices := vtl.from_1d([1, 0, 1])!
	first := vtl.from_1d([2, 4, 6])!
	second := vtl.from_1d([3, 5, 7])!
	got := vtl.choose(indices, [first, second])!
	assert got.to_array() == [3, 4, 7]
}

fn test_choose_rejects_empty_choices_and_invalid_indices() ! {
	indices := vtl.from_1d([0, 2])!
	choice := vtl.from_1d([10, 20])!
	empty_choices := []&vtl.Tensor[int]{}
	if _ := vtl.choose(indices, empty_choices) {
		assert false, 'choose must reject an empty choices list'
	} else {
		assert true
	}
	if _ := vtl.choose(indices, [choice]) {
		assert false, 'choose must reject an index beyond the choices list'
	} else {
		assert true
	}
	negative := vtl.from_1d([-1])!
	if _ := vtl.choose(negative, [choice]) {
		assert false, 'choose must reject negative indices'
	} else {
		assert true
	}
}

fn test_where_selects_and_broadcasts() {
	condition := vtl.from_array([true, false, false, true], [2, 2])!
	x := vtl.from_1d([1, 2])!
	y := vtl.from_2d([[10, 20], [30, 40]])!
	got := vtl.where(condition, x, y)!
	expected := vtl.from_2d([[1, 20], [30, 2]])!
	assert got.array_equal(expected)
}

fn test_where_rejects_incompatible_shapes() {
	condition := vtl.from_1d([true, false, true])!
	x := vtl.from_1d([1, 2])!
	y := vtl.from_1d([3, 4])!
	if _ := vtl.where(condition, x, y) {
		assert false, 'incompatible shapes must return an error'
	} else {
		assert true
	}
}

fn test_where_handles_scalar_choices() {
	condition := vtl.from_1d([true, false])!
	x := vtl.tensor(7, [], memory: .row_major)
	y := vtl.from_1d([3, 4])!
	got := vtl.where(condition, x, y)!
	expected := vtl.from_1d([7, 4])!
	assert got.array_equal(expected)
}

fn test_select_broadcasts_masks_and_choices_with_first_match_priority() ! {
	first_condition := vtl.from_2d([[true], [false]])!
	second_condition := vtl.from_2d([[false, true, true]])!
	first_choice := vtl.from_2d([[10], [20]])!
	second_choice := vtl.from_2d([[100, 200, 300]])!
	mut conditions := []&vtl.Tensor[bool]{cap: 2}
	conditions << first_condition
	conditions << second_condition
	mut choices := []&vtl.Tensor[int]{cap: 2}
	choices << first_choice
	choices << second_choice
	got := vtl.select(conditions, choices, -1)!
	expected := vtl.from_2d([[10, 10, 10], [-1, 200, 300]])!
	assert got.array_equal(expected)
}

fn test_select_uses_default_when_no_condition_matches() ! {
	first_condition := vtl.from_1d([false, true, false])!
	second_condition := vtl.from_1d([false, false, true])!
	first_choice := vtl.from_1d([10, 20, 30])!
	second_choice := vtl.tensor(7, [], memory: .row_major)
	mut conditions := []&vtl.Tensor[bool]{cap: 2}
	conditions << first_condition
	conditions << second_condition
	mut choices := []&vtl.Tensor[int]{cap: 2}
	choices << first_choice
	choices << second_choice
	got := vtl.select(conditions, choices, -1)!
	assert got.to_array() == [-1, 20, 7]
}

fn test_select_rejects_empty_or_mismatched_inputs() ! {
	condition := vtl.from_1d([true, false])!
	choice := vtl.from_1d([1, 2])!
	no_conditions := []&vtl.Tensor[bool]{}
	no_choices := []&vtl.Tensor[int]{}
	if _ := vtl.select(no_conditions, no_choices, 0) {
		assert false, 'select must reject empty conditions and choices'
	} else {
		assert true
	}
	if _ := vtl.select([condition], no_choices, 0) {
		assert false, 'select must reject mismatched condition and choice counts'
	} else {
		assert true
	}
	if _ := vtl.select([condition], [choice, choice], 0) {
		assert false, 'select must reject mismatched condition and choice counts'
	} else {
		assert true
	}
}

fn test_select_rejects_incompatible_broadcast_shapes() ! {
	condition := vtl.from_1d([true, false, true])!
	choice := vtl.from_1d([1, 2])!
	if _ := vtl.select([condition], [choice], 0) {
		assert false, 'select must reject incompatible shapes'
	} else {
		assert true
	}
}
