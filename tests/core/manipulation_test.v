module core

import vtl

fn test_roll_flat_wraps_positive_negative_and_large_shifts() {
	t := vtl.from_1d([0, 1, 2, 3])!
	assert vtl.roll(t, 1)!.to_array() == [3, 0, 1, 2]
	assert vtl.roll(t, -1)!.to_array() == [1, 2, 3, 0]
	assert vtl.roll(t, 6)!.to_array() == [2, 3, 0, 1]
}

fn test_roll_one_axis_and_scalar_shift_across_axes() {
	t := vtl.from_array([0, 1, 2, 3, 4, 5], [2, 3])!
	assert vtl.roll(t, 1, 1)!.to_array() == [2, 0, 1, 5, 3, 4]
	assert vtl.roll(t, 1, 0, 1)!.to_array() == [5, 3, 4, 2, 0, 1]
	assert vtl.roll(t, 1, -2, -1)!.to_array() == [5, 3, 4, 2, 0, 1]
}

fn test_roll_shifts_accumulate_repeated_axes_and_support_strided_views() {
	t := vtl.from_array([0, 1, 2, 3, 4, 5], [2, 3])!
	assert vtl.roll_shifts(t, [1, 2], [0, 0])!.to_array() == [3, 4, 5, 0, 1, 2]
	transposed := t.transpose([1, 0])!
	assert vtl.roll(transposed, 1, 0)!.to_array() == [2, 5, 0, 3, 1, 4]
}

fn test_roll_empty_and_scalar_tensors() {
	empty := vtl.from_array([]int{}, [0, 3])!
	assert vtl.roll(empty, 7)!.shape == [0, 3]
	assert vtl.roll(empty, -2, 1)!.shape == [0, 3]
	scalar := vtl.from_array([42], []int{})!
	assert vtl.roll(scalar, 19)!.shape.len == 0
	assert vtl.roll(scalar, 19)!.get_nth(0) == 42
}

fn test_roll_rejects_invalid_axes_and_shift_count() {
	t := vtl.from_array([0, 1, 2, 3], [2, 2])!
	if _ := vtl.roll(t, 1, 2) {
		assert false, 'roll must reject an out-of-range axis'
	}
	if _ := vtl.roll_shifts(t, [1], [0, 1]) {
		assert false, 'roll_shifts must reject mismatched shifts and axes'
	}
}
