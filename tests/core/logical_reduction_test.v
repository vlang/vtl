module core

import vtl

fn test_logical_axis_reductions_match_numpy_shapes_and_values() ! {
	values := vtl.from_array([0, 1, 2, 0, 3, 4], [2, 3])!
	all_rows := values.all_axis(1, false)!
	assert all_rows.shape == [2]
	assert all_rows.to_array() == [false, false]
	any_columns := values.any_axis(-2, true)!
	assert any_columns.shape == [1, 3]
	assert any_columns.to_array() == [false, true, true]
	assert values.any_axis(1, false)!.to_array() == [true, true]
	vector := vtl.from_1d([0, 2, 0])!
	assert vector.any_axis(0, false)!.to_array() == [true]
}

fn test_logical_axis_reductions_empty_identity_and_empty_outer_shape() ! {
	values := vtl.from_array([]int{}, [2, 0])!
	assert values.all_axis(1, false)!.to_array() == [true, true]
	assert values.any_axis(1, true)!.shape == [2, 1]
	assert values.any_axis(1, true)!.to_array() == [false, false]
	no_slices := vtl.from_array([]int{}, [0, 3])!
	assert no_slices.all_axis(1, false)!.shape == [0]
}

fn test_logical_axis_reductions_reject_invalid_axes() ! {
	values := vtl.from_1d([1, 0])!
	if _ := values.any_axis(1, false) {
		assert false, 'out-of-range axes must return an error'
	}
	if _ := values.all_axis(-2, true) {
		assert false, 'out-of-range axes must return an error'
	}
	if _ := vtl.from_array([true], []int{})!.all_axis(0, false) {
		assert false, 'scalar axis reductions must return an error'
	}
}

fn test_logical_multi_axis_reductions_and_validation() ! {
	values := vtl.from_array([0, 1, 2, 3, 0, 4, 5, 6], [2, 2, 2])!
	all := values.all_axes([0, -1], false)!
	assert all.shape == [2]
	assert all.to_array() == [false, true]
	any_keepdims := values.any_axes([0, 2], true)!
	assert any_keepdims.shape == [1, 2, 1]
	assert any_keepdims.to_array() == [true, true]
	assert values.all_axes([], false)!.to_array() == [false, true, true, true, false, true, true,
		true]
	if _ := values.any_axes([0, -3], false) {
		assert false, 'duplicate axes must return an error'
	}
}
