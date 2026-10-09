module core

import vtl

fn test_boolean_index_selects_matching_prefix_and_preserves_trailing_axes() ! {
	values := vtl.from_array[int]([]int{len: 24, init: index}, [2, 3, 4])!
	mask := vtl.from_2d([[true, false, true], [false, true, false]])!
	mut selected := vtl.boolean_index[int](values, mask)!
	assert selected.shape == [3, 4]
	assert selected.to_array() == [0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19]
	selected.set([0, 0], 99)
	assert values.get([0, 0, 0]) == 0
}

fn test_boolean_index_accepts_non_contiguous_masks_and_empty_matches() ! {
	values := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	mask := vtl.from_2d([[true, false], [false, true], [true, false]])!.transpose([1, 0])!
	selected := vtl.boolean_index[int](values, mask)!
	assert selected.shape == [3]
	assert selected.to_array() == [1, 3, 5]
	empty_mask := vtl.from_1d([false, false])!
	empty := vtl.boolean_index[int](values, empty_mask)!
	assert empty.shape == [0, 3]
	assert empty.size == 0
}

fn test_boolean_index_rejects_non_prefix_shapes() ! {
	values := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	wrong_shape := vtl.from_1d([true, false, true])!
	if _ := vtl.boolean_index[int](values, wrong_shape) {
		assert false, 'boolean_index must require a mask matching leading tensor dimensions'
	}
	if _ := vtl.boolean_index[int](values, vtl.from_1d([true, false, true])!.reshape([
		3,
		1,
	])!) {
		assert false, 'boolean_index must reject dimensions that do not match the tensor prefix'
	}
}

fn test_mixed_index_pairs_coordinates_with_slice() ! {
	values := vtl.from_array[int]([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17,
		18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34],
		[5, 7])!
	rows := vtl.from_1d([0, 2, 4])!
	columns := vtl.slice_index(1, 3, 1)!

	mut selected := vtl.mixed_index[int](values, [vtl.array_index(rows), columns])!
	assert selected.shape == [3, 2]
	assert selected.to_array() == [1, 2, 15, 16, 29, 30]
	selected.set([0, 0], 99)
	assert values.get([0, 1]) == 1
}

fn test_mixed_index_separated_coordinate_axes_move_to_front() ! {
	values := vtl.from_array[int]([]int{len: 24, init: index}, [2, 3, 4])!
	outer := vtl.from_1d([0, 1])!
	inner := vtl.from_1d([1, 2])!
	all_rows := vtl.slice_all(1)!

	selected := vtl.mixed_index[int](values, [vtl.array_index(outer), all_rows, vtl.array_index(inner)])!
	assert selected.shape == [2, 3]
	assert selected.to_array() == [1, 5, 9, 14, 18, 22]
}

fn test_mixed_index_contiguous_coordinates_preserve_slice_axis_order() ! {
	values := vtl.from_array[int]([]int{len: 120, init: index}, [3, 2, 4, 5])!
	rows := vtl.from_1d([0, 1])!
	columns := vtl.from_1d([1, 2])!
	row_slice := vtl.slice_index(1, 3, 1)!
	column_slice := vtl.slice_index(2, 4, 1)!

	selected := vtl.mixed_index[int](values, [row_slice, vtl.array_index(rows),
		vtl.array_index(columns), column_slice])!
	assert selected.shape == [2, 2, 2]
	assert selected.to_array() == [47, 48, 72, 73, 87, 88, 112, 113]
}

fn test_mixed_index_broadcasts_multidimensional_coordinate_tensors() ! {
	values := vtl.from_array[int]([]int{len: 15, init: index}, [3, 5])!
	rows := vtl.from_array[int]([0, 2], [2, 1])!
	columns := vtl.from_array[int]([1, 2, 4], [1, 3])!
	selected := vtl.mixed_index[int](values, [vtl.array_index(rows), vtl.array_index(columns)])!
	assert selected.shape == [2, 3]
	assert selected.to_array() == [1, 2, 4, 11, 12, 14]
}

fn test_mixed_index_integer_between_coordinate_axes_is_contiguous() ! {
	values := vtl.from_array[int]([]int{len: 60, init: index}, [3, 4, 5])!
	outer := vtl.from_1d([0, 2])!
	inner := vtl.from_1d([1, 3])!
	selected := vtl.mixed_index[int](values, [vtl.array_index(outer), vtl.integer_index(1),
		vtl.array_index(inner)])!
	assert selected.shape == [2]
	assert selected.to_array() == [6, 48]
}

fn test_mixed_index_integer_selects_a_view_and_normalizes_negative_index() ! {
	mut values := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	mut row := vtl.mixed_index[int](values, [vtl.integer_index(-1)])!
	assert row.shape == [3]
	assert row.to_array() == [4, 5, 6]
	row.set([0], 40)
	assert values.get([1, 0]) == 40
	mut scalar := vtl.mixed_index[int](values, [vtl.integer_index(1), vtl.integer_index(2)])!
	assert scalar.shape == []
	assert scalar.get_nth(0) == 6
	scalar.set_nth(0, 60)
	assert values.get([1, 2]) == 60
}

fn test_mixed_index_negative_slice_step_and_empty_slice() ! {
	values := vtl.from_1d([1, 2, 3, 4])!
	reversed := vtl.mixed_index[int](values, [vtl.slice_all(-1)!])!
	assert reversed.to_array() == [4, 3, 2, 1]
	large_negative_step := vtl.mixed_index[int](values, [vtl.slice_all(min_int)!])!
	assert large_negative_step.to_array() == [4]
	empty := vtl.mixed_index[int](values, [vtl.slice_index(2, 2, 1)!])!
	assert empty.shape == [0]
	assert empty.size == 0
}

fn test_mixed_index_rejects_invalid_indices() ! {
	values := vtl.from_2d([[1, 2], [3, 4]])!
	if _ := vtl.slice_all(0) {
		assert false, 'slice_index must reject a zero step'
	}
	if _ := vtl.mixed_index[int](values, [vtl.integer_index(2)]) {
		assert false, 'mixed_index must reject an out-of-range integer'
	}
	if _ := vtl.mixed_index[int](values, [vtl.integer_index(min_int)]) {
		assert false, 'mixed_index must reject an extreme negative index safely'
	}
	if _ := vtl.mixed_index[int](values, [vtl.array_index(vtl.from_1d([2])!)]) {
		assert false, 'mixed_index must reject an out-of-range coordinate'
	}
	rows := vtl.from_1d([0, 1])!
	columns := vtl.from_1d([0, 1, 0])!
	if _ := vtl.mixed_index[int](values, [vtl.array_index(rows), vtl.array_index(columns)]) {
		assert false, 'mixed_index must reject unbroadcastable coordinate tensors'
	}
	if _ := vtl.mixed_index[int](values, [vtl.full_index(), vtl.full_index(), vtl.full_index()])
	{
		assert false, 'mixed_index must reject more indices than tensor axes'
	}
}

fn test_mixed_index_expands_ellipsis_and_newaxis_as_basic_views() ! {
	values := vtl.from_array[int]([0, 1, 2, 3, 4, 5], [2, 3])!
	selected := vtl.mixed_index[int](values, [vtl.ellipsis_index(), vtl.integer_index(1)])!
	assert selected.shape == [2]
	assert selected.to_array() == [1, 4]

	mut expanded := vtl.mixed_index[int](values, [vtl.newaxis_index(), vtl.integer_index(1),
		vtl.ellipsis_index()])!
	assert expanded.shape == [1, 3]
	expanded.set([0, 0], 99)
	assert values.get([1, 0]) == 99

	trailing_axis := vtl.mixed_index[int](values, [vtl.ellipsis_index(), vtl.newaxis_index()])!
	assert trailing_axis.shape == [2, 3, 1]
	assert trailing_axis.to_array() == [0, 1, 2, 99, 4, 5]
}

fn test_mixed_index_newaxis_separates_advanced_axes_like_numpy() ! {
	values := vtl.from_array[int]([]int{len: 6, init: index}, [2, 3])!
	rows := vtl.from_array[int]([0, 1], [2, 1])!
	columns := vtl.from_array[int]([1, 2], [1, 2])!
	selected := vtl.mixed_index[int](values, [vtl.array_index(rows), vtl.newaxis_index(),
		vtl.array_index(columns)])!
	assert selected.shape == [2, 2, 1]
	assert selected.to_array() == [1, 2, 4, 5]
}

fn test_mixed_index_rejects_multiple_ellipses_and_extra_consuming_axes() ! {
	values := vtl.from_array[int]([0, 1, 2, 3], [2, 2])!
	if _ := vtl.mixed_index[int](values, [vtl.ellipsis_index(), vtl.ellipsis_index()]) {
		assert false, 'mixed_index must reject multiple ellipses'
	}
	if _ := vtl.mixed_index[int](values, [vtl.full_index(), vtl.full_index(), vtl.integer_index(0)]) {
		assert false, 'mixed_index must reject too many input-axis indices'
	}
}
