module core

import vtl

fn test_diff_computes_successive_vector_differences() ! {
	values := vtl.from_1d([1, 2, 4, 7])!
	first := vtl.diff[int](values, 1, 0)!
	assert first.shape == [3]
	assert first.to_array() == [1, 2, 3]

	second := vtl.diff[int](values, 2, -1)!
	assert second.shape == [2]
	assert second.to_array() == [1, 1]

	unchanged := vtl.diff[int](values, 0, 0)!
	assert unchanged.shape == values.shape
	assert unchanged.to_array() == values.to_array()
}

fn test_diff_supports_each_matrix_axis_and_empty_outputs() ! {
	values := vtl.from_2d([[1, 3, 6], [2, 5, 9]])!
	rows := vtl.diff[int](values, 1, 0)!
	assert rows.shape == [1, 3]
	assert rows.to_array() == [1, 2, 3]
	columns := vtl.diff[int](values, 1, -1)!
	assert columns.shape == [2, 2]
	assert columns.to_array() == [2, 3, 3, 4]

	empty := vtl.diff[int](vtl.from_1d([]int{})!, 1, 0)!
	assert empty.shape == [0]
}

fn test_diff_rejects_invalid_order_axis_and_scalar_inputs() ! {
	values := vtl.from_1d([1, 2])!
	if _ := vtl.diff[int](values, -1, 0) {
		assert false, 'diff must reject a negative order'
	}
	if _ := vtl.diff[int](values, 1, 1) {
		assert false, 'diff must reject an out-of-range axis'
	}
	if _ := vtl.diff[int](vtl.from_array([1], [])!, 1, 0) {
		assert false, 'diff must reject scalar inputs'
	}
}
