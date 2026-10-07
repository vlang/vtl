module la

import vtl

fn test_trace_axes_default_and_batched_offsets() ! {
	input := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], [2, 2, 3])!
	assert trace_axes(input)!.to_array() == [11.0, 13.0, 15.0]
	assert trace_axes(input, axis1: 1, axis2: 2)!.to_array() == [6.0, 18.0]
	assert trace_axes(input, axis1: -2, axis2: -1, offset: 1)!.to_array() == [8.0, 20.0]
	assert trace_axes(input, axis1: 1, axis2: 2, offset: -1)!.to_array() == [4.0, 10.0]
}

fn test_trace_axes_returns_one_value_for_matrix() ! {
	input := vtl.from_2d([[3, 0], [0, 4]])!
	result := trace_axes(input)!
	assert result.shape == [1]
	assert result.get_nth(0) == 7.0
}

fn test_trace_axes_rejects_invalid_rank_and_axes() {
	vector := vtl.from_1d([1, 2, 3])!
	if _ := trace_axes(vector) {
		assert false, 'trace_axes must reject rank-one input'
	}
	matrix := vtl.from_2d([[1, 2], [3, 4]])!
	if _ := trace_axes(matrix, axis1: 0, axis2: 0) {
		assert false, 'trace_axes must reject repeated axes'
	}
	if _ := trace_axes(matrix, axis1: 0, axis2: 2) {
		assert false, 'trace_axes must reject out-of-bounds axes'
	}
}
