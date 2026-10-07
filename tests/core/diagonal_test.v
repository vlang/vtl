module core

import vtl

fn test_diagonal_extracts_default_and_offset_nd_views() ! {
	input := vtl.from_array([]int{len: 24, init: index}, [2, 3, 4])!
	main := vtl.diagonal(input)!
	assert main.shape == [4, 2]
	assert main.to_array() == [0, 16, 1, 17, 2, 18, 3, 19]
	upper := vtl.diagonal(input, axis1: 0, axis2: 1, offset: 1)!
	assert upper.shape == [4, 2]
	assert upper.to_array() == [4, 20, 5, 21, 6, 22, 7, 23]
	lower := vtl.diagonal(input, axis1: 0, axis2: 1, offset: -1)!
	assert lower.shape == [4, 1]
	assert lower.to_array() == [12, 13, 14, 15]
}

fn test_diagonal_supports_arbitrary_and_negative_axes() ! {
	input := vtl.from_array([]int{len: 24, init: index}, [2, 3, 4])!
	selected := vtl.diagonal(input, axis1: 0, axis2: -1)!
	assert selected.shape == [3, 2]
	assert selected.to_array() == [0, 13, 4, 17, 8, 21]
}

fn test_diagonal_view_writes_through_to_input() ! {
	mut input := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9], [3, 3])!
	mut diagonal := vtl.diagonal(input)!
	diagonal.set([1], 50)
	assert input.get([1, 1]) == 50
	mut transposed := input.transpose([1, 0])!
	mut offset_diagonal := vtl.diagonal(transposed, offset: 1)!
	offset_diagonal.set([0], 80)
	assert input.get([1, 0]) == 80
}

fn test_diagonal_out_of_range_offset_is_empty() ! {
	input := vtl.from_array([1, 2, 3, 4], [2, 2])!
	assert vtl.diagonal(input, offset: 3)!.shape == [0]
	assert vtl.diagonal(input, offset: -2)!.shape == [0]
	assert vtl.diagonal(input, offset: min_int)!.shape == [0]
	assert vtl.diagonal(input, offset: max_int)!.shape == [0]
}

fn test_diagonal_rejects_invalid_rank_and_axes() {
	vector := vtl.from_1d([1, 2, 3])!
	if _ := vtl.diagonal(vector) {
		assert false
	} else {
		assert err.msg().contains('at least two dimensions')
	}
	matrix := vtl.from_array([1, 2, 3, 4], [2, 2])!
	if _ := vtl.diagonal(matrix, axis1: 0, axis2: 0) {
		assert false
	} else {
		assert err.msg().contains('different')
	}
	if _ := vtl.diagonal(matrix, axis1: 0, axis2: 2) {
		assert false
	} else {
		assert err.msg().contains('out of range')
	}
}

fn test_diagonal_handles_empty_dimensions_and_strided_input() ! {
	empty := vtl.from_array([]int{}, [2, 0, 3])!
	empty_diagonal := vtl.diagonal(empty, axis1: 0, axis2: 2)!
	assert empty_diagonal.shape == [0, 2]
	assert empty_diagonal.size == 0
	assert empty_diagonal.to_array() == []int{}

	input := vtl.from_array([0, 1, 2, 3, 4, 5], [2, 3])!
	mut transposed := input.transpose([1, 0])!
	mut diagonal := vtl.diagonal(transposed)!
	assert diagonal.shape == [2]
	assert diagonal.to_array() == [0, 4]
	diagonal.set([1], 40)
	assert input.get([1, 1]) == 40
}
