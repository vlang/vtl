module core

import vtl

fn test_unique_axis_rows_and_metadata() ! {
	tensor := vtl.from_2d([[2, 1], [1, 4], [2, 1]])!
	result := vtl.unique_axis_result(tensor, 0)!
	assert result.values.shape == [2, 2]
	assert result.values.to_array() == [1, 4, 2, 1]
	assert result.counts.to_array() == [1, 2]
	assert result.first_indices.to_array() == [1, 0]
	assert result.inverse.to_array() == [1, 0, 1]
}

fn test_unique_axis_columns_and_negative_axis() ! {
	tensor := vtl.from_2d([[1, 1], [2, 2]])!
	result := vtl.unique_axis_result(tensor, -1)!
	assert result.values.shape == [2, 1]
	assert result.values.to_array() == [1, 2]
	assert result.counts.to_array() == [2]
	assert result.first_indices.to_array() == [0]
	assert result.inverse.to_array() == [0, 0]
}

fn test_unique_axis_flattens_vectors_and_handles_empty_axes() ! {
	vector := vtl.from_1d([3, 1, 3, 2])!
	assert vtl.unique_axis(vector, 0)!.to_array() == [1, 2, 3]
	empty := vtl.from_array[int]([]int{}, [0, 2])!
	result := vtl.unique_axis_result(empty, 0)!
	assert result.values.shape == [0, 2]
	assert result.counts.to_array() == []int{}
	assert result.first_indices.to_array() == []int{}
	assert result.inverse.to_array() == []int{}
}

fn test_unique_axis_rejects_invalid_axes() ! {
	tensor := vtl.from_2d([[1, 2], [3, 4]])!
	if _ := vtl.unique_axis(tensor, 2) {
		assert false, 'out-of-range axis must return an error'
	} else {
		assert true
	}
	scalar := vtl.from_array[int]([1], []int{})!
	if _ := vtl.unique_axis(scalar, 0) {
		assert false, 'scalar input must return an error'
	} else {
		assert true
	}
}
