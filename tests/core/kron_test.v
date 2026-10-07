module core

import vtl

fn test_kron_vectors_and_matrices_match_numpy_order() ! {
	a := vtl.from_1d([1, 2])!
	b := vtl.from_1d([3, 4, 5])!
	vector_result := vtl.kron(a, b)!
	assert vector_result.shape == [6]
	assert vector_result.to_array() == [3, 4, 5, 6, 8, 10]

	left := vtl.from_2d([[1, 2], [3, 4]])!
	right := vtl.from_2d([[0, 5], [6, 7]])!
	matrix_result := vtl.kron(left, right)!
	assert matrix_result.shape == [4, 4]
	assert matrix_result.to_array() == [0, 5, 0, 10, 6, 7, 12, 14, 0, 15, 0, 20, 18, 21, 24, 28]
}

fn test_kron_promotes_rank_and_supports_scalars() ! {
	matrix := vtl.from_2d([[1, 2], [3, 4]])!
	vector := vtl.from_1d([5, 6])!
	result := vtl.kron(matrix, vector)!
	assert result.shape == [2, 4]
	assert result.to_array() == [5, 6, 10, 12, 15, 18, 20, 24]

	scalar := vtl.from_array([3], [])!
	scaled := vtl.kron(scalar, vector)!
	assert scaled.shape == [2]
	assert scaled.to_array() == [15, 18]
	scalar_product := vtl.kron(scalar, scalar)!
	assert scalar_product.shape == []
	assert scalar_product.to_array() == [9]
}

fn test_kron_handles_strided_and_empty_inputs() ! {
	base := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	transposed := base.transpose([1, 0])!
	identity := vtl.from_2d([[1, 0], [0, 1]])!
	result := vtl.kron(transposed, identity)!
	assert result.shape == [6, 4]
	assert result.to_array() == [1, 0, 4, 0, 0, 1, 0, 4, 2, 0, 5, 0, 0, 2, 0, 5, 3, 0, 6, 0, 0,
		3, 0, 6]

	empty := vtl.from_array([]int{}, [0, 2])!
	empty_result := vtl.kron(empty, identity)!
	assert empty_result.shape == [0, 4]
	assert empty_result.size == 0
	assert empty_result.to_array() == []int{}
}

fn test_kron_supports_boolean_tensors() ! {
	a := vtl.from_1d([true, false])!
	b := vtl.from_1d([false, true])!
	result := vtl.kron(a, b)!
	assert result.to_array() == [false, true, false, false]
}
