module la

import vtl

fn test_diag_builds_vector_matrices_with_offsets() ! {
	values := vtl.from_1d([1, 2, 3])!
	main := diag(values, 0)!
	assert main.shape == [3, 3]
	assert main.to_array() == [1, 0, 0, 0, 2, 0, 0, 0, 3]
	upper := diag(values, 1)!
	assert upper.shape == [4, 4]
	assert upper.to_array() == [0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0]
	lower := diag(values, -1)!
	assert lower.to_array() == [0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0]
}

fn test_diag_extracts_rectangular_matrix_diagonals() ! {
	values := vtl.from_array([1, 2, 3, 4, 5, 6], [2, 3])!
	assert diag(values, 0)!.to_array() == [1, 5]
	assert diag(values, 1)!.to_array() == [2, 6]
	assert diag(values, -1)!.to_array() == [4]
	assert diag(values, 3)!.to_array() == []
}

fn test_diag_rejects_other_ranks() {
	values := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])!
	if _ := diag(values, 0) {
		assert false
	} else {
		assert err.msg().contains('one-dimensional')
	}
}
