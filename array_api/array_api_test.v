module array_api

import vtl

fn test_capability_matrix_is_explicit_and_queryable() {
	matrix := capability_matrix()
	assert matrix.len == 6
	assert supports('zeros')
	assert supports('reshape')
	assert !supports('asarray')
	assert !supports('matmul')
	for capability in matrix {
		assert capability.supported
	}
}

fn test_creation_operations_match_vtl_values_and_shapes() {
	assert zeros[f64]([2, 2]).to_array() == [0.0, 0.0, 0.0, 0.0]
	assert ones[int]([2]).to_array() == [1, 1]
	assert full[f64]([2], 3.5).to_array() == [3.5, 3.5]
	assert empty[int]([0, 3]).shape == [0, 3]
	assert arange[int](1.0, 5.0, 2.0)!.to_array() == [1, 3]
}

fn test_reshape_preserves_values_and_rejects_incompatible_shapes() {
	input := vtl.from_array([1, 2, 3, 4], [2, 2])!
	output := reshape(input, [4])!
	assert output.shape == [4]
	assert output.to_array() == [1, 2, 3, 4]
	if _ := reshape(input, [3]) {
		assert false, 'reshape must reject incompatible element counts'
	}
}
