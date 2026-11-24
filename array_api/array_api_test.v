module array_api

import vtl

fn test_capability_matrix_is_explicit_and_queryable() {
	matrix := capability_matrix()
	assert matrix.len == 7
	assert supports('zeros')
	assert supports('reshape')
	assert supports('where')
	assert !supports('asarray')
	assert !supports('matmul')
	for capability in matrix {
		assert capability.supported
	}
}

fn test_where_broadcasts_condition_and_operands() {
	condition := vtl.from_2d[bool]([[true], [false]])!
	x := vtl.from_1d[int]([1, 2, 3])!
	y := vtl.from_2d[int]([[10]])!

	result := where(condition, x, y)!
	assert result.shape == [2, 3]
	assert result.to_array() == [1, 2, 3, 10, 10, 10]
}

fn test_where_rejects_incompatible_shapes() {
	condition := vtl.from_1d[bool]([true, false])!
	x := vtl.from_1d[int]([1, 2, 3])!
	y := vtl.from_1d[int]([10, 20, 30])!

	if _ := where(condition, x, y) {
		assert false, 'where must reject incompatible broadcast shapes'
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
