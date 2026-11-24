module array_api

import math
import vtl

fn test_capability_matrix_is_explicit_and_queryable() {
	matrix := capability_matrix()
	assert matrix.len == 16
	assert supports('zeros')
	assert supports('reshape')
	assert supports('where')
	assert supports('matmul')
	assert supports('clip')
	assert supports('sum')
	assert supports('mean')
	assert supports('isclose')
	assert supports('allclose')
	assert supports('isfinite')
	assert supports('isinf')
	assert supports('isnan')
	assert !supports('asarray')

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

fn test_matmul_supports_vectors_and_matrices() {
	vector := vtl.from_1d[int]([1, 2, 3])!
	matrix := vtl.from_2d[int]([[1, 2], [3, 4], [5, 6]])!

	assert matmul(vector, matrix)!.shape == [2]
	assert matmul(vector, matrix)!.to_array() == [22, 28]
	got := matmul(matrix, matrix.t()!)!
	assert got.to_array() == [5, 11, 17, 11, 25, 39, 17, 39, 61]
}

fn test_matmul_rejects_incompatible_shapes() {
	left := vtl.from_2d[int]([[1, 2]])!
	right := vtl.from_2d[int]([[1, 2]])!

	if _ := matmul(left, right) {
		assert false, 'matmul must reject incompatible inner dimensions'
	}
}

fn test_clip_limits_values_and_preserves_shape() {
	input := vtl.from_2d[f64]([[-2.0, 0.5, 4.0]])!
	result := clip(input, 0.0, 1.0)!

	assert result.shape == [1, 3]
	assert result.to_array() == [0.0, 0.5, 1.0]
}

fn test_clip_rejects_reversed_bounds() {
	input := vtl.from_1d[int]([1, 2, 3])!
	if _ := clip(input, 3, 1) {
		assert false, 'clip must reject reversed bounds'
	}
}

fn test_sum_and_mean_reduce_all_elements() {
	input := vtl.from_2d[f64]([[1.0, 2.0], [3.0, 4.0]])!

	assert sum(input) == 10.0
	assert mean(input) == 2.5
	assert sum(vtl.from_1d[int]([1, 2, 3])!) == 6
	assert mean(vtl.from_1d[int]([1, 2, 3])!) == 2.0
}

fn test_reductions_handle_empty_arrays() {
	empty := vtl.empty[f64]([0, 2])

	assert math.is_nan(mean(empty))
	assert sum(empty) == 0.0
}

fn test_isclose_and_allclose_broadcast_values() {
	left := vtl.from_2d[f64]([[1.0], [2.0]])!
	right := vtl.from_1d[f64]([1.0, 2.00001])!

	result := isclose(left, right, 1e-5, 1e-8, false)!
	assert result.shape == [2, 2]
	assert result.to_array() == [true, false, false, true]
	assert allclose(left, left, 1e-5, 1e-8, false)!
}

fn test_isclose_handles_nan_and_invalid_tolerances() {
	nan := vtl.from_1d[f64]([math.nan()])!
	assert !isclose(nan, nan, 1e-5, 1e-8, false)!.get_nth(0)
	assert isclose(nan, nan, 1e-5, 1e-8, true)!.get_nth(0)
	if _ := allclose(nan, nan, -1.0, 1e-8, false) {
		assert false, 'negative tolerances must fail'
	}
}

fn test_finiteness_ufuncs_classify_special_values() {
	input := vtl.from_1d[f64]([1.0, math.inf(1), math.nan()])!

	assert isfinite(input).to_array() == [true, false, false]
	assert isinf(input).to_array() == [false, true, false]
	assert isnan(input).to_array() == [false, false, true]
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
