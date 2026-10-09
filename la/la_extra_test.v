module la

import vtl
import math

fn test_trace_identity() {
	a := vtl.from_2d([[1.0, 0.0], [0.0, 1.0]])!
	result := trace(a)!
	assert result.get_nth(0) == f64(2)
}

fn test_trace_general() {
	a := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
	result := trace(a)!
	assert result.get_nth(0) == f64(5)
}

fn test_norm_frobenius() {
	// ||[3,4]|| = 5
	a := vtl.from_2d([[3.0, 0.0], [0.0, 4.0]])!
	result := norm(a, 'fro')!
	diff := result.get_nth(0) - 5.0
	assert diff * diff < 1e-10
}

fn test_outer_product() {
	u := vtl.from_1d([1.0, 2.0])!
	v := vtl.from_1d([3.0, 4.0])!
	result := outer(u, v)!
	assert result.shape == [2, 2]
	// outer([1,2],[3,4]) = [[3,4],[6,8]]
	assert result.get_nth(0) == f64(3)
	assert result.get_nth(1) == f64(4)
	assert result.get_nth(2) == f64(6)
	assert result.get_nth(3) == f64(8)
}

fn test_cross_product() {
	u := vtl.from_1d([1.0, 0.0, 0.0])!
	v := vtl.from_1d([0.0, 1.0, 0.0])!
	result := cross(u, v)!
	assert result.shape == [3]
	assert result.get_nth(0) == f64(0)
	assert result.get_nth(1) == f64(0)
	assert result.get_nth(2) == f64(1)
}

fn test_qr_shape() {
	a := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
	q, r := qr(a)!
	assert q.shape[0] == 2
	assert r.shape == [2, 2]
}

fn test_lu_shape() {
	a := vtl.from_2d([[2.0, 1.0], [4.0, 3.0]])!
	l, u, _ := lu(a)!
	assert l.shape == [2, 2]
	assert u.shape == [2, 2]
}

fn test_matrix_rank_identity() {
	a := vtl.from_2d([[1.0, 0.0], [0.0, 1.0]])!
	r := matrix_rank(a, 1e-10)!
	assert r == 2
}

fn test_matrix_rank_singular() {
	a := vtl.from_2d([[1.0, 2.0], [2.0, 4.0]])!
	r := matrix_rank(a, 1e-10)!
	assert r == 1
}

fn test_matrix_rank_uses_dtype_aware_default_tolerance() {
	f64_matrix := vtl.from_2d([[1e-200, 0.0], [0.0, 2e-200]])!
	assert matrix_rank(f64_matrix, 0)! == 2
	f32_matrix := vtl.from_array([f32(1), 0, 0, 1e-8], [2, 2])!
	assert matrix_rank(f32_matrix, 0)! == 1
}

fn test_matrix_rank_batch_supports_stacked_rectangular_matrices() {
	input := vtl.from_array([1.0, 0, 0, 1, 1, 1, 1, 2, 2, 4, 3, 6], [2, 3, 2])!
	ranks := matrix_rank_batch(input, 1e-10)!
	assert ranks.shape == [2]
	assert ranks.to_array() == [2, 1]
}

fn test_solve_handles_row_pivoting_and_multiple_rhs() {
	a := vtl.from_2d([[0.0, 2.0], [1.0, 3.0]])!
	b := vtl.from_1d([4.0, 7.0])!
	assert solve(a, b)!.to_array() == [1.0, 2.0]
	rhs := vtl.from_2d([[4.0, 6.0], [7.0, 11.0]])!
	solution := solve(a, rhs)!
	assert solution.shape == [2, 2]
	assert solution.to_array() == [1.0, 2.0, 2.0, 3.0]
}

fn test_solve_broadcasts_matrix_and_vector_right_hand_sides() {
	a := vtl.from_array([1.0, 0, 0, 1, 2, 0, 0, 2], [2, 2, 2])!
	rhs_matrix := vtl.from_array([4.0, 6, 8, 10], [2, 2])!
	matrix_solution := solve(a, rhs_matrix)!
	assert matrix_solution.shape == [2, 2, 2]
	assert matrix_solution.to_array() == [4.0, 6.0, 8.0, 10.0, 2.0, 3.0, 4.0, 5.0]
	rhs_vector := vtl.from_1d([4.0, 8.0])!
	vector_solution := solve(a, rhs_vector)!
	assert vector_solution.shape == [2, 2]
	assert vector_solution.to_array() == [4.0, 8.0, 2.0, 4.0]
}

fn test_solve_rejects_singular_and_incompatible_batches() {
	singular := vtl.from_2d([[1.0, 2.0], [2.0, 4.0]])!
	rhs := vtl.from_1d([1.0, 2.0])!
	if _ := solve(singular, rhs) {
		assert false, 'solve must reject singular matrices'
	} else {
		assert true
	}
	a := vtl.ones[f64]([2, 2, 2])
	b := vtl.ones[f64]([3, 2])
	if _ := solve(a, b) {
		assert false, 'solve must reject incompatible batch dimensions'
	} else {
		assert true
	}
}

fn test_lstsq_vector_rhs_preserves_numpy_shape_and_squared_residuals() {
	a := vtl.from_2d([[1.0, 0], [0, 1], [1, 1]])!
	b := vtl.from_1d([1.0, 2, 4])!
	x, residuals, rank, singular_values := lstsq(a, b)!
	assert x.shape == [2]
	assert math.abs(x.get([0]) - 4.0 / 3.0) < 1e-10
	assert math.abs(x.get([1]) - 7.0 / 3.0) < 1e-10
	assert residuals.shape == [1]
	assert math.abs(residuals.get([0]) - 1.0 / 3.0) < 1e-10
	assert rank == 2
	assert singular_values.shape == [2]
}

fn test_lstsq_rank_deficient_system_has_empty_residuals_and_validates_rows() {
	a := vtl.from_2d([[1.0, 1], [2, 2], [3, 3]])!
	b := vtl.from_1d([2.0, 4, 6])!
	x, residuals, rank, _ := lstsq(a, b)!
	assert x.shape == [2]
	assert x.to_array() == [1.0, 1.0]
	assert residuals.shape == [0]
	assert rank == 1
	wrong_rows := vtl.from_1d([1.0, 2])!
	if _, _, _, _ := lstsq(a, wrong_rows) {
		assert false, 'lstsq must reject mismatched row counts'
	} else {
		assert true
	}
}

fn test_tensordot_contracts_trailing_a_with_leading_b_axes() {
	a := vtl.from_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])!
	b := vtl.from_2d([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])!
	result := tensordot(a, b, 1)!
	assert result.shape == [2, 2]
	assert result.to_array() == [22.0, 28.0, 49.0, 64.0]
}

fn test_tensordot_explicit_nontrailing_axes_and_strided_view() {
	base := vtl.from_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])!
	a := base.transpose([1, 0])!
	b := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
	result := tensordot_axes(a, b, [-1], [0])!
	assert result.shape == [3, 2]
	assert result.to_array() == [13.0, 18.0, 17.0, 24.0, 21.0, 30.0]
}

fn test_tensordot_zero_axes_is_outer_product() {
	a := vtl.from_1d([1.0, 2.0])!
	b := vtl.from_1d([3.0, 4.0, 5.0])!
	result := tensordot(a, b, 0)!
	assert result.shape == [2, 3]
	assert result.to_array() == [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
}

fn test_tensordot_full_contraction_returns_scalar() {
	a := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
	b := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
	result := tensordot_axes(a, b, [0, 1], [1, 0])!
	assert result.shape.len == 0
	assert result.get([]) == 29.0
}

fn test_tensordot_preserves_integer_dtype_and_empty_contractions() {
	a := vtl.from_1d[int]([1, 2])!
	b := vtl.from_1d[int]([3, 4])!
	integer_result := tensordot(a, b, 1)!
	assert integer_result.get([]) == 11

	large_a := vtl.from_1d[i64]([9007199254740993, 0])!
	large_b := vtl.from_1d[i64]([1, 1])!
	assert tensordot(large_a, large_b, 1)!.get([]) == i64(9007199254740993)

	large_matrix := vtl.from_2d[i64]([[9007199254740993, 0]])!
	column := vtl.from_2d[i64]([[1], [1]])!
	assert tensordot(large_matrix, column, 1)!.get([0, 0]) == i64(9007199254740993)

	empty_a := vtl.from_array[f64]([], [2, 0])!
	empty_b := vtl.from_array[f64]([], [0, 3])!
	empty_result := tensordot(empty_a, empty_b, 1)!
	assert empty_result.shape == [2, 3]
	assert empty_result.to_array() == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

fn test_tensordot_rejects_invalid_axis_lists() {
	a := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
	other := vtl.from_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])!
	if _ := tensordot_axes(a, a, [0], []) {
		assert false
	}
	if _ := tensordot_axes(a, a, [0, 0], [0, 1]) {
		assert false
	}
	if _ := tensordot_axes(a, a, [2], [0]) {
		assert false
	}
	if _ := tensordot_axes(a, other, [0], [1]) {
		assert false
	}
	if _ := tensordot(a, a, 3) {
		assert false
	}
}

fn test_tensordot_multiple_reordered_axes() {
	a := vtl.ones[f64]([2, 3, 4])
	b := vtl.ones[f64]([3, 2, 5])
	result := tensordot_axes(a, b, [1, 0], [0, 1])!
	assert result.shape == [4, 5]
	for value in result.to_array() {
		assert value == 6.0
	}
}
