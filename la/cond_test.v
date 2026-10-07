module la

import math
import vtl

fn test_cond_spectral_orders_match_numpy() {
	input := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
	condition := cond(input, CondOptions{})!
	assert math.abs(condition.get_nth(0) - 14.933034373659265) < 1e-9
	inverse_condition := cond(input, CondOptions{
		ord: '-2'
	})!
	assert math.abs(inverse_condition.get_nth(0) - 1.0 / 14.933034373659265) < 1e-9
}

fn test_cond_matrix_norm_orders_and_batches() {
	input := vtl.from_array([1, 2, 3, 4, 2, 0, 0, 2], [2, 2, 2])!
	condition := cond(input, CondOptions{
		ord: '1'
	})!
	assert condition.shape == [2]
	assert math.abs(condition.get_nth(0) - 21.0) < 1e-10
	assert math.abs(condition.get_nth(1) - 1.0) < 1e-10
	frobenius := cond(input, CondOptions{
		ord: 'fro'
	})!
	assert math.abs(frobenius.get_nth(1) - 2.0) < 1e-10
}

fn test_cond_handles_singular_spectral_matrix() {
	input := vtl.from_2d([[1.0, 2.0], [2.0, 4.0]])!
	assert math.is_inf(cond(input, CondOptions{})!.get_nth(0), 1)
	assert cond(input, CondOptions{
		ord: '-2'
	})!.get_nth(0) == 0
}

fn test_cond_rejects_invalid_shapes_orders_and_singular_inverse_orders() {
	vector := vtl.from_1d([1.0, 2.0])!
	if _ := cond(vector, CondOptions{}) {
		assert false, 'cond must reject vectors'
	} else {
		assert true
	}
	nonsquare := vtl.zeros[f64]([2, 3])
	if _ := cond(nonsquare, CondOptions{}) {
		assert false, 'cond must reject non-square matrices'
	} else {
		assert true
	}
	input := vtl.from_2d([[1.0, 2.0], [2.0, 4.0]])!
	if _ := cond(input, CondOptions{
		ord: 'invalid'
	}) {
		assert false, 'cond must reject unsupported orders'
	} else {
		assert true
	}
	if _ := cond(input, CondOptions{
		ord: '1'
	}) {
		assert false, 'non-spectral condition orders require an invertible matrix'
	} else {
		assert true
	}
}
