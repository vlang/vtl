module la

import math
import vtl

fn test_matrix_power_handles_zero_and_positive_exponents() {
	input := vtl.from_2d([[0.0, 1.0], [-1.0, 0.0]])!
	square := matrix_power(input, 2)!
	assert square.to_array() == [-1.0, 0.0, 0.0, -1.0]
	identity := matrix_power(input, 0)!
	assert identity.to_array() == [1.0, 0.0, 0.0, 1.0]
}

fn test_matrix_power_handles_negative_exponents() {
	input := vtl.from_2d([[0.0, 1.0], [-1.0, 0.0]])!
	inverse_power := matrix_power(input, -3)!
	assert inverse_power.to_array() == [0.0, 1.0, -1.0, 0.0]
}

fn test_matrix_power_supports_batched_matrices() {
	input := vtl.from_array([1, 1, 0, 1, 2, 0, 0, 3], [2, 2, 2])!
	result := matrix_power(input, 2)!
	assert result.shape == [2, 2, 2]
	assert result.to_array() == [1.0, 2.0, 0.0, 1.0, 4.0, 0.0, 0.0, 9.0]
}

fn test_matrix_power_rejects_invalid_shapes_and_singular_inverse() {
	vector := vtl.from_1d([1.0, 2.0])!
	if _ := matrix_power(vector, 2) {
		assert false, 'matrix_power must reject vectors'
	} else {
		assert true
	}
	nonsquare := vtl.zeros[f64]([2, 3])
	if _ := matrix_power(nonsquare, 2) {
		assert false, 'matrix_power must reject non-square matrices'
	} else {
		assert true
	}
	singular := vtl.from_2d([[1.0, 2.0], [2.0, 4.0]])!
	if _ := matrix_power(singular, -1) {
		assert false, 'matrix_power must reject singular inverses'
	} else {
		assert true
	}
}

fn test_matrix_power_negative_inverse_is_accurate() {
	input := vtl.from_2d([[4.0, 7.0], [2.0, 6.0]])!
	result := matrix_power(input, -1)!
	assert math.abs(result.get([0, 0]) - 0.6) < 1e-12
	assert math.abs(result.get([0, 1]) + 0.7) < 1e-12
	assert math.abs(result.get([1, 0]) + 0.2) < 1e-12
	assert math.abs(result.get([1, 1]) - 0.4) < 1e-12
}
