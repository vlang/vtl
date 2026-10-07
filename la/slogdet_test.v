module la

import math
import vtl

fn test_slogdet_matches_numpy_and_handles_pivot_swaps() {
	input := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
	sign, logabsdet := slogdet(input)!
	assert sign.to_array() == [-1.0]
	assert math.abs(logabsdet.get_nth(0) - math.log(2.0)) < 1e-12
}

fn test_slogdet_supports_batches_and_singular_matrices() {
	input := vtl.from_array([1, 2, 2, 4, 0, 0, 0, 0], [2, 2, 2])!
	sign, logabsdet := slogdet(input)!
	assert sign.shape == [2]
	assert sign.to_array() == [0.0, 0.0]
	assert math.is_inf(logabsdet.get_nth(0), -1)
	assert math.is_inf(logabsdet.get_nth(1), -1)
}

fn test_slogdet_avoids_determinant_overflow() {
	input := vtl.from_2d([[1e200, 0.0], [0.0, 1e200]])!
	sign, logabsdet := slogdet(input)!
	assert sign.to_array() == [1.0]
	assert math.abs(logabsdet.get_nth(0) - math.log(1e200) * 2) < 1e-12
}

fn test_slogdet_handles_empty_square_matrix() {
	input := vtl.empty[f64]([0, 0])
	sign, logabsdet := slogdet(input)!
	assert sign.to_array() == [1.0]
	assert logabsdet.to_array() == [0.0]
}

fn test_slogdet_rejects_non_square_and_non_finite_inputs() {
	nonsquare := vtl.zeros[f64]([2, 3])
	if _, _ := slogdet(nonsquare) {
		assert false, 'slogdet must reject non-square matrices'
	} else {
		assert true
	}
	nonfinite := vtl.from_2d([[f64(math.inf(1))]])!
	if _, _ := slogdet(nonfinite) {
		assert false, 'slogdet must reject non-finite input'
	} else {
		assert true
	}
}
