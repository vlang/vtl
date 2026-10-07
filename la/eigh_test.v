module la

import math
import vtl

fn test_eigh_returns_numpy_ordered_eigenvalues_and_vectors() {
	input := vtl.from_2d([[1.0, 2.0], [2.0, 1.0]])!
	values, vectors := eigh(input, EighOptions{})!
	assert values.to_array() == [-1.0, 3.0]
	assert vectors.shape == [2, 2]
	for column in 0 .. 2 {
		lambda := values.get_nth(column)
		for row in 0 .. 2 {
			mut product := 0.0
			for inner in 0 .. 2 {
				product += input.get([row, inner]) * vectors.get([inner, column])
			}
			assert math.abs(product - lambda * vectors.get([row, column])) < 1e-12
		}
	}
}

fn test_eigh_supports_batched_matrices_and_upper_triangle() {
	input := vtl.from_array([1.0, 2, 99, 3, 2, 1, 99, 4], [2, 2, 2])!
	values, vectors := eigh(input, EighOptions{
		uplo: 'U'
	})!
	assert values.shape == [2, 2]
	assert vectors.shape == [2, 2, 2]
	assert math.abs(values.get_nth(0) - (4.0 - math.sqrt(20.0)) / 2.0) < 1e-12
	assert math.abs(values.get_nth(1) - (4.0 + math.sqrt(20.0)) / 2.0) < 1e-12
	assert math.abs(values.get_nth(2) - (6.0 - math.sqrt(8.0)) / 2.0) < 1e-12
	assert math.abs(values.get_nth(3) - (6.0 + math.sqrt(8.0)) / 2.0) < 1e-12
}

fn test_eigh_handles_empty_matrix() {
	input := vtl.empty[f64]([0, 0])
	values, vectors := eigh(input, EighOptions{})!
	assert values.shape == [0]
	assert vectors.shape == [0, 0]
}

fn test_eigh_scales_large_magnitudes_and_eigvalsh_skips_vector_results() {
	input := vtl.from_2d([[1e200, 0.0], [0.0, -2e200]])!
	values := eigvalsh(input, EighOptions{})!
	assert math.abs(values.get_nth(0) / 1e200 + 2.0) < 1e-12
	assert math.abs(values.get_nth(1) / 1e200 - 1.0) < 1e-12
}

fn test_eigh_rejects_non_square_and_invalid_triangle() {
	nonsquare := vtl.zeros[f64]([2, 3])
	if _, _ := eigh(nonsquare, EighOptions{}) {
		assert false, 'eigh must reject non-square matrices'
	} else {
		assert true
	}
	square := vtl.zeros[f64]([2, 2])
	if _, _ := eigh(square, EighOptions{
		uplo: 'X'
	}) {
		assert false, 'eigh must reject invalid UPLO'
	} else {
		assert true
	}
	nonfinite := vtl.from_2d([[f64(math.inf(1))]])!
	if _, _ := eigh(nonfinite, EighOptions{}) {
		assert false, 'eigh must reject non-finite input'
	} else {
		assert true
	}
}
