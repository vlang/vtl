module la

import math
import math.complex as vcomplex
import vtl

fn test_eig_returns_complex_conjugate_pairs_and_right_eigenvectors() ! {
	input := vtl.from_2d([[0.0, -1.0], [1.0, 0.0]])!
	values, vectors := eig(input)!
	assert values.shape == [2]
	assert vectors.shape == [2, 2]
	assert math.abs(values.get_nth(0).re) < 1e-12
	assert math.abs(values.get_nth(0).im - 1.0) < 1e-12
	assert math.abs(values.get_nth(1).re) < 1e-12
	assert math.abs(values.get_nth(1).im + 1.0) < 1e-12
	for component in 0 .. 2 {
		lambda := values.get_nth(component)
		for row in 0 .. 2 {
			mut product := vcomplex.Complex{}
			for column in 0 .. 2 {
				product += vcomplex.Complex{
					re: input.get([row, column]) * vectors.get([column, component]).re
					im: input.get([row, column]) * vectors.get([column, component]).im
				}
			}
			vector_value := vectors.get([row, component])
			assert math.abs(product.re - (lambda.re * vector_value.re - lambda.im * vector_value.im)) < 1e-12
			assert math.abs(product.im - (lambda.re * vector_value.im + lambda.im * vector_value.re)) < 1e-12
		}
	}
	assert eigvals(input)!.array_equal(values)
}

fn test_eig_supports_batched_real_eigenvalues_and_empty_matrices() ! {
	input := vtl.from_array[f64]([2.0, 0, 0, 3, 0, -1, 1, 0], [2, 2, 2])!
	values, vectors := eig(input)!
	assert values.shape == [2, 2]
	assert vectors.shape == [2, 2, 2]
	assert math.abs(values.get_nth(0).re + values.get_nth(1).re - 5.0) < 1e-12
	assert math.abs(values.get_nth(0).re * values.get_nth(1).re - 6.0) < 1e-12
	assert values.get_nth(2).im == 1.0
	assert values.get_nth(3).im == -1.0
	empty := vtl.empty[f64]([0, 0])
	empty_values, empty_vectors := eig(empty)!
	assert empty_values.shape == [0]
	assert empty_vectors.shape == [0, 0]
}

fn test_eig_rejects_non_square_and_non_finite_inputs() ! {
	if _, _ := eig(vtl.zeros[f64]([2, 3])) {
		assert false, 'eig must reject non-square matrices'
	}
	if _, _ := eig(vtl.from_2d([[math.inf(1)]])!) {
		assert false, 'eig must reject non-finite values'
	}
}
