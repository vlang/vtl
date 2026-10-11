module la

import math
import math.complex as vcomplex
import vtl

fn test_lstsq_complex_recovers_overdetermined_vector_solution() ! {
	a := vtl.from_2d[vcomplex.Complex]([
		[vcomplex.Complex{ re: 1, im: 1 }, vcomplex.Complex{ re: 2 }],
		[vcomplex.Complex{ re: 2 }, vcomplex.Complex{ re: 1, im: -1 }],
		[vcomplex.Complex{ re: 1 }, vcomplex.Complex{ re: -1, im: 1 }],
	])!
	b := vtl.from_1d[vcomplex.Complex]([
		vcomplex.Complex{ re: -3, im: 4 },
		vcomplex.Complex{ re: 1.5, im: 5.5 },
		vcomplex.Complex{ re: 1.5, im: 0.5 },
	])!

	x, residuals, rank, singular_values := lstsq_complex(a, b)!
	assert x.shape == [2]
	assert math.abs(x.get_nth(0).re - 1) < 1e-10
	assert math.abs(x.get_nth(0).im - 2) < 1e-10
	assert math.abs(x.get_nth(1).re + 1) < 1e-10
	assert math.abs(x.get_nth(1).im - 0.5) < 1e-10
	assert residuals.size == 1
	assert residuals.get_nth(0) < 1e-20
	assert rank == 2
	assert singular_values.shape == [2]
	assert math.abs(singular_values.get_nth(0) - 3.431035098211135) < 1e-8
	assert math.abs(singular_values.get_nth(1) - 1.796663046910603) < 1e-8
}

fn test_lstsq_complex_preserves_matrix_rhs_shape_and_rank_deficiency() ! {
	a := vtl.from_2d[vcomplex.Complex]([
		[vcomplex.Complex{ re: 1, im: 1 }, vcomplex.Complex{ re: 2, im: 2 }],
		[vcomplex.Complex{ re: 2 }, vcomplex.Complex{ re: 4 }],
	])!
	b := vtl.from_2d[vcomplex.Complex]([
		[vcomplex.Complex{ re: 1, im: 1 }, vcomplex.Complex{ re: -1, im: 1 }],
		[vcomplex.Complex{ re: 2 }, vcomplex.Complex{ re: 0, im: 2 }],
	])!

	x, residuals, rank, singular_values := lstsq_complex(a, b)!
	assert x.shape == [2, 2]
	assert residuals.size == 0
	assert rank == 1
	assert singular_values.shape == [2]
	assert math.abs(x.get([0, 0]).re - 0.2) < 1e-10
	assert math.abs(x.get([1, 0]).re - 0.4) < 1e-10
	assert math.abs(x.get([0, 1]).im - 0.2) < 1e-10
	assert math.abs(x.get([1, 1]).im - 0.4) < 1e-10
}

fn test_lstsq_complex_rejects_mismatched_rhs_rows() {
	a := vtl.from_2d[vcomplex.Complex]([[vcomplex.Complex{ re: 1 }]])!
	b := vtl.from_1d[vcomplex.Complex]([])!
	if _, _, _, _ := lstsq_complex(a, b) {
		assert false, 'lstsq_complex must reject a right-hand side with mismatched rows'
	}
}

fn test_lstsq_complex_matches_numpy_default_rank_cutoff() ! {
	a := vtl.from_2d[vcomplex.Complex]([
		[vcomplex.Complex{ re: 1 }, vcomplex.Complex{}],
		[vcomplex.Complex{}, vcomplex.Complex{ re: 6.661338147750939e-16 }],
	])!
	b := vtl.from_1d[vcomplex.Complex]([vcomplex.Complex{ re: 1 }, vcomplex.Complex{}])!
	_, _, rank, _ := lstsq_complex(a, b)!
	assert rank == 2
}

fn test_lstsq_complex_handles_empty_dimensions_and_rhs() ! {
	empty_rows_a := vtl.empty[vcomplex.Complex]([0, 2])
	empty_rows_b := vtl.empty[vcomplex.Complex]([0])
	empty_rows_x, empty_rows_residuals, empty_rows_rank, empty_rows_singular_values := lstsq_complex(empty_rows_a,
		empty_rows_b)!
	assert empty_rows_x.shape == [2]
	assert empty_rows_residuals.size == 0
	assert empty_rows_rank == 0
	assert empty_rows_singular_values.size == 0

	empty_columns_a := vtl.empty[vcomplex.Complex]([2, 0])
	b := vtl.from_1d[vcomplex.Complex]([vcomplex.Complex{ re: 1 }, vcomplex.Complex{ re: 2 }])!
	empty_columns_x, empty_columns_residuals, empty_columns_rank, empty_columns_singular_values := lstsq_complex(empty_columns_a,
		b)!
	assert empty_columns_x.shape == [0]
	assert empty_columns_residuals.size == 0
	assert empty_columns_rank == 0
	assert empty_columns_singular_values.size == 0

	a := vtl.from_2d[vcomplex.Complex]([
		[vcomplex.Complex{ re: 1 }, vcomplex.Complex{}],
		[vcomplex.Complex{}, vcomplex.Complex{ re: 1 }],
	])!
	no_columns_b := vtl.empty[vcomplex.Complex]([2, 0])
	no_columns_x, no_columns_residuals, no_columns_rank, no_columns_singular_values := lstsq_complex(a,
		no_columns_b)!
	assert no_columns_x.shape == [2, 0]
	assert no_columns_residuals.size == 0
	assert no_columns_rank == 2
	assert no_columns_singular_values.shape == [2]
}
