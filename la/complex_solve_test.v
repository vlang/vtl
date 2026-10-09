module la

import math
import math.complex as vcomplex
import vtl

fn test_solve_complex_uses_partial_pivoting_for_vector_rhs() ! {
	a := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{},
		vcomplex.Complex{ re: 1, im: 1 },
		vcomplex.Complex{ re: 2 },
		vcomplex.Complex{ re: 3, im: -1 },
	], [2, 2])!
	b := vtl.from_1d[vcomplex.Complex]([
		vcomplex.Complex{ re: 1, im: 3 },
		vcomplex.Complex{ re: 9, im: -1 },
	])!

	result := solve_complex(a, b)!
	assert result.dtype() == .complex128
	assert result.shape == [2]
	assert math.abs(result.get_nth(0).re - 1) < 1e-12
	assert math.abs(result.get_nth(0).im + 1) < 1e-12
	assert math.abs(result.get_nth(1).re - 2) < 1e-12
	assert math.abs(result.get_nth(1).im - 1) < 1e-12
}

fn test_solve_complex_supports_matrix_rhs_and_broadcast_batches() ! {
	base := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{},
		vcomplex.Complex{ re: 1, im: 1 },
		vcomplex.Complex{ re: 2 },
		vcomplex.Complex{ re: 3, im: -1 },
	], [2, 2])!
	stack := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{},
		vcomplex.Complex{ re: 1, im: 1 },
		vcomplex.Complex{ re: 2 },
		vcomplex.Complex{ re: 3, im: -1 },
		vcomplex.Complex{ re: 1 },
		vcomplex.Complex{},
		vcomplex.Complex{},
		vcomplex.Complex{ re: 1 },
	], [2, 2, 2])!
	b := vtl.from_1d[vcomplex.Complex]([
		vcomplex.Complex{ re: 1, im: 3 },
		vcomplex.Complex{ re: 9, im: -1 },
	])!

	result := solve_complex(stack, b)!
	assert result.shape == [2, 2]
	assert math.abs(result.get([0, 0]).re - 1) < 1e-12
	assert math.abs(result.get([0, 0]).im + 1) < 1e-12
	assert math.abs(result.get([0, 1]).re - 2) < 1e-12
	assert math.abs(result.get([0, 1]).im - 1) < 1e-12
	assert result.get([1, 0]) == b.get_nth(0)
	assert result.get([1, 1]) == b.get_nth(1)

	identity := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 1 },
		vcomplex.Complex{},
		vcomplex.Complex{},
		vcomplex.Complex{ re: 1 },
	], [2, 2])!
	inverse := solve_complex(base, identity)!
	product := matmul(base, inverse)!
	assert math.abs(product.get([0, 0]).re - 1) < 1e-12
	assert math.abs(product.get([0, 1]).re) < 1e-12
	assert math.abs(product.get([1, 0]).re) < 1e-12
	assert math.abs(product.get([1, 1]).re - 1) < 1e-12
}

fn test_solve_complex_rejects_singular_matrix() ! {
	a := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 1, im: 1 },
		vcomplex.Complex{ re: 2, im: 2 },
		vcomplex.Complex{ re: 2 },
		vcomplex.Complex{ re: 4 },
	], [2, 2])!
	b := vtl.from_1d[vcomplex.Complex]([vcomplex.Complex{ re: 1 }, vcomplex.Complex{ re: 2 }])!
	if _ := solve_complex(a, b) {
		assert false, 'solve_complex must reject a singular matrix'
	}
}
