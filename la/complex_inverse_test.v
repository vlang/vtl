module la

import math
import math.complex as vcomplex
import vtl

fn test_inv_complex_supports_batched_pivoted_matrices() ! {
	a := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{},
		vcomplex.Complex{ re: 1, im: 1 },
		vcomplex.Complex{ re: 2 },
		vcomplex.Complex{ re: 3, im: -1 },
		vcomplex.Complex{ re: 1 },
		vcomplex.Complex{},
		vcomplex.Complex{},
		vcomplex.Complex{ re: 1 },
	], [2, 2, 2])!

	inverse := inv_complex(a)!
	assert inverse.shape == [2, 2, 2]
	identity := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 1 },
		vcomplex.Complex{},
		vcomplex.Complex{},
		vcomplex.Complex{ re: 1 },
	], [2, 2])!
	product := matmul(a, inverse)!
	for batch in 0 .. 2 {
		for i in 0 .. 2 {
			for j in 0 .. 2 {
				expected := identity.get([i, j])
				actual := product.get([batch, i, j])
				assert math.abs(actual.re - expected.re) < 1e-12
				assert math.abs(actual.im - expected.im) < 1e-12
			}
		}
	}
}

fn test_inv_complex_rejects_singular_matrices() ! {
	a := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 1, im: 1 },
		vcomplex.Complex{ re: 2, im: 2 },
		vcomplex.Complex{ re: 2 },
		vcomplex.Complex{ re: 4 },
	], [2, 2])!
	if _ := inv_complex(a) {
		assert false, 'inv_complex must reject singular matrices'
	}
}

fn test_inv_complex_preserves_empty_shapes_and_rejects_non_square_input() ! {
	empty := vtl.empty[vcomplex.Complex]([0, 0])
	assert inv_complex(empty)!.shape == [0, 0]
	empty_batch := vtl.empty[vcomplex.Complex]([0, 2, 2])
	assert inv_complex(empty_batch)!.shape == [0, 2, 2]

	non_square := vtl.zeros[vcomplex.Complex]([2, 3])
	if _ := inv_complex(non_square) {
		assert false, 'inv_complex must reject non-square matrices'
	}
}
