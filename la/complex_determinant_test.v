module la

import math
import math.complex as vcomplex
import vtl

fn test_det_complex_uses_pivoting_and_preserves_batches() ! {
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

	determinants := det_complex(a)!
	assert determinants.shape == [2]
	assert math.abs(determinants.get_nth(0).re + 2) < 1e-12
	assert math.abs(determinants.get_nth(0).im + 2) < 1e-12
	assert determinants.get_nth(1) == vcomplex.Complex{ re: 1 }
}

fn test_det_complex_handles_singular_and_empty_matrices() ! {
	singular := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 1, im: 1 },
		vcomplex.Complex{ re: 2, im: 2 },
		vcomplex.Complex{ re: 2 },
		vcomplex.Complex{ re: 4 },
	], [2, 2])!
	determinant := det_complex(singular)!
	assert determinant.shape == [1]
	assert determinant.get_nth(0) == vcomplex.Complex{}

	empty := vtl.empty[vcomplex.Complex]([0, 0])
	assert det_complex(empty)!.get_nth(0) == vcomplex.Complex{ re: 1 }
	empty_batch := vtl.empty[vcomplex.Complex]([0, 2, 2])
	assert det_complex(empty_batch)!.shape == [0]
}
