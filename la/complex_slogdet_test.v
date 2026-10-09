module la

import math
import math.complex as vcomplex
import vtl

fn test_slogdet_complex_matches_phase_and_log_magnitude_with_batches() ! {
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
	phase, logabsdet := slogdet_complex(a)!
	assert phase.shape == [2]
	assert logabsdet.shape == [2]
	assert math.abs(phase.get_nth(0).re + 1 / math.sqrt(2)) < 1e-12
	assert math.abs(phase.get_nth(0).im + 1 / math.sqrt(2)) < 1e-12
	assert math.abs(logabsdet.get_nth(0) - math.log(math.sqrt(8))) < 1e-12
	assert phase.get_nth(1) == vcomplex.Complex{ re: 1 }
	assert logabsdet.get_nth(1) == 0
}

fn test_slogdet_complex_handles_singular_and_empty_matrices() ! {
	singular := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: 1, im: 1 },
		vcomplex.Complex{ re: 2, im: 2 },
		vcomplex.Complex{ re: 2 },
		vcomplex.Complex{ re: 4 },
	], [2, 2])!
	phase, logabsdet := slogdet_complex(singular)!
	assert phase.get_nth(0) == vcomplex.Complex{}
	assert math.is_inf(logabsdet.get_nth(0), -1)

	empty := vtl.empty[vcomplex.Complex]([0, 0])
	empty_phase, empty_logabsdet := slogdet_complex(empty)!
	assert empty_phase.get_nth(0) == vcomplex.Complex{ re: 1 }
	assert empty_logabsdet.get_nth(0) == 0
	empty_batch := vtl.empty[vcomplex.Complex]([0, 2, 2])
	empty_batch_phase, _ := slogdet_complex(empty_batch)!
	assert empty_batch_phase.shape == [0]
}

fn test_slogdet_complex_rejects_non_square_and_non_finite_inputs() ! {
	nonsquare := vtl.zeros[vcomplex.Complex]([2, 3])
	if _, _ := slogdet_complex(nonsquare) {
		assert false, 'slogdet_complex must reject non-square matrices'
	}
	nonfinite := vtl.from_array[vcomplex.Complex]([
		vcomplex.Complex{ re: math.inf(1) },
	], [1, 1])!
	if _, _ := slogdet_complex(nonfinite) {
		assert false, 'slogdet_complex must reject non-finite values'
	}
}
