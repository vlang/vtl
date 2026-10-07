module la

import math
import vtl

fn test_svdvals_matches_numpy_and_sorts_descending() ! {
	input := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
	values := svdvals(input)!
	assert values.shape == [2]
	assert math.abs(values.get_nth(0) - 5.464985704219043) < 1e-10
	assert math.abs(values.get_nth(1) - 0.36596619062625746) < 1e-10
}

fn test_svdvals_supports_batched_rectangular_inputs() ! {
	input := vtl.from_array([3, 0, 0, 0, 4, 0, 5, 0, 0, 0, 12, 0], [2, 2, 3])!
	values := svdvals(input)!
	assert values.shape == [2, 2]
	assert values.to_array() == [4.0, 3.0, 12.0, 5.0]
}

fn test_svdvals_handles_empty_inputs() {
	empty := vtl.empty[f64]([3, 0])
	assert svdvals(empty)!.shape == [0]
}

fn test_svdvals_rejects_rank_below_two() {
	vector := vtl.from_1d([1.0, 2.0])!
	if _ := svdvals(vector) {
		assert false, 'svdvals must reject rank-one input'
	} else {
		assert true
	}
}

fn test_svdvals_rejects_non_finite_inputs() {
	non_finite := vtl.from_2d([[f64(math.inf(1))]])!
	if _ := svdvals(non_finite) {
		assert false, 'svdvals must reject non-finite inputs'
	} else {
		assert true
	}
}
