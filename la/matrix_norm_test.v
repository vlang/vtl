module la

import vtl
import math

fn test_matrix_norm_orders_match_expected_batched_values() ! {
	input := vtl.from_array([3, 0, 0, 0, 4, 0, 5, 0, 0, 0, 12, 0], [2, 2, 3])!
	assert matrix_norm(input, ord: 'fro')!.to_array() == [5.0, 13.0]
	assert matrix_norm(input, ord: '1')!.to_array() == [4.0, 12.0]
	assert matrix_norm(input, ord: '-1')!.to_array() == [0.0, 0.0]
	assert matrix_norm(input, ord: 'inf')!.to_array() == [4.0, 12.0]
	assert matrix_norm(input, ord: '-inf')!.to_array() == [3.0, 5.0]
	assert matrix_norm(input, ord: '2')!.to_array() == [4.0, 12.0]
	assert matrix_norm(input, ord: '-2')!.to_array() == [3.0, 5.0]
	assert matrix_norm(input, ord: 'nuc')!.to_array() == [7.0, 17.0]
}

fn test_matrix_norm_default_keepdims_and_aliases() ! {
	input := vtl.from_array([3, 0, 0, 0, 4, 0], [1, 2, 3])!
	assert matrix_norm(input)!.to_array() == [5.0]
	kept := matrix_norm(input, keepdims: true)!
	assert kept.shape == [1, 1, 1]
	assert kept.get_nth(0) == 5.0
	assert matrix_norm(input, ord: 'I')!.get_nth(0) == 4.0
}

fn test_matrix_norm_spectral_orders_match_numpy_fixture() ! {
	input := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
	assert math.abs(matrix_norm(input, ord: '2')!.get_nth(0) - 5.464985704219043) < 1e-10
	assert math.abs(matrix_norm(input, ord: '-2')!.get_nth(0) - 0.36596619062625746) < 1e-10
	assert math.abs(matrix_norm(input, ord: 'nuc')!.get_nth(0) - 5.8309518948453) < 1e-10
	assert matrix_norm(input, ord: '1')!.get_nth(0) == 6.0
	assert matrix_norm(input, ord: '-1')!.get_nth(0) == 4.0
	assert matrix_norm(input, ord: 'inf')!.get_nth(0) == 7.0
	assert matrix_norm(input, ord: '-inf')!.get_nth(0) == 3.0
}

fn test_matrix_norm_svd_handles_tall_rectangular_inputs() ! {
	input := vtl.from_2d([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])!
	assert math.abs(matrix_norm(input, ord: '2')!.get_nth(0) - 9.525518091565107) < 1e-10
	assert math.abs(matrix_norm(input, ord: '-2')!.get_nth(0) - 0.5143005806586443) < 1e-10
	assert math.abs(matrix_norm(input, ord: 'nuc')!.get_nth(0) - 10.039818672223753) < 1e-10
}

fn test_matrix_norm_svd_matches_numpy_for_wide_mixed_matrix() ! {
	input := vtl.from_array([0.25, -1.5, 3, 2, -0.25, 4, 2.25, -0.5, 1.75, 5, -3, 0.75, 2.5, -4,
		1, 6, -2, 0.5, 3.5, -1],
		[4, 5])!
	assert math.abs(matrix_norm(input, ord: '2')!.get_nth(0) - 9.65756322062387) < 1e-10
	assert math.abs(matrix_norm(input, ord: '-2')!.get_nth(0) - 2.02844842801882) < 1e-10
	assert math.abs(matrix_norm(input, ord: 'nuc')!.get_nth(0) - 21.823374602362687) < 1e-10
}

fn test_matrix_norm_rejects_invalid_rank_and_order() {
	vector := vtl.from_1d([1, 2, 3])!
	if _ := matrix_norm(vector) {
		assert false, 'matrix_norm must reject rank-one input'
	}
	matrix := vtl.from_2d([[1, 2], [3, 4]])!
	if _ := matrix_norm(matrix, ord: '0') {
		assert false, 'matrix_norm must reject unsupported orders'
	}
}

fn test_matrix_norm_empty_matrix_rules() ! {
	empty := vtl.empty[f64]([2, 0])
	assert matrix_norm(empty, ord: 'fro')!.get_nth(0) == 0.0
	assert matrix_norm(empty, ord: 'nuc')!.get_nth(0) == 0.0
	if _ := matrix_norm(empty, ord: '-2') {
		assert false, 'smallest singular value is undefined for an empty matrix'
	}
}
