module la

import vtl

fn test_multi_dot_selects_lowest_cost_matrix_chain_order() ! {
	a := vtl.ones[f64]([100, 10])
	b := vtl.ones[f64]([10, 1000])
	c := vtl.ones[f64]([1000, 1])
	dimensions := [100, 10, 1000, 1]
	splits := matrix_chain_splits(dimensions)
	assert splits[2] == 0 // A * (B * C), rather than (A * B) * C.
	result := multi_dot[f64]([a, b, c])!
	assert result.shape == [100, 1]
	assert result.get_nth(0) == 10000.0
}

fn test_multi_dot_supports_vector_endpoints_and_vector_dot() ! {
	left := vtl.from_1d([1.0, 2.0])!
	matrix := vtl.from_2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])!
	right := vtl.from_1d([1.0, 1.0, 1.0])!
	assert multi_dot[f64]([left, matrix, right])!.get_nth(0) == 36.0
	assert multi_dot[f64]([left, vtl.from_1d([3.0, 4.0])!])!.get_nth(0) == 11.0
}

fn test_multi_dot_rejects_invalid_operand_lists() {
	vector := vtl.from_1d([1.0, 2.0])!
	matrix := vtl.ones[f64]([2, 2])
	if _ := multi_dot[f64]([matrix]) {
		assert false, 'multi_dot must require at least two operands'
	}
	if _ := multi_dot[f64]([matrix, vector, matrix]) {
		assert false, 'multi_dot must reject vectors in the middle'
	}
	if _ := multi_dot[f64]([matrix, vtl.ones[f64]([3, 2])]) {
		assert false, 'multi_dot must reject incompatible matrix dimensions'
	}
}
