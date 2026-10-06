module stats

import vtl

fn test_bincount_counts_values_and_respects_minlength() ! {
	input := vtl.from_1d([0, 1, 1, 3, 2, 1])!
	assert bincount[int](input, 0)!.to_array() == [1, 3, 1, 1]
	assert bincount[int](input, 6)!.to_array() == [1, 3, 1, 1, 0, 0]
	assert bincount(vtl.from_1d([]u8{})!, 3)!.to_array() == [0, 0, 0]
}

fn test_bincount_weighted_sums_values() ! {
	input := vtl.from_1d([0, 1, 1, 3])!
	weights := vtl.from_1d([0.5, 1.5, 2.0, 4.0])!
	assert bincount_weighted[int, f64](input, weights, 5)!.to_array() == [0.5, 3.5, 0, 4.0, 0]
}

fn test_bincount_rejects_invalid_values_and_shapes() {
	if _ := bincount(vtl.from_1d([0, -1, 1])!, 0) {
		assert false, 'bincount must reject negative values'
	}
	if _ := bincount(vtl.from_1d([0.0, 1.0])!, 0) {
		assert false, 'bincount must reject non-integer input'
	}
	if _ := bincount(vtl.from_array([0, 1], [1, 2])!, 0) {
		assert false, 'bincount must reject non-vector input'
	}
	if _ := bincount(vtl.from_1d([0, 1])!, -1) {
		assert false, 'bincount must reject negative minlength'
	}
	if _ := bincount_weighted[int, f64](vtl.from_1d([0, 1])!, vtl.from_1d([1.0])!, 0) {
		assert false, 'bincount must reject weights with a different length'
	}
}
