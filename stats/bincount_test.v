module stats

import vtl

fn test_bincount_counts_values_and_respects_minlength() ! {
	input := vtl.from_1d([0, 1, 1, 3, 2, 1])!
	assert bincount[int](input, 0)!.to_array() == [1, 3, 1, 1]
	assert bincount[int](input, 6)!.to_array() == [1, 3, 1, 1, 0, 0]
	assert bincount(vtl.from_1d([]u8{})!, 3)!.to_array() == [0, 0, 0]
	assert bincount(vtl.from_1d[i32]([0, 2, 2])!, 4)!.to_array() == [1, 0, 2, 0]
}

fn test_bincount_weighted_sums_values() ! {
	input := vtl.from_1d([0, 1, 1, 3])!
	weights := vtl.from_1d([0.5, 1.5, 2.0, 4.0])!
	assert bincount_weighted[int, f64](input, weights, 5)!.to_array() == [0.5, 3.5, 0, 4.0, 0]
}

fn test_bincount_explicit_output_and_accumulator_dtype() ! {
	input := vtl.from_1d([0, 1, 1, 3])!
	assert bincount_as[int, f64](input, 5)!.to_array() == [1.0, 2.0, 0.0, 1.0, 0.0]

	repeated := vtl.from_1d([0, 0])!
	weights := vtl.from_1d([f32(16_777_216), 1])!
	f32_counts := bincount_weighted_as[int, f32, f32](repeated, weights, 1)!
	f64_counts := bincount_weighted_as[int, f32, f64](repeated, weights, 1)!
	assert f32_counts.to_array() == [f32(16_777_216)]
	assert f64_counts.to_array() == [16_777_217.0]

	empty_input := vtl.from_1d([]u8{})!
	assert bincount_as[u8, f64](empty_input, 3)!.to_array() == [0.0, 0.0, 0.0]
	assert bincount_weighted_as[u8, f32, f64](empty_input, vtl.from_1d([]f32{})!,
		2)!.to_array() == [0.0, 0.0]
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
	if _ := bincount_weighted_as[int, bool, f64](vtl.from_1d([0])!, vtl.from_1d([true])!,
		0) {
		assert false, 'bincount must reject non-numeric weights'
	}
}
