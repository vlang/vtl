module core

import math
import math.complex
import vtl

fn test_square_preserves_integer_dtype_and_exact_large_values() ! {
	values := vtl.from_1d[i64]([1_000_000_001, -3])!
	result := values.square()!

	assert result.dtype() == .int64
	assert result.shape == values.shape
	assert result.to_array() == [i64(1_000_000_002_000_000_001), 9]

	unsigned := vtl.from_1d[u64]([4_000_000_001])!.square()!
	assert unsigned.dtype() == .uint64
	assert unsigned.to_array() == [u64(16_000_000_008_000_000_001)]
}

fn test_square_handles_boolean_float_special_values_and_empty_shapes() ! {
	booleans := vtl.from_1d([true, false])!.square()!
	assert booleans.dtype() == .boolean
	assert booleans.to_array() == [true, false]

	floats := vtl.from_1d[f64]([-0.0, math.nan(), math.inf(1)])!.square()!
	assert floats.to_array()[0] == 0.0
	assert !math.signbit(floats.to_array()[0])
	assert math.is_nan(floats.to_array()[1])
	assert math.is_inf(floats.to_array()[2], 1)

	empty := vtl.empty[f32]([2, 0]).square()!
	assert empty.shape == [2, 0]
	assert empty.size == 0
}

fn test_square_supports_complex_and_rejects_strings() ! {
	complex_values := vtl.from_1d[complex.Complex]([complex.complex(3.0, 4.0)])!
	squared := complex_values.square()!
	assert squared.dtype() == .complex128
	assert squared.get_nth(0) == complex.complex(-7.0, 24.0)

	strings := vtl.from_1d(['not numeric'])!
	if _ := strings.square() {
		assert false, 'square must reject string tensors'
	}
}

fn test_square_into_reuses_output_and_supports_strided_views() ! {
	values := vtl.from_1d[int]([1, 2, 3])!
	mut output := vtl.full[int]([3], -1)
	result := values.square_into(mut output)!

	assert result.data == output.data
	assert output.to_array() == [1, 4, 9]

	input_matrix := vtl.from_2d([[1, 2], [3, 4]])!
	input_view := input_matrix.transpose([1, 0])!
	mut output_base := vtl.zeros[int]([2, 2])
	mut output_view := output_base.transpose([1, 0])!
	input_view.square_into(mut output_view)!

	assert output_view.to_array() == [1, 9, 4, 16]
	assert output_base.to_array() == [1, 4, 9, 16]

	mut gapped_base := vtl.full[int]([6], -1)
	gapped_parent := gapped_base.slice([0, 4])!
	mut gapped_output := gapped_parent.as_strided([2, 2], [3, 1])!
	input_matrix.square_into(mut gapped_output)!
	assert gapped_base.to_array() == [1, 4, -1, 9, 16, -1]
}

fn test_square_into_snapshots_overlapping_views() ! {
	mut storage := vtl.from_1d[int]([1, 2, 3, 4, 0])!
	input := storage.slice([0, 4])!
	mut output := storage.slice([1, 5])!

	input.square_into(mut output)!

	assert output.to_array() == [1, 4, 9, 16]
	assert storage.to_array() == [1, 1, 4, 9, 16]
}

fn test_square_into_rejects_internally_overlapping_output_views() ! {
	input := vtl.from_2d([[1, 2], [3, 4]])!

	mut zero_stride_base := vtl.from_1d[int]([10, 20, 30, 40])!
	mut zero_stride_output := zero_stride_base.as_strided([2, 2], [0, 1])!
	if _ := input.square_into(mut zero_stride_output) {
		assert false, 'square_into must reject zero-stride output dimensions'
	} else {
		assert err.msg().contains('non-overlapping')
		assert zero_stride_base.to_array() == [10, 20, 30, 40]
	}

	mut overlapping_base := vtl.from_1d[int]([10, 20, 30, 40])!
	mut overlapping_output := overlapping_base.as_strided([2, 2], [1, 1])!
	if _ := input.square_into(mut overlapping_output) {
		assert false, 'square_into must reject overlapping nonzero strides'
	} else {
		assert err.msg().contains('non-overlapping')
		assert overlapping_base.to_array() == [10, 20, 30, 40]
	}

	mut negative_stride_base := vtl.from_1d[int]([10, 20, 30])!
	mut negative_stride_output := negative_stride_base.as_strided([3], [-2])!
	negative_values := vtl.from_1d[int]([1, 2, 3])!
	if _ := negative_values.square_into(mut negative_stride_output) {
		assert false, 'square_into must reject negative-stride output layouts'
	} else {
		assert err.msg().contains('non-overlapping')
		assert negative_stride_base.to_array() == [10, 20, 30]
	}
}

fn test_square_into_validates_shape_and_handles_empty_tensors() ! {
	values := vtl.from_1d[int]([2, 3])!
	mut wrong_shape := vtl.full[int]([1], 7)
	if _ := values.square_into(mut wrong_shape) {
		assert false, 'square_into must reject a mismatched output shape'
	} else {
		assert err.msg().contains('shape')
		assert wrong_shape.to_array() == [7]
	}

	empty_input := vtl.empty[f32]([2, 0])
	mut empty_output := vtl.empty[f32]([2, 0])
	result := empty_input.square_into(mut empty_output)!
	assert result.shape == [2, 0]
	assert result.size == 0

	strings := vtl.from_1d(['not numeric'])!
	mut string_output := vtl.from_1d(['unchanged'])!
	if _ := strings.square_into(mut string_output) {
		assert false, 'square_into must reject string tensors'
	} else {
		assert err.msg().contains('numeric')
		assert string_output.to_array() == ['unchanged']
	}
}
