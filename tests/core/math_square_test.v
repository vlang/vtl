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
