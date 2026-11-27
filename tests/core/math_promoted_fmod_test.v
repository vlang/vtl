module core

import math
import math.complex
import vtl

fn test_fmod_promoted_mixed_dtype_broadcast_and_output_validation() ! {
	values := vtl.from_array[f32]([-5.0, 5.0], [2, 1])!
	divisors := vtl.from_array[f64]([3.0, -3.0], [1, 2])!

	result := vtl.fmod_promoted[f64, f32, f64](values, divisors)!
	assert result.dtype() == .float64
	assert result.shape == [2, 2]
	assert result.to_array() == [-2.0, -2.0, 2.0, 2.0]

	if _ := vtl.fmod_promoted[f32, f32, f64](values, divisors) {
		assert false, 'fmod must reject an output dtype that differs from promotion'
	}
}

fn test_fmod_promoted_integer_precision_and_zero_divisors() ! {
	minimum := i64(-9223372036854775807 - 1)
	values := vtl.from_1d[i64]([9007199254740993, minimum, 5])!
	divisors := vtl.from_1d[i64]([2, -1, 0])!

	result := vtl.fmod_promoted[i64, i64, i64](values, divisors)!
	assert result.dtype() == .int64
	assert result.to_array() == [i64(1), 0, 0]
}

fn test_fmod_promoted_float_ieee_values_and_signed_zero() ! {
	values := vtl.from_1d[f64]([-0.0, 5.0, -5.0, math.inf(1), math.nan()])!
	divisors := vtl.from_1d[f32]([3.0, 0.0, -3.0, 2.0, 2.0])!

	result := vtl.fmod_promoted[f64, f64, f32](values, divisors)!
	assert result.get_nth(0) == 0.0
	assert math.signbit(result.get_nth(0))
	assert math.is_nan(result.get_nth(1))
	assert result.get_nth(2) == -2.0
	assert math.is_nan(result.get_nth(3))
	assert math.is_nan(result.get_nth(4))
}

fn test_fmod_promoted_weak_scalars_preserve_operand_order_and_empty_shapes() ! {
	float_values := vtl.from_1d[f32]([-5.0, 5.0])!
	right_scalar := vtl.fmod_scalar_promoted[f32, f32, f64](float_values, 3.0)!
	left_scalar := vtl.fmod_scalar_left_promoted[f32, f32, f64](3.0, float_values)!

	assert right_scalar.dtype() == .float32
	assert right_scalar.to_array() == [f32(-2.0), 2.0]
	assert left_scalar.dtype() == .float32
	assert left_scalar.to_array() == [f32(3.0), 3.0]

	integer_values := vtl.from_1d[i16]([-5, 5])!
	integer_right := vtl.fmod_scalar_promoted[f64, i16, f32](integer_values, f32(3.0))!
	assert integer_right.dtype() == .float64
	assert integer_right.to_array() == [-2.0, 2.0]

	empty_values := vtl.empty[i16]([0])
	empty_result := vtl.fmod_scalar_promoted[i16, i16, i8](empty_values, i8(3))!
	assert empty_result.shape == [0]
	assert empty_result.to_array().len == 0
}

fn test_fmod_promoted_accepts_boolean_and_rejects_complex_inputs() ! {
	boolean_values := vtl.from_1d([true, false])!
	boolean_remainders := vtl.fmod_promoted[i8, bool, i8](boolean_values,
		vtl.from_1d[i8]([2])!)!
	assert boolean_remainders.dtype() == .int8
	assert boolean_remainders.to_array() == [i8(1), 0]

	complex_values := vtl.from_1d[complex.Complex]([complex.complex(1.0, 1.0)])!
	if _ := vtl.fmod_promoted[complex.Complex, complex.Complex, complex.Complex](complex_values,
		complex_values) {
		assert false, 'fmod must reject complex tensors'
	}
	if _ := vtl.fmod_scalar_promoted[complex.Complex, complex.Complex, f64](complex_values, 2.0) {
		assert false, 'fmod must reject complex weak scalars'
	}
}
