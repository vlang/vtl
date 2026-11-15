module core

import math
import math.complex
import vtl

fn test_floor_divide_promoted_mixed_dtype_broadcast_and_output_validation() ! {
	numerators := vtl.from_array[f32]([-3.5, 5.0], [2, 1])!
	divisors := vtl.from_array[i16]([2, -2], [1, 2])!

	result := vtl.floor_divide_promoted[f32, f32, i16](numerators, divisors)!
	assert result.dtype() == .float32
	assert result.shape == [2, 2]
	assert result.to_array() == [f32(-2), 1, 2, -3]

	if _ := vtl.floor_divide_promoted[f64, f32, i16](numerators, divisors) {
		assert false, 'floor division must reject an output dtype that differs from promotion'
	}
}

fn test_floor_divide_promoted_integer_zero_and_signed_overflow() ! {
	values := vtl.from_1d[i8]([-3, 3, -128, 1])!
	divisors := vtl.from_1d[i8]([2, -2, -1, 0])!
	result := vtl.floor_divide_promoted[i8, i8, i8](values, divisors)!

	assert result.to_array() == [i8(-2), -2, -128, 0]

	unsigned_values := vtl.from_1d[u8]([255, 0, 7])!
	unsigned_divisors := vtl.from_1d[u16]([2, 0, 3])!
	unsigned_result := vtl.floor_divide_promoted[u16, u8, u16](unsigned_values,
		unsigned_divisors)!
	assert unsigned_result.dtype() == .uint16
	assert unsigned_result.to_array() == [u16(127), 0, 2]
}

fn test_floor_divide_weak_scalar_promoted_preserves_operand_order() ! {
	values := vtl.from_1d[i16]([3, -3])!
	integer_scalar := vtl.floor_divide_scalar_promoted[i16, i16, i8](values, i8(2))!
	array_left := vtl.floor_divide_scalar_promoted[f64, i16, f32](values, f32(2.5))!
	scalar_left := vtl.floor_divide_scalar_left_promoted[f64, i16, f32](f32(-3.0),
		vtl.from_1d[i16]([2, -2])!)!

	assert integer_scalar.dtype() == .int16
	assert integer_scalar.to_array() == [i16(1), -2]
	assert array_left.dtype() == .float64
	assert array_left.to_array() == [1.0, -2.0]
	assert scalar_left.dtype() == .float64
	assert scalar_left.to_array() == [-2.0, 1.0]

	if _ := vtl.floor_divide_scalar_promoted[i8, i8, i16](vtl.from_1d[i8]([1])!, i16(128)) {
		assert false, 'out-of-range weak integer scalars must be rejected'
	}
}

fn test_floor_divide_weak_scalar_promoted_float_ieee_and_dtype() ! {
	values := vtl.from_1d[f32]([-3.5, 3.5, 0.0])!
	result := vtl.floor_divide_scalar_promoted[f32, f32, int](values, 0)!

	assert result.dtype() == .float32
	assert math.is_inf(result.get_nth(0), -1)
	assert math.is_inf(result.get_nth(1), 1)
	assert math.is_nan(result.get_nth(2))
}

fn test_floor_divide_promoted_rejects_boolean_and_complex_inputs() ! {
	boolean_values := vtl.from_1d([true])!
	if _ := vtl.floor_divide_promoted[f64, bool, bool](boolean_values, boolean_values) {
		assert false, 'floor division must reject boolean tensors'
	}

	complex_values := vtl.from_1d[complex.Complex]([complex.complex(1.0, 1.0)])!
	if _ := vtl.floor_divide_promoted[complex.Complex, complex.Complex, complex.Complex](complex_values,
		complex_values) {
		assert false, 'floor division must reject complex tensors'
	}
	if _ := vtl.floor_divide_scalar_promoted[complex.Complex, complex.Complex, f64](complex_values, 2.0) {
		assert false, 'floor division must reject complex weak scalars'
	}
}
