module core

import math
import math.complex
import vtl

fn test_pow_promoted_broadcasts_and_uses_explicit_dtype_promotion() ! {
	bases := vtl.from_array[i16]([2, 3], [2, 1])!
	exponents := vtl.from_array[f32]([2.0, 3.0, 0.0], [1, 3])!

	result := vtl.pow_promoted[f32, i16, f32](bases, exponents)!
	expected := vtl.from_array[f32]([4.0, 8.0, 1.0, 9.0, 27.0, 1.0], [2, 3])!

	assert result.dtype() == .float32
	assert result.shape == [2, 3]
	assert result.allclose(expected, rtol: 0.0, atol: 0.0)!

	if _ := vtl.pow_promoted[f32, i16, f32](bases,
		vtl.from_array[f32]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], [
			3,
			4,
		])!) {
		assert false, 'power must reject incompatible broadcast shapes'
	}

	integer_bases := vtl.from_1d[i8]([2, 3])!
	integer_exponents := vtl.from_1d[i8]([3, 2])!
	integer_result := vtl.pow_promoted[i8, i8, i8](integer_bases, integer_exponents)!
	assert integer_result.to_array() == [i8(8), 9]
}

fn test_pow_promoted_handles_scalar_and_empty_broadcasts() ! {
	scalar := vtl.from_array[f64]([3.0], [])!
	exponent := vtl.from_array[i16]([2], [])!
	scalar_result := vtl.pow_promoted[f64, f64, i16](scalar, exponent)!
	assert scalar_result.shape.len == 0
	assert scalar_result.get_nth(0) == 9.0

	empty := vtl.empty[f32]([0, 2])
	row := vtl.from_2d[i8]([[2, 3]])!
	empty_result := vtl.pow_promoted[f32, f32, i8](empty, row)!
	assert empty_result.shape == [0, 2]
	assert empty_result.to_array().len == 0
}

fn test_pow_promoted_preserves_integer_precision_and_wraps_overflow() ! {
	large := i64(9007199254740993)
	large_result := vtl.pow_promoted[i64, i64, i8](vtl.from_1d([large])!,
		vtl.from_1d[i8]([1])!)!
	assert large_result.to_array() == [large]

	overflow := vtl.pow_promoted[i8, i8, i8](vtl.from_1d[i8]([3])!,
		vtl.from_1d[i8]([5])!)!
	assert overflow.to_array() == [i8(-13)]
	unsigned_overflow := vtl.pow_promoted[u8, u8, u8](vtl.from_1d[u8]([3])!,
		vtl.from_1d[u8]([6])!)!
	assert unsigned_overflow.to_array() == [u8(217)]

	negative_exponent := vtl.from_1d[i16]([-1])!
	if _ := vtl.pow_promoted[i16, i16, i16](vtl.from_1d[i16]([2])!,
		negative_exponent) {
		assert false, 'integer powers must reject negative exponents'
	}
}

fn test_pow_promoted_preserves_float_special_values_and_signed_zero() ! {
	bases := vtl.from_1d[f64]([-0.0, -0.0, -0.0, -2.0, math.nan(), 0.0])!
	exponents := vtl.from_1d[f64]([3.0, 2.0, -3.0, 0.5, 0.0, -1.0])!
	result := vtl.pow_promoted[f64, f64, f64](bases, exponents)!

	assert result.get_nth(0) == 0.0
	assert math.signbit(result.get_nth(0))
	assert result.get_nth(1) == 0.0
	assert !math.signbit(result.get_nth(1))
	assert math.is_inf(result.get_nth(2), -1)
	assert math.is_nan(result.get_nth(3))
	assert result.get_nth(4) == 1.0
	assert math.is_inf(result.get_nth(5), 1)
}

fn test_pow_promoted_checks_output_dtype_and_rejects_lossy_or_unsupported_inputs() ! {
	left := vtl.from_1d[i16]([2])!
	right := vtl.from_1d[i8]([3])!
	if _ := vtl.pow_promoted[i32, i16, i8](left, right) {
		assert false, 'output dtype must equal promote_types'
	}

	large_integer := vtl.from_1d[i64]([9007199254740993])!
	float_exponent := vtl.from_1d[f64]([1.0])!
	if _ := vtl.pow_promoted[f64, i64, f64](large_integer, float_exponent) {
		assert false, '64-bit integers must not be silently rounded to float64'
	}

	complex_values := vtl.from_1d[complex.Complex]([complex.complex(1.0, 1.0)])!
	if _ := vtl.pow_promoted[complex.Complex, complex.Complex, complex.Complex](complex_values,
		complex_values) {
		assert false, 'complex power is intentionally unsupported'
	}

	string_values := vtl.from_1d[string](['a'])!
	if _ := vtl.pow_promoted[string, string, string](string_values, string_values) {
		assert false, 'string power is intentionally unsupported'
	}
}

fn test_pow_promoted_boolean_pair_uses_existing_promotion_contract() ! {
	bases := vtl.from_1d([false, true, true, false])!
	exponents := vtl.from_1d([false, false, true, true])!

	result := vtl.pow_promoted[bool, bool, bool](bases, exponents)!
	assert result.dtype() == .boolean
	assert result.to_array() == [true, true, true, false]
}
