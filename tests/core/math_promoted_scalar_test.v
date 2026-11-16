module core

import vtl
import math
import math.complex

fn test_weak_scalar_promotion_preserves_float_tensor_dtype() ! {
	values := vtl.from_1d[f32]([1.5, -2.0])!

	added := vtl.add_scalar_promoted[f32, f32, f64](values, 0.25)!
	assert added.dtype() == .float32
	assert math.abs(f64(added.get_nth(0)) - 1.75) < 1e-6
	assert math.abs(f64(added.get_nth(1)) + 1.75) < 1e-6

	subtracted := vtl.subtract_scalar_promoted[f32, f32, f64](values, 0.25)!
	assert math.abs(f64(subtracted.get_nth(0)) - 1.25) < 1e-6
	assert math.abs(f64(subtracted.get_nth(1)) + 2.25) < 1e-6

	multiplied := vtl.multiply_scalar_promoted[f32, f32, i64](values, 2)!
	assert multiplied.dtype() == .float32
	assert math.abs(f64(multiplied.get_nth(0)) - 3.0) < 1e-6
	assert math.abs(f64(multiplied.get_nth(1)) + 4.0) < 1e-6

	divided := vtl.divide_scalar_promoted[f32, f32, f64](values, 2.0)!
	assert divided.dtype() == .float32
	assert math.abs(f64(divided.get_nth(0)) - 0.75) < 1e-6
	assert math.abs(f64(divided.get_nth(1)) + 1.0) < 1e-6
}

fn test_weak_scalar_promotion_preserves_integer_dtype_when_scalar_fits() ! {
	values := vtl.from_1d[i8]([10, -5])!

	added := vtl.add_scalar_promoted[i8, i8, int](values, 2)!
	assert added.dtype() == .int8
	assert added.to_array() == [12, -3]

	subtracted := vtl.subtract_scalar_promoted[i8, i8, int](values, 2)!
	assert subtracted.dtype() == .int8
	assert subtracted.to_array() == [8, -7]

	remainders := vtl.remainder_scalar_promoted[i8, i8, int](values, 3)!
	assert remainders.dtype() == .int8
	assert remainders.to_array() == [1, 1]

	divided := vtl.divide_scalar_promoted[f64, i8, int](values, 2)!
	assert divided.dtype() == .float64
	assert divided.to_array() == [5.0, -2.5]
}

fn test_weak_scalar_left_promotion_preserves_operand_order_and_dtype() ! {
	values := vtl.from_1d[i8]([-3, 2])!

	added := vtl.add_scalar_left_promoted[i8, i8, int](5, values)!
	assert added.dtype() == .int8
	assert added.to_array() == [2, 7]

	subtracted := vtl.subtract_scalar_left_promoted[i8, i8, int](5, values)!
	assert subtracted.dtype() == .int8
	assert subtracted.to_array() == [8, 3]

	multiplied := vtl.multiply_scalar_left_promoted[i8, i8, int](5, values)!
	assert multiplied.dtype() == .int8
	assert multiplied.to_array() == [-15, 10]

	divided := vtl.divide_scalar_left_promoted[f64, i8, int](5, values)!
	assert divided.dtype() == .float64
	assert divided.to_array() == [-5.0 / 3.0, 2.5]

	remainders := vtl.remainder_scalar_left_promoted[i8, i8, int](5, values)!
	assert remainders.dtype() == .int8
	assert remainders.to_array() == [-1, 1]

	if _ := vtl.add_scalar_left_promoted[i8, i8, int](128, values) {
		assert false, 'out-of-range left scalar must not wrap into the tensor dtype'
	} else {
		assert err.msg().contains('out of range')
	}
}

fn test_weak_complex_scalar_left_promotion_preserves_operand_order() ! {
	values := vtl.from_1d[f32]([1.5, -2.0])!
	scalar := complex.complex(0.25, 0.5)

	subtracted := vtl.subtract_scalar_left_promoted[complex.Complex, f32, complex.Complex](scalar,
		values)!
	assert subtracted.dtype() == .complex128
	assert subtracted.get_nth(0).re == -1.25
	assert subtracted.get_nth(0).im == 0.5
	assert subtracted.get_nth(1).re == 2.25
	assert subtracted.get_nth(1).im == 0.5

	added := vtl.add_scalar_left_promoted[complex.Complex, f32, complex.Complex](scalar, values)!
	assert added.get_nth(0).re == 1.75
	assert added.get_nth(0).im == 0.5
	assert added.get_nth(1).re == -1.75
	assert added.get_nth(1).im == 0.5

	multiplied := vtl.multiply_scalar_left_promoted[complex.Complex, f32, complex.Complex](scalar,
		values)!
	assert multiplied.get_nth(0).re == 0.375
	assert multiplied.get_nth(0).im == 0.75
	assert multiplied.get_nth(1).re == -0.5
	assert multiplied.get_nth(1).im == -1.0

	divided := vtl.divide_scalar_left_promoted[complex.Complex, f32, complex.Complex](scalar,
		values)!
	assert math.abs(divided.get_nth(0).re - 1.0 / 6.0) < 1e-12
	assert math.abs(divided.get_nth(0).im - 1.0 / 3.0) < 1e-12
	assert divided.get_nth(1).re == -0.125
	assert divided.get_nth(1).im == -0.25

	if _ := vtl.remainder_scalar_left_promoted[complex.Complex, f32, complex.Complex](scalar,
		values) {
		assert false, 'complex remainder must be rejected when the scalar is on the left'
	} else {
		assert err.msg().contains('remainder')
	}
}

fn test_weak_float_scalar_promotes_integer_tensor_to_float64() ! {
	values := vtl.from_1d[i8]([10, -5])!
	added := vtl.add_scalar_promoted[f64, i8, f64](values, 0.5)!

	assert added.dtype() == .float64
	assert added.to_array() == [10.5, -4.5]
	assert values.to_array() == [i8(10), -5]
}

fn test_weak_integer_scalar_range_and_result_dtype_are_validated() ! {
	values := vtl.from_1d[i8]([10, -5])!
	if _ := vtl.add_scalar_promoted[i8, i8, int](values, 128) {
		assert false, 'weak integer scalar outside the array dtype range must fail'
	} else {
		assert err.msg().contains('out of range')
	}

	if _ := vtl.add_scalar_promoted[i64, i8, int](values, 2) {
		assert false, 'weak scalar operation must reject an incorrect output dtype'
	} else {
		assert err.msg().contains('does not match')
	}

	unsigned := vtl.from_1d[u8]([1])!
	if _ := vtl.add_scalar_promoted[u8, u8, int](unsigned, -1) {
		assert false, 'negative weak integer scalar must not wrap into an unsigned dtype'
	} else {
		assert err.msg().contains('out of range')
	}

	wide := vtl.from_1d[i64]([1])!
	too_wide := (u64(1) << 63)
	if _ := vtl.add_scalar_promoted[i64, i64, u64](wide, too_wide) {
		assert false, 'unsigned scalar outside signed range must fail'
	} else {
		assert err.msg().contains('out of range')
	}
}

fn test_weak_scalar_promotion_handles_strided_and_empty_tensors() ! {
	matrix := vtl.from_2d[f32]([[1, 2, 3], [4, 5, 6]])!
	transposed := matrix.transpose([1, 0])!
	shifted := vtl.add_scalar_promoted[f32, f32, f64](transposed, 0.5)!
	assert shifted.shape == [3, 2]
	expected := [1.5, 4.5, 2.5, 5.5, 3.5, 6.5]
	for index, value in expected {
		assert math.abs(f64(shifted.get_nth(index)) - value) < 1e-6
	}

	empty := vtl.from_array[f32]([]f32{}, [0])!
	assert vtl.add_scalar_promoted[f32, f32, f64](empty, 1.0)!.size == 0
}

fn test_weak_real_scalar_preserves_complex_tensor_dtype() ! {
	values := vtl.from_1d[complex.Complex]([
		complex.complex(1.0, 2.0),
		complex.complex(-3.0, 4.0),
	])!
	added := vtl.add_scalar_promoted[complex.Complex, complex.Complex, f64](values, 0.5)!
	assert added.dtype() == .complex128
	assert added.get_nth(0).re == 1.5
	assert added.get_nth(0).im == 2.0
	assert added.get_nth(1).re == -2.5
	assert added.get_nth(1).im == 4.0

	multiplied := vtl.multiply_scalar_promoted[complex.Complex, complex.Complex, i32](values,
		2)!
	assert multiplied.get_nth(0).re == 2.0
	assert multiplied.get_nth(0).im == 4.0
	assert multiplied.get_nth(1).re == -6.0
	assert multiplied.get_nth(1).im == 8.0
}

fn test_weak_complex_scalar_promotes_real_tensor_to_complex128() ! {
	values := vtl.from_1d[f32]([1.5, -2.0])!
	scalar := complex.complex(0.25, 0.5)
	added := vtl.add_scalar_promoted[complex.Complex, f32, complex.Complex](values, scalar)!
	assert added.dtype() == .complex128
	assert added.get_nth(0).re == 1.75
	assert added.get_nth(0).im == 0.5
	assert added.get_nth(1).re == -1.75
	assert added.get_nth(1).im == 0.5

	subtracted := vtl.subtract_scalar_promoted[complex.Complex, f32, complex.Complex](values,
		scalar)!
	assert subtracted.get_nth(0).re == 1.25
	assert subtracted.get_nth(0).im == -0.5
	assert subtracted.get_nth(1).re == -2.25
	assert subtracted.get_nth(1).im == -0.5

	integers := vtl.from_1d[i8]([2, -3])!
	divided := vtl.divide_scalar_promoted[complex.Complex, i8, complex.Complex](integers,
		complex.complex(0.0, 2.0))!
	assert divided.get_nth(0).re == 0.0
	assert divided.get_nth(0).im == -1.0
	assert divided.get_nth(1).re == 0.0
	assert divided.get_nth(1).im == 1.5

	booleans := vtl.from_1d[bool]([true, false])!
	boolean_values := vtl.add_scalar_promoted[complex.Complex, bool, complex.Complex](booleans,
		complex.complex(0.0, 1.0))!
	assert boolean_values.get_nth(0).re == 1.0
	assert boolean_values.get_nth(0).im == 1.0
	assert boolean_values.get_nth(1).re == 0.0
	assert boolean_values.get_nth(1).im == 1.0
}

fn test_weak_complex_scalar_promotion_rejects_remainder() ! {
	values := vtl.from_1d[complex.Complex]([complex.complex(1.0, 2.0)])!
	if _ := vtl.remainder_scalar_promoted[complex.Complex, complex.Complex, f64](values, 2.0) {
		assert false, 'complex remainder must be rejected'
	} else {
		assert err.msg().contains('remainder')
	}
}

fn test_weak_scalar_extrema_promote_and_preserve_operand_order() ! {
	values := vtl.from_1d[f32]([1.5, -2.0])!
	maximum := vtl.maximum_scalar_promoted[f32, f32, f64](values, 0.25)!
	minimum := vtl.minimum_scalar_left_promoted[f32, f32, f64](0.25, values)!
	assert maximum.dtype() == .float32
	assert maximum.to_array() == [f32(1.5), 0.25]
	assert minimum.dtype() == .float32
	assert minimum.to_array() == [f32(0.25), -2.0]

	integers := vtl.from_1d[i8]([2, -3])!
	integer_maximum := vtl.maximum_scalar_promoted[i8, i8, int](integers, 1)!
	float_minimum := vtl.minimum_scalar_promoted[f64, i8, f64](integers, 0.5)!
	assert integer_maximum.dtype() == .int8
	assert integer_maximum.to_array() == [i8(2), 1]
	assert float_minimum.dtype() == .float64
	assert float_minimum.to_array() == [0.5, -3.0]

	if _ := vtl.maximum_scalar_promoted[i8, i8, int](integers, 128) {
		assert false, 'out-of-range weak integer scalar must fail'
	} else {
		assert err.msg().contains('out of range')
	}
	if _ := vtl.maximum_scalar_promoted[i16, i8, int](integers, 1) {
		assert false, 'weak extrema must reject an incorrect output dtype'
	} else {
		assert err.msg().contains('does not match')
	}
}

fn test_weak_scalar_extrema_follow_nan_and_signed_zero_rules() ! {
	nan := math.nan()
	values := vtl.from_1d[f64]([nan, 1.0, nan])!
	maximum := vtl.maximum_scalar_promoted[f64, f64, f64](values, 2.0)!
	fmaximum := vtl.fmax_scalar_promoted[f64, f64, f64](values, 2.0)!
	fminimum := vtl.fmin_scalar_left_promoted[f64, f64, f64](2.0, values)!
	assert math.is_nan(maximum.get_nth(0))
	assert maximum.get_nth(1) == 2.0
	assert math.is_nan(maximum.get_nth(2))
	assert fmaximum.to_array() == [2.0, 2.0, 2.0]
	assert fminimum.to_array() == [2.0, 1.0, 2.0]

	zeros := vtl.from_1d[f64]([0.0, -0.0])!
	maximum_right := vtl.maximum_scalar_promoted[f64, f64, f64](zeros, -0.0)!
	maximum_left := vtl.maximum_scalar_left_promoted[f64, f64, f64](-0.0, zeros)!
	fmaximum_right := vtl.fmax_scalar_promoted[f64, f64, f64](zeros, -0.0)!
	fminimum_left := vtl.fmin_scalar_left_promoted[f64, f64, f64](-0.0, zeros)!
	assert math.signbit(maximum_right.get_nth(0))
	assert math.signbit(maximum_right.get_nth(1))
	assert !math.signbit(maximum_left.get_nth(0))
	assert math.signbit(maximum_left.get_nth(1))
	assert !math.signbit(fmaximum_right.get_nth(0))
	assert math.signbit(fmaximum_right.get_nth(1))
	assert math.signbit(fminimum_left.get_nth(0))
	assert math.signbit(fminimum_left.get_nth(1))
}

fn test_weak_scalar_extrema_support_boolean_and_complex_values() ! {
	booleans := vtl.from_1d[bool]([true, false])!
	boolean_maximum := vtl.maximum_scalar_promoted[bool, bool, bool](booleans, false)!
	mixed_maximum := vtl.maximum_scalar_promoted[int, bool, int](booleans, 1)!
	assert boolean_maximum.dtype() == .boolean
	assert boolean_maximum.to_array() == [true, false]
	assert mixed_maximum.dtype() == .native_int
	assert mixed_maximum.to_array() == [1, 1]

	complex_values := vtl.from_1d[complex.Complex]([
		complex.complex(1.0, 2.0),
		complex.complex(math.nan(), 0.0),
	])!
	complex_maximum := vtl.maximum_scalar_promoted[complex.Complex, complex.Complex, int](complex_values, 2)!
	complex_fmaximum := vtl.fmax_scalar_left_promoted[complex.Complex, complex.Complex, int](2,
		complex_values)!
	assert complex_maximum.dtype() == .complex128
	assert complex_maximum.get_nth(0) == complex.complex(2.0, 0.0)
	assert math.is_nan(complex_maximum.get_nth(1).re)
	assert complex_fmaximum.get_nth(0) == complex.complex(2.0, 0.0)
	assert complex_fmaximum.get_nth(1) == complex.complex(2.0, 0.0)

	strings := vtl.from_1d(['left'])!
	if _ := vtl.maximum_scalar_promoted[string, string, string](strings, 'right') {
		assert false, 'promoted extrema must reject string tensors'
	} else {
		assert err.msg().contains('numeric')
	}
}

fn test_weak_scalar_extrema_handle_strided_and_empty_tensors() ! {
	matrix := vtl.from_2d[f32]([[1, 2, 3], [4, 5, 6]])!
	transposed := matrix.transpose([1, 0])!
	result := vtl.fmin_scalar_promoted[f32, f32, f64](transposed, 4.5)!
	assert result.shape == [3, 2]
	assert result.to_array() == [f32(1.0), 4.0, 2.0, 4.5, 3.0, 4.5]

	empty := vtl.from_array[f32]([], [0, 1])!
	empty_result := vtl.fmax_scalar_left_promoted[f32, f32, f64](1.0, empty)!
	assert empty_result.shape == [0, 1]
	assert empty_result.size == 0
}
