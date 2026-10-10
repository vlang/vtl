module core

import vtl
import math
import math.complex as vcomplex

fn test_dtype_of_and_tensor_dtype_report_element_types() {
	assert vtl.dtype_of[bool]() == .boolean
	assert vtl.dtype_of[i8]() == .int8
	assert vtl.dtype_of[i16]() == .int16
	assert vtl.dtype_of[i32]() == .int32
	assert vtl.dtype_of[i64]() == .int64
	assert vtl.dtype_of[int]() == .native_int
	assert vtl.dtype_of[u8]() == .uint8
	assert vtl.dtype_of[u16]() == .uint16
	assert vtl.dtype_of[u32]() == .uint32
	assert vtl.dtype_of[u64]() == .uint64
	assert vtl.dtype_of[f32]() == .float32
	assert vtl.dtype_of[f64]() == .float64
	assert vtl.dtype_of[vcomplex.Complex]() == .complex128
	assert vtl.dtype_of[string]() == .string

	tensor := vtl.from_1d([1, 2, 3])!
	assert tensor.dtype() == .native_int
	tensor32 := vtl.from_1d[i32]([1, 2, 3])!
	assert tensor32.dtype() == .int32
	assert tensor32.add(tensor32)!.to_array() == [i32(2), 4, 6]
}

fn test_complex128_tensor_creation_and_elementwise_arithmetic() ! {
	a := vcomplex.Complex{ re: 1.0, im: 2.0 }
	b := vcomplex.Complex{ re: 3.0, im: -1.0 }
	tensor := vtl.from_1d([a, b])!

	assert tensor.dtype() == .complex128
	assert tensor.shape == [2]
	assert tensor.str().contains('1.000000+2.000000i')
	assert tensor.add(tensor)!.to_array() == [vcomplex.Complex{ re: 2.0, im: 4.0 },
		vcomplex.Complex{ re: 6.0, im: -2.0 }]
	assert tensor.subtract(tensor)!.to_array() == [
		vcomplex.Complex{ re: 0.0, im: 0.0 },
		vcomplex.Complex{ re: 0.0, im: 0.0 },
	]
	assert tensor.multiply(tensor)!.to_array() == [
		vcomplex.Complex{ re: -3.0, im: 4.0 },
		vcomplex.Complex{ re: 8.0, im: -6.0 },
	]
	assert tensor.divide(tensor)!.to_array() == [vcomplex.Complex{ re: 1.0, im: 0.0 },
		vcomplex.Complex{ re: 1.0, im: 0.0 }]
}

fn test_complex_tensor_zeros_and_ones() {
	zeros := vtl.zeros[vcomplex.Complex]([2])
	ones := vtl.ones[vcomplex.Complex]([2])
	assert zeros.to_array() == [vcomplex.Complex{}, vcomplex.Complex{}]
	assert ones.to_array() == [vcomplex.Complex{ re: 1, im: 0 }, vcomplex.Complex{ re: 1, im: 0 }]
}

fn test_complex_real_imag_conj_and_absolute() ! {
	values := vtl.from_1d([
		vcomplex.Complex{ re: 3.0, im: 4.0 },
		vcomplex.Complex{ re: -5.0, im: 12.0 },
	])!

	assert vtl.real(values)!.to_array() == [3.0, -5.0]
	assert vtl.imag(values)!.to_array() == [4.0, 12.0]
	assert vtl.conj(values)!.to_array() == [
		vcomplex.Complex{ re: 3.0, im: -4.0 },
		vcomplex.Complex{ re: -5.0, im: -12.0 },
	]
	assert vtl.absolute(values)!.to_array() == [5.0, 13.0]
	assert vtl.abs(values)!.to_array() == [5.0, 13.0]

	large := vtl.from_1d([vcomplex.Complex{ re: 1e308, im: 1e308 }])!
	assert vtl.absolute(large)!.get_nth(0) < 1.5e308

	matrix := vtl.from_array([
		vcomplex.Complex{ re: 1.0, im: 2.0 },
		vcomplex.Complex{ re: 3.0, im: 4.0 },
		vcomplex.Complex{ re: 5.0, im: 6.0 },
		vcomplex.Complex{ re: 7.0, im: 8.0 },
	], [2, 2])!
	transposed := matrix.transpose([1, 0])!
	assert vtl.real(transposed)!.to_array() == [1.0, 5.0, 3.0, 7.0]
	assert vtl.imag(transposed)!.to_array() == [2.0, 6.0, 4.0, 8.0]
}

fn test_complex_angle_and_finite_value_predicates() ! {
	values := vtl.from_1d([
		vcomplex.Complex{ re: 1.0, im: 0.0 },
		vcomplex.Complex{ re: 0.0, im: 1.0 },
		vcomplex.Complex{ re: -1.0, im: 0.0 },
		vcomplex.Complex{ re: math.nan(), im: 2.0 },
		vcomplex.Complex{ re: math.inf(1), im: math.nan() },
	])!
	angles := vtl.complex_angle(values, false)!
	assert angles.get_nth(0) == 0.0
	assert angles.get_nth(1) == math.pi / 2.0
	assert angles.get_nth(2) == math.pi
	degrees := vtl.complex_angle(values, true)!
	assert degrees.to_array()[0] == 0.0
	assert degrees.to_array()[1] == 90.0
	assert degrees.to_array()[2] == 180.0
	assert vtl.complex_is_nan(values)!.to_array() == [false, false, false, true, true]
	assert vtl.complex_is_inf(values)!.to_array() == [false, false, false, false, true]
	assert vtl.complex_is_finite(values)!.to_array() == [true, true, true, false, false]
}

fn test_complex_transcendental_functions() ! {
	values := vtl.from_1d([
		vcomplex.Complex{ re: 0.0, im: 0.0 },
		vcomplex.Complex{ re: -4.0, im: 0.0 },
		vcomplex.Complex{ re: 3.0, im: 4.0 },
		vcomplex.Complex{ re: 1.0, im: 0.0 },
	])!

	assert vtl.exp(values)!.get_nth(0) == vcomplex.Complex{ re: 1.0, im: 0.0 }
	assert math.abs(vtl.log(values)!.get_nth(2).re - 1.6094379124341003) < 1e-15
	assert vtl.sqrt(values)!.to_array() == [
		vcomplex.Complex{ re: 0.0, im: 0.0 },
		vcomplex.Complex{ re: 0.0, im: 2.0 },
		vcomplex.Complex{ re: 2.0, im: 1.0 },
		vcomplex.Complex{ re: 1.0, im: 0.0 },
	]
	assert vtl.sin(values)!.get_nth(0) == vcomplex.Complex{ re: 0.0, im: 0.0 }
	assert vtl.cos(values)!.get_nth(0) == vcomplex.Complex{ re: 1.0, im: 0.0 }
	assert vtl.tan(values)!.get_nth(0) == vcomplex.Complex{ re: 0.0, im: 0.0 }
	assert vtl.sinh(values)!.get_nth(0) == vcomplex.Complex{ re: 0.0, im: 0.0 }
	assert vtl.cosh(values)!.get_nth(0) == vcomplex.Complex{ re: 1.0, im: 0.0 }
	assert vtl.tanh(values)!.get_nth(0) == vcomplex.Complex{ re: 0.0, im: 0.0 }
	assert vtl.arcsin(values)!.get_nth(0) == vcomplex.Complex{ re: 0.0, im: 0.0 }
	assert vtl.arccos(values)!.get_nth(3) == vcomplex.Complex{ re: 0.0, im: 0.0 }
	assert vtl.arctan(values)!.get_nth(0) == vcomplex.Complex{ re: 0.0, im: 0.0 }
	assert vtl.arcsinh(values)!.get_nth(0) == vcomplex.Complex{ re: 0.0, im: 0.0 }
	assert vtl.arccosh(values)!.get_nth(3) == vcomplex.Complex{ re: 0.0, im: 0.0 }
	assert vtl.arctanh(values)!.get_nth(0) == vcomplex.Complex{ re: 0.0, im: 0.0 }
}

fn test_promote_types_for_matching_numeric_kinds() {
	assert vtl.promote_types(.int8, .int16)! == .int16
	assert vtl.promote_types(.int16, .uint16)! == .int32
	assert vtl.promote_types(.uint8, .uint16)! == .uint16
	assert vtl.promote_types(.int16, .uint8)! == .int16
	assert vtl.promote_types(.uint8, .int8)! == .int16
	assert vtl.promote_types(.int32, .uint16)! == .int32
	assert vtl.promote_types(.uint32, .int64)! == .int64
	assert vtl.promote_types(.uint64, .int64)! == .float64
	assert vtl.promote_types(.float32, .uint16)! == .float32
	assert vtl.promote_types(.float32, .uint32)! == .float64
	assert vtl.promote_types(.float32, .native_int)! == .float64
	assert vtl.promote_types(.boolean, .uint8)! == .uint8
	assert vtl.promote_types(.float32, .float64)! == .float64
	assert vtl.promote_types(.complex128, .float64)! == .complex128
	assert vtl.promote_types(.int32, .complex128)! == .complex128
	assert vtl.promote_types(.boolean, .complex128)! == .complex128
	assert vtl.promote_types(.boolean, .boolean)! == .boolean
	assert vtl.promote_types(.string, .string)! == .string
}

fn test_promote_types_rejects_string_and_numeric_mixtures() {
	if _ := vtl.promote_types(.string, .int8) {
		assert false, 'promote_types must reject string/numeric mixtures'
	} else {
		assert true
	}
}

fn test_promoted_integer_arithmetic_uses_numpy_dtype_and_broadcasting() ! {
	signed := vtl.from_1d[i8]([100, -3])!
	unsigned := vtl.from_1d[u8]([60, 4])!

	sums := vtl.add_promoted[i16, i8, u8](signed, unsigned)!
	assert sums.dtype() == .int16
	assert sums.to_array() == [i16(160), 1]
	assert vtl.subtract_promoted[i16, i8, u8](signed, unsigned)!.to_array() == [
		i16(40),
		-7,
	]
	assert vtl.multiply_promoted[i16, i8, u8](signed, unsigned)!.to_array() == [
		i16(6000),
		-12,
	]

	rows := vtl.from_array[i8]([-2, 5], [2, 1])!
	columns := vtl.from_1d[u8]([1, 2, 3])!
	broadcast_sum := vtl.add_promoted[i16, i8, u8](rows, columns)!
	assert broadcast_sum.shape == [2, 3]
	assert broadcast_sum.to_array() == [i16(-1), 0, 1, 6, 7, 8]
}

fn test_promoted_arithmetic_preserves_dtype_precision_and_rejects_wrong_dtype() ! {
	floats := vtl.from_1d[f32]([1.5])!
	integers := vtl.from_1d[u32]([u32(16_777_217)])!
	result := vtl.add_promoted[f64, f32, u32](floats, integers)!
	assert result.dtype() == .float64
	assert result.to_array() == [16_777_218.5]

	left := vtl.from_1d[i8]([2])!
	right := vtl.from_1d[u8]([3])!
	if _ := vtl.add_promoted[i8, i8, u8](left, right) {
		assert false, 'promoted arithmetic must reject an incorrect output dtype'
	}
}

fn test_promoted_division_uses_numpy_true_divide_dtypes() ! {
	integers := vtl.from_2d[i8]([[1, 2], [3, 4]])!
	unsigned := vtl.from_1d[u8]([2, 4])!
	integer_quotients := vtl.divide_promoted[f64, i8, u8](integers, unsigned)!
	assert integer_quotients.shape == [2, 2]
	assert integer_quotients.dtype() == .float64
	assert integer_quotients.to_array() == [0.5, 0.5, 1.5, 1.0]

	floats := vtl.from_2d[f32]([[8, 6], [4, 2]])!
	divisors := vtl.from_1d[f32]([2, 3])!
	float_quotients := vtl.divide_promoted[f32, f32, f32](floats, divisors)!
	assert float_quotients.dtype() == .float32
	float_values := float_quotients.to_array()
	float_expected := [4.0, 2.0, 2.0, 2.0 / 3]
	for i, expected in float_expected {
		assert math.abs(float_values[i] - expected) < 1e-6
	}

	f32_values := vtl.from_1d[f32]([2, 4])!
	i16_values := vtl.from_1d[i16]([2])!
	mixed_quotients := vtl.divide_promoted[f32, f32, i16](f32_values, i16_values)!
	assert mixed_quotients.dtype() == .float32
	for i, expected in [1.0, 2.0] {
		assert math.abs(mixed_quotients.to_array()[i] - expected) < 1e-6
	}

	truth := vtl.from_1d[bool]([true, false])!
	boolean_divisors := vtl.from_1d[bool]([true, true])!
	boolean_quotients := vtl.divide_promoted[f64, bool, bool](truth, boolean_divisors)!
	assert boolean_quotients.dtype() == .float64
	assert boolean_quotients.to_array() == [1.0, 0.0]
}

fn test_promoted_division_rejects_wrong_output_dtype() ! {
	left := vtl.from_1d[i16]([5])!
	right := vtl.from_1d[i16]([2])!
	if _ := vtl.divide_promoted[i16, i16, i16](left, right) {
		assert false, 'true division must reject an integer output dtype'
	}
	if _ := vtl.divide_promoted[f32, i16, i16](left, right) {
		assert false, 'integer true division must use float64 output'
	}
}

fn test_promoted_integer_remainder_matches_numpy_sign_and_broadcasting() ! {
	values := vtl.from_2d[i8]([[-5, 5], [-5, 5]])!
	divisors := vtl.from_1d[i16]([3, -3])!
	remainders := vtl.remainder_promoted[i16, i8, i16](values, divisors)!

	assert remainders.shape == [2, 2]
	assert remainders.dtype() == .int16
	assert remainders.to_array() == [1, -1, 1, -1]
}

fn test_promoted_integer_remainder_handles_zero_divisor_and_rejects_float() ! {
	integers := vtl.from_1d[i8]([5, -5])!
	zero_divisors := vtl.from_1d[i16]([2, 0])!
	remainders := vtl.remainder_promoted[i16, i8, i16](integers, zero_divisors)!
	assert remainders.to_array() == [1, 0]

	floats := vtl.from_1d[f32]([5.0])!
	float_divisor := vtl.from_1d[f32]([2.0])!
	if _ := vtl.remainder_promoted[f32, f32, f32](floats, float_divisor) {
		assert false, 'promoted remainder only supports integer tensors'
	}
}

fn test_promoted_integer_remainder_handles_signed_minimum_modulo_negative_one() ! {
	values := vtl.from_1d[i64]([i64(-9223372036854775807 - 1)])!
	divisor := vtl.from_1d[i64]([-1])!
	remainders := vtl.remainder_promoted[i64, i64, i64](values, divisor)!

	assert remainders.to_array() == [i64(0)]
}

fn test_promoted_remainder_reads_strided_views_in_logical_order() ! {
	values := vtl.from_2d[i8]([[-5, 4], [8, -10]])!
	transposed := values.transpose([1, 0])!
	divisor := vtl.from_1d[i16]([3])!
	remainders := vtl.remainder_promoted[i16, i8, i16](transposed, divisor)!

	assert remainders.shape == [2, 2]
	assert remainders.to_array() == [1, 2, 1, 2]
}
