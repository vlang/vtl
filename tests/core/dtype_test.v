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

fn test_complex_transcendental_functions() ! {
	values := vtl.from_1d([
		vcomplex.Complex{ re: 0.0, im: 0.0 },
		vcomplex.Complex{ re: -4.0, im: 0.0 },
		vcomplex.Complex{ re: 3.0, im: 4.0 },
	])!

	assert vtl.exp(values)!.get_nth(0) == vcomplex.Complex{ re: 1.0, im: 0.0 }
	assert math.abs(vtl.log(values)!.get_nth(2).re - 1.6094379124341003) < 1e-15
	assert vtl.sqrt(values)!.to_array() == [
		vcomplex.Complex{ re: 0.0, im: 0.0 },
		vcomplex.Complex{ re: 0.0, im: 2.0 },
		vcomplex.Complex{ re: 2.0, im: 1.0 },
	]
	assert vtl.sin(values)!.get_nth(0) == vcomplex.Complex{ re: 0.0, im: 0.0 }
	assert vtl.cos(values)!.get_nth(0) == vcomplex.Complex{ re: 1.0, im: 0.0 }
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
