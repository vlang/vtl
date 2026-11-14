module core

import vtl
import math

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
