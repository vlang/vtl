module core

import math
import vtl

fn test_add() {
	a := vtl.from_1d([1, 2, 3, 4])!
	b := vtl.from_1d([0, 1, 2, 3])!
	result := a.add(b)!
	expected := vtl.from_1d([1, 3, 5, 7])!
	assert result.array_equal(expected)
}

fn test_add_scalar() {
	a := vtl.from_1d([1, 2, 3, 4])!
	result := a.add_scalar(1)!
	expected := vtl.from_1d([2, 3, 4, 5])!
	assert result.array_equal(expected)
}

fn test_add_2d() {
	a := vtl.from_2d([[1, 2, 3, 4], [5, 6, 7, 8]])!
	b := vtl.from_2d([[0, 1, 2, 3], [4, 5, 6, 7]])!
	result := a.add(b)!
	expected := vtl.from_2d([[1, 3, 5, 7], [9, 11, 13, 15]])!
	assert result.array_equal(expected)
}

fn test_add_2d_scalar() {
	a := vtl.from_2d([[1, 2, 3, 4], [5, 6, 7, 8]])!
	result := a.add_scalar(1)!
	expected := vtl.from_2d([[2, 3, 4, 5], [6, 7, 8, 9]])!
	assert result.array_equal(expected)
}

fn test_subtract() {
	a := vtl.from_1d([1, 2, 3, 4])!
	b := vtl.from_1d([0, 1, 2, 3])!
	result := a.subtract(b)!
	expected := vtl.from_1d([1, 1, 1, 1])!
	assert result.array_equal(expected)
}

fn test_subtract_scalar() {
	a := vtl.from_1d([1, 2, 3, 4])!
	result := a.subtract_scalar(1)!
	expected := vtl.from_1d([0, 1, 2, 3])!
	assert result.array_equal(expected)
}

fn test_subtract_2d() {
	a := vtl.from_2d([[1, 2, 3, 4], [5, 6, 7, 8]])!
	b := vtl.from_2d([[0, 1, 2, 3], [4, 5, 6, 7]])!
	result := a.subtract(b)!
	expected := vtl.from_2d([[1, 1, 1, 1], [1, 1, 1, 1]])!
	assert result.array_equal(expected)
}

fn test_subtract_2d_scalar() {
	a := vtl.from_2d([[1, 2, 3, 4], [5, 6, 7, 8]])!
	result := a.subtract_scalar(1)!
	expected := vtl.from_2d([[0, 1, 2, 3], [4, 5, 6, 7]])!
	assert result.array_equal(expected)
}

fn test_multiply() {
	a := vtl.from_1d([1, 2, 3, 4])!
	b := vtl.from_1d([0, 1, 2, 3])!
	result := a.multiply(b)!
	expected := vtl.from_1d([0, 2, 6, 12])!
	assert result.array_equal(expected)
}

fn test_multiply_scalar() {
	a := vtl.from_1d([1, 2, 3, 4])!
	result := a.multiply_scalar(2)!
	expected := vtl.from_1d([2, 4, 6, 8])!
	assert result.array_equal(expected)
}

fn test_multiply_2d() {
	a := vtl.from_2d([[1, 2, 3, 4], [5, 6, 7, 8]])!
	b := vtl.from_2d([[0, 1, 2, 3], [4, 5, 6, 7]])!
	result := a.multiply(b)!
	expected := vtl.from_2d([[0, 2, 6, 12], [20, 30, 42, 56]])!
	assert result.array_equal(expected)
}

fn test_multiply_2d_scalar() {
	a := vtl.from_2d([[1, 2, 3, 4], [5, 6, 7, 8]])!
	result := a.multiply_scalar(2)!
	expected := vtl.from_2d([[2, 4, 6, 8], [10, 12, 14, 16]])!
	assert result.array_equal(expected)
}

fn test_divide() {
	a := vtl.from_1d([1, 2, 3, 4])!
	b := vtl.from_1d([1, 2, 3, 4])!
	result := a.divide(b)!
	expected := vtl.from_1d([1, 1, 1, 1])!
	assert result.array_equal(expected)
}

fn test_divide_scalar() {
	a := vtl.from_1d([1.0, 2, 3, 4])!
	result := a.divide_scalar(2.0)!
	expected := vtl.from_1d([0.5, 1, 1.5, 2])!
	assert result.array_equal(expected)
}

fn test_divide_2d() {
	a := vtl.from_2d([[1, 2, 3, 4], [5, 6, 7, 8]])!
	b := vtl.from_2d([[1, 2, 3, 4], [5, 6, 7, 8]])!
	result := a.divide(b)!
	expected := vtl.from_2d([[1, 1, 1, 1], [1, 1, 1, 1]])!
	assert result.array_equal(expected)
}

fn test_divide_2d_scalar() {
	a := vtl.from_2d([[1.0, 2, 3, 4], [5.0, 6, 7, 8]])!
	result := a.divide_scalar(2.0)!
	expected := vtl.from_2d([[0.5, 1, 1.5, 2], [2.5, 3, 3.5, 4]])!
	assert result.array_equal(expected)
}

fn test_scalar_elementwise_operations_on_noncontiguous_views() {
	a := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	transposed := a.transpose([1, 0])!
	result := transposed.add_scalar(10)!
	expected := vtl.from_2d([[11, 14], [12, 15], [13, 16]])!
	assert result.array_equal(expected)
}

fn test_binary_elementwise_operation_broadcast_fallback() {
	a := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	b := vtl.from_1d([10, 20, 30])!
	result := a.add(b)!
	expected := vtl.from_2d([[11, 22, 33], [14, 25, 36]])!
	assert result.array_equal(expected)
}

fn test_floor_divide_signed_integers_rounds_toward_negative_infinity() ! {
	numerators := vtl.from_1d([-3, -3, 3, 3])!
	divisors := vtl.from_1d([2, -2, 2, -2])!

	result := numerators.floor_divide(divisors)!

	assert result.to_array() == [-2, 1, 1, -2]
	assert result.dtype() == .native_int
}

fn test_floor_divide_preserves_float_dtype_and_ieee_results() ! {
	numerators := vtl.from_1d[f64]([-3.5, 3.5, -0.0, 1.0, -1.0, 0.0, math.nan()])!
	divisors := vtl.from_1d[f64]([2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0])!

	result := numerators.floor_divide(divisors)!

	assert result.to_array()[0..2] == [-2.0, 1.0]
	assert result.to_array()[2] == 0.0
	assert math.signbit(result.to_array()[2])
	assert math.is_inf(result.to_array()[3], 1)
	assert math.is_inf(result.to_array()[4], -1)
	assert math.is_nan(result.to_array()[5])
	assert math.is_nan(result.to_array()[6])
	assert result.dtype() == .float64
}

fn test_floor_divide_scalar_and_noncontiguous_broadcast() ! {
	float_values := vtl.from_1d[f32]([-3.5, 3.5])!
	scalar_result := float_values.floor_divide_scalar(f32(2.0))!
	assert scalar_result.to_array() == [f32(-2.0), f32(1.0)]
	assert scalar_result.dtype() == .float32

	matrix := vtl.from_2d([[-3, 1], [-1, 3]])!
	transposed := matrix.transpose([1, 0])!
	divisors := vtl.from_1d([2, -2])!
	broadcast_result := transposed.floor_divide(divisors)!
	expected := vtl.from_2d([[-2, 0], [0, -2]])!
	assert broadcast_result.array_equal(expected)
}

fn test_floor_divide_integer_zero_and_signed_overflow_match_numpy_values() ! {
	numerators := vtl.from_1d([1, -3])!
	zero_divisors := vtl.from_1d([0, 0])!
	assert numerators.floor_divide(zero_divisors)!.to_array() == [0, 0]

	min_i8 := i8(-128)
	assert vtl.from_1d([min_i8])!.floor_divide(vtl.from_1d([i8(-1)])!)!.to_array() == [
		min_i8,
	]
	min_i16 := i16(-32768)
	assert vtl.from_1d([min_i16])!.floor_divide(vtl.from_1d([i16(-1)])!)!.to_array() == [
		min_i16,
	]
	min_i32 := i32(-2147483647 - 1)
	assert vtl.from_1d([min_i32])!.floor_divide(vtl.from_1d([i32(-1)])!)!.to_array() == [
		min_i32,
	]
	min_i64 := i64(-9223372036854775807 - 1)
	min_tensor := vtl.from_1d([min_i64])!
	min_result := min_tensor.floor_divide(vtl.from_1d([i64(-1)])!)!
	assert min_result.to_array() == [min_i64]
}

fn test_floor_divide_rounds_fixed_width_signed_integers() ! {
	assert vtl.from_1d[i8]([-3, 3])!.floor_divide(vtl.from_1d[i8]([2, -2])!)!.to_array() == [
		i8(-2),
		-2,
	]
	assert vtl.from_1d[i16]([-3, 3])!.floor_divide(vtl.from_1d[i16]([2, -2])!)!.to_array() == [
		i16(-2),
		-2,
	]
	assert vtl.from_1d[i32]([-3, 3])!.floor_divide(vtl.from_1d[i32]([2, -2])!)!.to_array() == [
		i32(-2),
		-2,
	]
	assert vtl.from_1d[i64]([-3, 3])!.floor_divide(vtl.from_1d[i64]([2, -2])!)!.to_array() == [
		i64(-2),
		-2,
	]
}

fn test_floor_divide_supports_unsigned_integer_widths() ! {
	assert vtl.from_1d[u8]([5, 3])!.floor_divide(vtl.from_1d[u8]([2, 2])!)!.to_array() == [
		u8(2),
		1,
	]
	assert vtl.from_1d[u16]([5, 3])!.floor_divide(vtl.from_1d[u16]([2, 2])!)!.to_array() == [
		u16(2),
		1,
	]
	assert vtl.from_1d[u32]([5, 3])!.floor_divide(vtl.from_1d[u32]([2, 2])!)!.to_array() == [
		u32(2),
		1,
	]
	assert vtl.from_1d[u64]([5, 3])!.floor_divide(vtl.from_1d[u64]([2, 2])!)!.to_array() == [
		u64(2),
		1,
	]
	assert vtl.from_1d[u8]([5])!.floor_divide(vtl.from_1d[u8]([0])!)!.to_array() == [u8(0)]
}

fn test_floor_divide_empty_result_and_unsupported_dtype() ! {
	empty := vtl.empty[int]([0])
	result := empty.floor_divide(vtl.from_1d([0])!)!
	assert result.shape == [0]
	assert result.size == 0

	booleans := vtl.from_1d([true, false])!
	booleans.floor_divide(booleans) or {
		assert err.msg().contains('real integer')
		return
	}
	assert false, 'floor_divide should reject boolean tensors'
}
