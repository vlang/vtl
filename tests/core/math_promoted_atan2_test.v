module core

import math
import math.complex
import vtl

fn test_atan2_promoted_broadcasts_mixed_dtypes() ! {
	y := vtl.from_array[i16]([0, 1, 2, 3], [2, 2])!
	x := vtl.from_1d[f64]([1.0, 1.0])!

	result := vtl.atan2_promoted[f64, i16, f64](y, x)!
	expected := vtl.from_array[f64]([0.0, math.atan2(1.0, 1.0), math.atan2(2.0, 1.0),
		math.atan2(3.0, 1.0)], [2, 2])!

	assert result.dtype() == .float64
	assert result.shape == [2, 2]
	assert result.allclose(expected, rtol: 1e-14, atol: 1e-14)!
}

fn test_atan2_promoted_retains_float32_for_small_integer_mix() ! {
	y := vtl.from_1d[f32]([1.0, 0.0])!
	x := vtl.from_1d[i8]([1, -1])!

	result := vtl.atan2_promoted[f32, f32, i8](y, x)!
	expected := vtl.from_1d[f32]([f32(math.pi / 4.0), f32(math.pi)])!

	assert result.dtype() == .float32
	assert result.allclose(expected, rtol: 1e-6, atol: 1e-6)!
}

fn test_atan2_promoted_uses_supported_float_width_for_boolean_and_integer_inputs() ! {
	booleans := vtl.from_1d([true, false])!
	boolean_angles := vtl.atan2_promoted[f32, bool, bool](booleans, booleans)!
	assert boolean_angles.dtype() == .float32
	expected_boolean := vtl.from_1d[f32]([f32(math.pi / 4.0), 0.0])!
	assert boolean_angles.allclose(expected_boolean, rtol: 1e-6, atol: 1e-6)!

	small_integers := vtl.from_1d[i8]([1, -1])!
	small_integer_angles := vtl.atan2_promoted[f32, i8, i8](small_integers,
		small_integers)!
	assert small_integer_angles.dtype() == .float32
	expected_small_integers := vtl.from_1d[f32]([f32(math.pi / 4.0), f32(-3.0 * math.pi / 4.0)])!
	assert small_integer_angles.allclose(expected_small_integers, rtol: 1e-6, atol: 1e-6)!

	signed := vtl.from_1d[i8]([1, -1])!
	unsigned := vtl.from_1d[u16]([1, 1])!
	mixed_small_integers := vtl.atan2_promoted[f32, i8, u16](signed, unsigned)!
	assert mixed_small_integers.dtype() == .float32

	integers := vtl.from_1d[i32]([1, -1])!
	integer_angles := vtl.atan2_promoted[f64, i32, i32](integers, integers)!
	assert integer_angles.dtype() == .float64
	expected_integer := vtl.from_1d[f64]([math.pi / 4.0, -3.0 * math.pi / 4.0])!
	assert integer_angles.allclose(expected_integer, rtol: 0.0, atol: 1e-15)!
}

fn test_atan2_promoted_matches_ieee_special_values() ! {
	y := vtl.from_1d[f64]([-0.0, 0.0, math.inf(1), math.nan()])!
	x := vtl.from_1d[f64]([-0.0, -0.0, math.inf(1), 1.0])!

	result := vtl.atan2_promoted[f64, f64, f64](y, x)!
	values := result.to_array()

	assert values[0] == -math.pi
	assert values[1] == math.pi
	assert math.abs(values[2] - math.pi / 4.0) < 1e-15
	assert math.is_nan(values[3])
}

fn test_atan2_promoted_preserves_empty_broadcast_shape() ! {
	y := vtl.empty[i32]([2, 0])
	x := vtl.from_1d[f64]([1.0])!

	result := vtl.atan2_promoted[f64, i32, f64](y, x)!

	assert result.dtype() == .float64
	assert result.shape == [2, 0]
	assert result.size == 0
}

fn test_atan2_promoted_rejects_unsupported_inputs_and_result_dtype() {
	integers := vtl.from_1d([1, 2])!
	if _ := vtl.atan2_promoted[f32, int, int](integers, integers) {
		assert false, 'integer atan2 must produce float64'
	}

	complex_values := vtl.from_1d([complex.complex(1.0, 2.0)])!
	if _ := vtl.atan2_promoted[f64, complex.Complex, complex.Complex](complex_values, complex_values) {
		assert false, 'complex atan2 is unsupported'
	}

	strings := vtl.from_1d(['1'])!
	if _ := vtl.atan2_promoted[f64, string, string](strings, strings) {
		assert false, 'string atan2 is unsupported'
	}

	incompatible := vtl.from_array[f64]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])!
	if _ := vtl.atan2_promoted[f64, int, f64](integers, incompatible) {
		assert false, 'incompatible shapes must be rejected'
	}
}
