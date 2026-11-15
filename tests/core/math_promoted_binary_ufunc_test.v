module core

import math
import vtl

fn test_promoted_hypot_broadcasts_mixed_inputs() ! {
	a := vtl.from_2d[i32]([[3], [5]])!
	b := vtl.from_1d[f32]([4.0, 12.0, 0.0])!
	values := vtl.hypot_promoted[f64, i32, f32](a, b)!
	expected := vtl.from_2d[f64]([
		[5.0, math.sqrt(153.0), 3.0],
		[math.sqrt(41.0), 13.0, 5.0],
	])!
	assert values.shape == [2, 3]
	assert values.dtype() == .float64
	assert values.allclose(expected, rtol: 1e-14, atol: 1e-14)!
}

fn test_promoted_logaddexp_variants_are_stable_and_follow_numpy_dtype_rules() ! {
	left := vtl.from_1d[i16]([1000, -1000])!
	right := vtl.from_1d[f32]([1000.0, -1000.0])!
	values := vtl.logaddexp_promoted[f32, i16, f32](left, right)!
	expected := vtl.from_1d[f32]([
		f32(1000.0 + math.log(2.0)),
		f32(-1000.0 + math.log(2.0)),
	])!
	assert values.dtype() == .float32
	assert values.allclose(expected, rtol: 1e-6, atol: 1e-5)!

	log_inputs := vtl.from_2d[f64]([[0.0, 1.0]])!
	bool_input := vtl.from_1d([true])!
	base_two := vtl.logaddexp2_promoted[f64, f64, bool](log_inputs, bool_input)!
	expected_base_two := vtl.from_2d[f64]([[math.log2(3.0), 2.0]])!
	assert base_two.allclose(expected_base_two, rtol: 1e-14, atol: 1e-14)!

	special_left := vtl.from_1d[f64]([math.inf(1), math.nan()])!
	special_right := vtl.from_1d[f64]([math.inf(1), 0.0])!
	special := vtl.logaddexp_promoted[f64, f64, f64](special_left, special_right)!
	assert math.is_inf(special.get_nth(0), 1)
	assert math.is_nan(special.get_nth(1))
}

fn test_promoted_copysign_broadcasts_mixed_dtypes() ! {
	magnitude := vtl.from_2d[i16]([[-1, 2], [-3, 4]])!
	sign := vtl.from_1d[f32]([-0.0, math.inf(1)])!

	result := vtl.copysign_promoted[f32, i16, f32](magnitude, sign)!
	expected := vtl.from_2d[f32]([[-1.0, 2.0], [-3.0, 4.0]])!

	assert result.dtype() == .float32
	assert result.shape == [2, 2]
	assert result.allclose(expected, rtol: 0.0, atol: 0.0)!
}

fn test_promoted_copysign_preserves_ieee_sign_bits() ! {
	magnitude := vtl.from_1d[f64]([0.0, 0.0, math.nan(), math.inf(1)])!
	sign := vtl.from_1d[f64]([-0.0, 0.0, -math.inf(1), -1.0])!
	result := vtl.copysign_promoted[f64, f64, f64](magnitude, sign)!

	assert math.signbit(result.get_nth(0))
	assert !math.signbit(result.get_nth(1))
	assert math.is_nan(result.get_nth(2))
	assert math.signbit(result.get_nth(2))
	assert math.is_inf(result.get_nth(3), -1)
}

fn test_promoted_nextafter_broadcasts_at_float32_precision() ! {
	x := vtl.from_2d[i16]([[0, 1]])!
	y := vtl.from_1d[f32]([1.0, -math.inf(1)])!

	result := vtl.nextafter_promoted[f32, i16, f32](x, y)!

	assert result.dtype() == .float32
	assert result.shape == [1, 2]
	assert result.get_nth(0) == math.nextafter32(f32(0.0), f32(1.0))
	assert result.get_nth(1) == math.nextafter32(f32(1.0), -f32(math.inf(1)))
}

fn test_promoted_nextafter_handles_signed_zero_infinity_and_nan() ! {
	x := vtl.from_1d[f64]([-0.0, 0.0, math.inf(1), math.nan()])!
	y := vtl.from_1d[f64]([1.0, -1.0, 0.0, 0.0])!
	result := vtl.nextafter_promoted[f64, f64, f64](x, y)!

	assert result.get_nth(0) == math.nextafter(-0.0, 1.0)
	assert !math.signbit(result.get_nth(0))
	assert result.get_nth(1) == math.nextafter(0.0, -1.0)
	assert math.signbit(result.get_nth(1))
	assert result.get_nth(2) == math.nextafter(math.inf(1), 0.0)
	assert math.is_nan(result.get_nth(3))
}

fn test_promoted_binary_ufuncs_validate_output_dtype() ! {
	left := vtl.from_1d[i32]([1])!
	right := vtl.from_1d[i16]([2])!
	if _ := vtl.hypot_promoted[f32, i32, i16](left, right) {
		assert false, '32-bit integer inputs must produce float64'
	}
	if _ := vtl.copysign_promoted[f32, i32, i16](left, right) {
		assert false, '32-bit integer inputs must produce float64'
	}
	if _ := vtl.nextafter_promoted[f32, i32, i16](left, right) {
		assert false, '32-bit integer inputs must produce float64'
	}
}
