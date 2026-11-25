module core

import math
import math.complex
import vtl

fn test_float_power_promoted_broadcasts_and_allows_negative_integer_exponents() ! {
	bases := vtl.from_array[i16]([2, -2], [2, 1])!
	exponents := vtl.from_1d[i8]([-1, 0, 2])!

	result := vtl.float_power_promoted(bases, exponents)!
	expected := vtl.from_array[f64]([0.5, 1.0, 4.0, -0.5, 1.0, 4.0], [2, 3])!

	assert result.dtype() == .float64
	assert result.shape == [2, 3]
	assert result.array_equal(expected)
}

fn test_float_power_promoted_always_returns_f64_for_real_and_boolean_inputs() ! {
	f32_bases := vtl.from_1d[f32]([4.0, 9.0])!
	f32_exponents := vtl.from_1d[f32]([0.5, 0.5])!
	f32_result := vtl.float_power_promoted(f32_bases, f32_exponents)!
	assert f32_result.dtype() == .float64
	assert f32_result.to_array() == [2.0, 3.0]

	boolean_bases := vtl.from_1d([false, true])!
	boolean_exponents := vtl.from_1d[i8]([2, -1])!
	boolean_result := vtl.float_power_promoted(boolean_bases, boolean_exponents)!
	assert boolean_result.dtype() == .float64
	assert boolean_result.to_array() == [0.0, 1.0]

	large_integer := vtl.from_1d[i64]([9_007_199_254_740_993])!
	unit_exponent := vtl.from_1d[i8]([1])!
	assert vtl.float_power_promoted(large_integer, unit_exponent)!.to_array() == [
		9_007_199_254_740_992.0,
	]
}

fn test_float_power_promoted_matches_numpy_ieee_edge_cases() ! {
	bases := vtl.from_1d[f64]([-0.0, -0.0, 0.0, -2.0, 0.0, 1.0, math.inf(1)])!
	exponents := vtl.from_1d[f64]([3.0, -3.0, -1.0, 0.5, 0.0, math.nan(), math.inf(-1)])!

	result := vtl.float_power_promoted(bases, exponents)!
	assert result.get_nth(0) == 0.0 && math.signbit(result.get_nth(0))
	assert math.is_inf(result.get_nth(1), -1) && math.signbit(result.get_nth(1))
	assert math.is_inf(result.get_nth(2), 1) && !math.signbit(result.get_nth(2))
	assert math.is_nan(result.get_nth(3))
	assert result.get_nth(4) == 1.0
	assert result.get_nth(5) == 1.0
	assert result.get_nth(6) == 0.0
}

fn test_float_power_promoted_handles_scalar_and_empty_broadcasts_and_rejects_bad_inputs() ! {
	scalar_base := vtl.from_array[f32]([2.0], [])!
	scalar_exponent := vtl.from_array[i8]([-1], [])!
	scalar_result := vtl.float_power_promoted(scalar_base, scalar_exponent)!
	assert scalar_result.shape.len == 0
	assert scalar_result.get_nth(0) == 0.5

	empty := vtl.empty[i16]([0, 1])
	row := vtl.from_array[i8]([1, 2], [1, 2])!
	empty_result := vtl.float_power_promoted(empty, row)!
	assert empty_result.shape == [0, 2]
	assert empty_result.size == 0

	left := vtl.from_1d[i16]([2, 3])!
	right := vtl.from_1d[i8]([1, 2, 3])!
	if _ := vtl.float_power_promoted(left, right) {
		assert false, 'float_power must reject incompatible broadcast shapes'
	}

	complex_values := vtl.from_1d[complex.Complex]([complex.complex(2.0, 1.0)])!
	if _ := vtl.float_power_promoted(complex_values, complex_values) {
		assert false, 'float_power must reject complex inputs'
	}

	string_values := vtl.from_1d(['2'])!
	if _ := vtl.float_power_promoted(string_values, string_values) {
		assert false, 'float_power must reject string inputs'
	}
}
