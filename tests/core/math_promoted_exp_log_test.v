module core

import math
import vtl

fn test_exp_promoted_selects_result_dtype_and_values() ! {
	small_integers := vtl.from_1d[i16]([0, 1])!
	small_exponents := vtl.exp_promoted[f32, i16](small_integers)!
	expected_small := vtl.from_1d[f32]([1.0, f32(math.e)])!
	assert small_exponents.dtype() == .float32
	assert small_exponents.allclose(expected_small, rtol: 1e-6, atol: 1e-6)!

	wide_integers := vtl.from_1d[i32]([0, 1])!
	wide_exponents := vtl.exp_promoted[f64, i32](wide_integers)!
	expected_wide := vtl.from_1d[f64]([1.0, math.e])!
	assert wide_exponents.dtype() == .float64
	assert wide_exponents.allclose(expected_wide, rtol: 1e-14, atol: 1e-14)!
}

fn test_log_promoted_preserves_ieee_domain_results() ! {
	values := vtl.from_1d[f32]([1.0, 0.0, -1.0])!
	logarithms := vtl.log_promoted[f32, f32](values)!
	results := logarithms.to_array()
	assert results[0] == 0.0
	assert math.is_inf(results[1], -1)
	assert math.is_nan(results[2])
}

fn test_promoted_base_log_and_exponential_variants() ! {
	exponents := vtl.from_1d[i16]([-1, 0, 1])!
	powers := vtl.exp2_promoted[f32, i16](exponents)!
	assert powers.dtype() == .float32
	assert powers.get_nth(0) == f32(0.5)
	assert powers.get_nth(1) == f32(1.0)
	assert powers.get_nth(2) == f32(2.0)

	small_values := vtl.from_1d[f64]([1e-8, -1e-8])!
	exponential_differences := vtl.expm1_promoted[f64, f64](small_values)!
	logarithmic_differences := vtl.log1p_promoted[f64, f64](small_values)!
	assert math.abs(exponential_differences.get_nth(0) - 1e-8) < 1e-16
	assert math.abs(exponential_differences.get_nth(1) + 1e-8) < 1e-16
	assert math.abs(logarithmic_differences.get_nth(0) - 1e-8) < 1e-16
	assert math.abs(logarithmic_differences.get_nth(1) + 1e-8) < 1e-16

	log_inputs := vtl.from_1d[i32]([1, 2, 10, 0])!
	log2_values := vtl.log2_promoted[f64, i32](log_inputs)!
	log10_values := vtl.log10_promoted[f64, i32](log_inputs)!
	assert log2_values.dtype() == .float64
	assert log10_values.dtype() == .float64
	assert log2_values.get_nth(0) == 0.0
	assert log2_values.get_nth(1) == 1.0
	assert math.is_inf(log2_values.get_nth(3), -1)
	assert math.abs(log10_values.get_nth(2) - 1.0) < 1e-14
	assert math.is_inf(log10_values.get_nth(3), -1)
}
