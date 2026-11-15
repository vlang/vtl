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
