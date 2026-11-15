module core

import math
import vtl

fn test_promoted_inverse_trig_values_and_domain() ! {
	values := vtl.from_1d[f64]([-1.0, 0.0, 1.0])!
	asines := vtl.asin_promoted[f64, f64](values)!
	acosines := vtl.acos_promoted[f64, f64](values)!
	assert math.abs(asines.get_nth(0) + math.pi / 2.0) < 1e-14
	assert asines.get_nth(1) == 0.0
	assert math.abs(acosines.get_nth(0) - math.pi) < 1e-14
	assert acosines.get_nth(2) == 0.0

	integers := vtl.from_1d[i32]([1])!
	arctangents := vtl.atan_promoted[f64, i32](integers)!
	assert arctangents.dtype() == .float64
	assert math.abs(arctangents.get_nth(0) - math.pi / 4.0) < 1e-14

	out_of_domain := vtl.from_1d[f64]([2.0])!
	assert math.is_nan(vtl.asin_promoted[f64, f64](out_of_domain)!.get_nth(0))
}

fn test_promoted_hyperbolic_functions_values_and_nan() ! {
	values := vtl.from_1d[f32]([0.0, 1.0, math.nan()])!
	sines := vtl.sinh_promoted[f32, f32](values)!
	cosines := vtl.cosh_promoted[f32, f32](values)!
	tangents := vtl.tanh_promoted[f32, f32](values)!

	assert sines.dtype() == .float32
	assert cosines.dtype() == .float32
	assert tangents.dtype() == .float32
	assert sines.get_nth(0) == 0.0
	assert cosines.get_nth(0) == 1.0
	assert tangents.get_nth(0) == 0.0
	assert math.abs(f64(sines.get_nth(1)) - math.sinh(1.0)) < 1e-6
	assert math.abs(f64(cosines.get_nth(1)) - math.cosh(1.0)) < 1e-6
	assert math.abs(f64(tangents.get_nth(1)) - math.tanh(1.0)) < 1e-6
	assert math.is_nan(sines.get_nth(2))
	assert math.is_nan(cosines.get_nth(2))
	assert math.is_nan(tangents.get_nth(2))
}
