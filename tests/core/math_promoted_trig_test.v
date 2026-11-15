module core

import math
import vtl

fn test_promoted_trig_preserves_dtype_and_values() ! {
	input := vtl.from_1d[f32]([0.0, f32(math.pi / 4.0), f32(math.pi / 2.0)])!
	sines := vtl.sin_promoted[f32, f32](input)!
	cosines := vtl.cos_promoted[f32, f32](input)!
	tangents := vtl.tan_promoted[f32, f32](input)!

	assert sines.dtype() == .float32
	assert cosines.dtype() == .float32
	assert tangents.dtype() == .float32
	assert math.abs(f64(sines.get_nth(0))) < 1e-7
	assert math.abs(f64(sines.get_nth(1)) - math.sqrt(0.5)) < 1e-6
	assert math.abs(f64(cosines.get_nth(1)) - math.sqrt(0.5)) < 1e-6
	assert math.abs(f64(tangents.get_nth(1)) - 1.0) < 1e-6
	assert math.abs(f64(tangents.get_nth(2))) > 1e6
}

fn test_promoted_trig_accepts_integer_inputs_and_propagates_nan() ! {
	integers := vtl.from_1d[i32]([0, 1])!
	sines := vtl.sin_promoted[f64, i32](integers)!
	assert sines.dtype() == .float64
	assert sines.get_nth(0) == 0.0
	assert math.abs(sines.get_nth(1) - math.sin(1.0)) < 1e-14

	values := vtl.from_1d[f64]([math.nan()])!
	assert math.is_nan(vtl.cos_promoted[f64, f64](values)!.get_nth(0))
	assert math.is_nan(vtl.tan_promoted[f64, f64](values)!.get_nth(0))
}
