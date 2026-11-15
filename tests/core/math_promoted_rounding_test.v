module core

import math
import vtl

fn test_promoted_rint_uses_ties_to_even_and_float_output() ! {
	values := vtl.from_1d[f32]([-2.5, -1.5, 0.5, 1.5, 2.5])!
	rounded := vtl.rint_promoted[f32, f32](values)!
	assert rounded.dtype() == .float32
	expected := vtl.from_1d[f32]([-2.0, -2.0, 0.0, 2.0, 2.0])!
	assert rounded.allclose(expected, rtol: 0.0, atol: 0.0)!

	integer_values := vtl.from_1d[i16]([1, 2])!
	integer_rounded := vtl.rint_promoted[f32, i16](integer_values)!
	assert integer_rounded.dtype() == .float32
	assert integer_rounded.get_nth(0) == 1.0
	assert integer_rounded.get_nth(1) == 2.0
}

fn test_promoted_floor_ceil_trunc_preserve_dtype_and_integer_precision() ! {
	large_values := vtl.from_1d[i64]([9_007_199_254_740_993, -9_007_199_254_740_993])!
	floors := vtl.floor_promoted[i64, i64](large_values)!
	ceilings := vtl.ceil_promoted[i64, i64](large_values)!
	truncated := vtl.trunc_promoted[i64, i64](large_values)!
	assert floors.dtype() == .int64
	assert ceilings.dtype() == .int64
	assert truncated.dtype() == .int64
	assert floors.get_nth(0) == 9_007_199_254_740_993
	assert ceilings.get_nth(1) == -9_007_199_254_740_993
	assert truncated.get_nth(0) == 9_007_199_254_740_993

	float_values := vtl.from_1d[f32]([1.75, -1.75, math.nan()])!
	float_floors := vtl.floor_promoted[f32, f32](float_values)!
	float_ceilings := vtl.ceil_promoted[f32, f32](float_values)!
	float_truncated := vtl.trunc_promoted[f32, f32](float_values)!
	assert float_floors.get_nth(0) == 1.0
	assert float_floors.get_nth(1) == -2.0
	assert math.is_nan(float_floors.get_nth(2))
	assert float_ceilings.get_nth(0) == 2.0
	assert float_ceilings.get_nth(1) == -1.0
	assert math.is_nan(float_ceilings.get_nth(2))
	assert float_truncated.get_nth(0) == 1.0
	assert float_truncated.get_nth(1) == -1.0
	assert math.is_nan(float_truncated.get_nth(2))

	booleans := vtl.from_1d([true, false])!
	assert vtl.floor_promoted[bool, bool](booleans)!.to_array() == [true, false]
	assert vtl.ceil_promoted[bool, bool](booleans)!.to_array() == [true, false]
	assert vtl.trunc_promoted[bool, bool](booleans)!.to_array() == [true, false]
}

fn test_promoted_rounding_validates_output_dtype() ! {
	values := vtl.from_1d[i16]([1])!
	_ := vtl.floor_promoted[f32, i16](values) or { return }
	assert false, 'expected output dtype mismatch to return an error'
}
