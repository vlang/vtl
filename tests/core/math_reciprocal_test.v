module core

import math
import vtl

fn test_reciprocal_f64_values_and_ieee_special_values() ! {
	input := vtl.from_1d[f64]([
		0.0,
		-0.0,
		2.0,
		-4.0,
		math.inf(1),
		math.inf(-1),
		math.nan(),
		1e-320,
		-1e-320,
	])!
	result := input.reciprocal()!

	assert result.dtype() == .float64
	assert result.shape == input.shape
	assert math.is_inf(result.get_nth(0), 1)
	assert math.is_inf(result.get_nth(1), -1)
	assert result.get_nth(2) == 0.5
	assert result.get_nth(3) == -0.25
	assert result.get_nth(4) == 0.0 && !math.signbit(result.get_nth(4))
	assert result.get_nth(5) == 0.0 && math.signbit(result.get_nth(5))
	assert math.is_nan(result.get_nth(6))
	assert math.is_inf(result.get_nth(7), 1)
	assert math.is_inf(result.get_nth(8), -1)
}

fn test_reciprocal_f32_preserves_dtype() ! {
	input := vtl.from_1d[f32]([0.5, -2.0, f32(math.inf(1)), f32(math.inf(-1))])!
	result := input.reciprocal()!

	assert result.dtype() == .float32
	assert result.to_array() == [f32(2.0), f32(-0.5), f32(0.0), f32(-0.0)]
}

fn test_reciprocal_reads_strided_views_and_preserves_empty_shape() ! {
	input := vtl.from_1d[f64]([1.0, 2.0, 4.0])!.flip()!
	result := input.reciprocal()!
	assert result.to_array() == [0.25, 0.5, 1.0]

	empty := vtl.from_1d[f64]([]f64{})!
	empty_result := empty.reciprocal()!
	assert empty_result.shape == empty.shape
	assert empty_result.size == 0
}

fn test_reciprocal_rejects_non_floating_tensors() ! {
	input := vtl.from_1d([1, 2, 3])!
	input.reciprocal() or {
		assert err.msg().contains('f32 or f64')
		return
	}
	assert false, 'expected reciprocal to reject integer tensors'
}
