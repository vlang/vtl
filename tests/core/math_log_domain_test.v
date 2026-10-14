module core

import math
import vtl

fn test_signbit_preserves_float_sign_and_marks_integrals_false() ! {
	floats := vtl.from_1d([1.0, -2.0, 0.0, -0.0, math.inf(-1), math.nan()])!
	mask := floats.signbit()!
	assert mask.to_array() == [false, true, false, true, true, false]

	integers := vtl.from_1d([-3, 0, 4])!
	assert integers.signbit()!.to_array() == [true, false, false]
	unsigned_integers := vtl.from_1d[u32]([0, 1, 4])!
	assert unsigned_integers.signbit()!.to_array() == [false, false, false]
	booleans := vtl.from_1d([true, false])!
	assert booleans.signbit()!.to_array() == [false, false]
}

fn test_logaddexp_is_stable_broadcasting_and_ieee_aware() ! {
	a := vtl.from_2d([[0.0], [1000.0], [math.inf(1)], [math.inf(-1)]])!
	b := vtl.from_1d([0.0, 1.0])!
	result := vtl.logaddexp(a, b)!
	assert result.shape == [4, 2]
	assert result.get([0, 0]) == math.log(2.0)
	assert result.get([1, 0]) == 1000.0
	assert result.get([1, 1]) == 1000.0
	assert result.get([2, 0]) == math.inf(1)
	assert result.get([3, 0]) == 0.0
	assert result.get([3, 1]) == 1.0
}

fn test_logaddexp2_and_f32_dtype() ! {
	a := vtl.from_1d[f32]([0.0, 1000.0, f32(math.inf(1))])!
	b := vtl.from_1d[f32]([0.0, 1.0, f32(math.inf(1))])!
	result := vtl.logaddexp2(a, b)!
	assert typeof(result.get_nth(0)).name == 'f32'
	assert math.abs(f64(result.get_nth(0)) - 1.0) < 1e-6
	assert result.get_nth(1) == f32(1000.0)
	assert result.get_nth(2) == f32(math.inf(1))
}

fn test_logaddexp_propagates_nan_and_rejects_incompatible_shapes() ! {
	a := vtl.from_1d([math.nan()])!
	b := vtl.from_1d([1.0])!
	assert math.is_nan(vtl.logaddexp(a, b)!.get_nth(0))

	left := vtl.from_1d([1.0, 2.0])!
	right := vtl.from_1d([1.0, 2.0, 3.0])!
	vtl.logaddexp(left, right) or {
		assert err.msg().contains('broadcast')
		return
	}
	assert false, 'expected incompatible operand shapes to be rejected'
}
