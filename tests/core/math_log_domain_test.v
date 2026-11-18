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

fn test_logaddexp_scalar_supports_both_operand_orders_and_preserves_dtype() ! {
	values := vtl.from_1d([0.0, 1000.0, -1000.0])!
	right := vtl.logaddexp_scalar(values, 0.0)!
	left := vtl.logaddexp_scalar_left(0.0, values)!
	assert right.shape == values.shape
	assert right.dtype() == .float64
	assert right.to_array() == left.to_array()
	assert right.get_nth(0) == math.log(2.0)
	assert right.get_nth(1) == 1000.0
	assert right.get_nth(2) == 0.0

	values32 := vtl.from_1d[f32]([0.0, 1000.0, -1000.0])!
	right32 := vtl.logaddexp_scalar[f32](values32, 0.0)!
	left32 := vtl.logaddexp_scalar_left[f32](0.0, values32)!
	assert right32.shape == values32.shape
	assert right32.dtype() == .float32
	assert typeof(right32.get_nth(0)).name == 'f32'
	assert right32.to_array() == left32.to_array()
	assert math.abs(f64(right32.get_nth(0)) - math.log(2.0)) < 1e-6
	assert right32.get_nth(1) == f32(1000.0)
	assert right32.get_nth(2) == f32(0.0)
}

fn test_logaddexp_scalar_handles_empty_scalar_shaped_and_strided_tensors() ! {
	empty := vtl.empty[f32]([0])
	assert vtl.logaddexp_scalar(empty, f32(1.0))!.shape == [0]
	assert vtl.logaddexp_scalar_left(f32(1.0), empty)!.dtype() == .float32

	scalar_tensor := vtl.from_array([2.0], [])!
	right_scalar := vtl.logaddexp_scalar(scalar_tensor, 3.0)!
	left_scalar := vtl.logaddexp_scalar_left(3.0, scalar_tensor)!
	assert right_scalar.shape.len == 0
	assert right_scalar.get_nth(0) == 3.0 + math.log1p(math.exp(-1.0))
	assert left_scalar.get_nth(0) == right_scalar.get_nth(0)

	base := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
	view := base.transpose([1, 0])!
	right_view := vtl.logaddexp_scalar(view, 10.0)!
	left_view := vtl.logaddexp_scalar_left(10.0, view)!
	assert right_view.shape == [2, 2]
	assert right_view.to_array() == left_view.to_array()
	assert right_view.get([0, 0]) == 10.0 + math.log1p(math.exp(-9.0))
	assert right_view.get([0, 1]) == 10.0 + math.log1p(math.exp(-7.0))
	assert right_view.get([1, 0]) == 10.0 + math.log1p(math.exp(-8.0))
	assert right_view.get([1, 1]) == 10.0 + math.log1p(math.exp(-6.0))
}

fn test_logaddexp_scalar_handles_extremes_nan_and_infinities() ! {
	extremes := vtl.from_1d([1000.0, -1000.0])!
	right := vtl.logaddexp_scalar(extremes, 1000.0)!
	left := vtl.logaddexp_scalar_left(-1000.0, extremes)!
	assert right.get_nth(0) == 1000.0 + math.log(2.0)
	assert right.get_nth(1) == 1000.0
	assert left.get_nth(0) == 1000.0
	assert left.get_nth(1) == -1000.0 + math.log(2.0)

	values := vtl.from_1d([math.nan(), math.inf(1), math.inf(-1), 2.0])!
	right_special := vtl.logaddexp_scalar(values, math.inf(-1))!
	left_special := vtl.logaddexp_scalar_left(math.inf(-1), values)!
	assert math.is_nan(right_special.get_nth(0))
	assert math.is_nan(left_special.get_nth(0))
	assert right_special.get_nth(1) == math.inf(1)
	assert left_special.get_nth(1) == math.inf(1)
	assert right_special.get_nth(2) == math.inf(-1)
	assert left_special.get_nth(2) == math.inf(-1)
	assert right_special.get_nth(3) == 2.0

	positive_infinity := vtl.from_1d([math.inf(-1)])!
	assert vtl.logaddexp_scalar(positive_infinity, math.inf(1))!.get_nth(0) == math.inf(1)
	assert vtl.logaddexp_scalar_left(math.inf(1), positive_infinity)!.get_nth(0) == math.inf(1)
}

fn test_logaddexp_scalar_rejects_unsupported_dtypes() ! {
	integers := vtl.from_1d([1, 2])!
	vtl.logaddexp_scalar(integers, 1) or {
		assert err.msg().contains('f32 or f64')
		return
	}
	assert false, 'logaddexp_scalar should reject integer values'
}

fn test_logaddexp_scalar_left_rejects_unsupported_dtypes() ! {
	integers := vtl.from_1d([1, 2])!
	vtl.logaddexp_scalar_left(1, integers) or {
		assert err.msg().contains('f32 or f64')
		return
	}
	assert false, 'logaddexp_scalar_left should reject integer values'
}

fn test_copysign_broadcasts_and_preserves_special_sign_bits() ! {
	magnitude := vtl.from_2d([[1.0, 0.0], [math.inf(1), math.nan()]])!
	sign := vtl.from_1d([-1.0, 1.0])!
	result := vtl.copysign(magnitude, sign)!
	assert result.shape == [2, 2]
	assert result.get([0, 0]) == -1.0
	assert result.get([0, 1]) == 0.0
	assert !math.signbit(result.get([0, 1]))
	assert result.get([1, 0]) == math.inf(-1)
	assert math.is_nan(result.get([1, 1]))
	assert !math.signbit(result.get([1, 1]))

	negative_sign := vtl.from_1d([math.copysign(0.0, -1.0)])!
	negative_zero := vtl.copysign(vtl.from_1d([0.0])!, negative_sign)!
	assert negative_zero.get_nth(0) == 0.0
	assert math.signbit(negative_zero.get_nth(0))
}

fn test_copysign_preserves_f32_dtype() ! {
	magnitude := vtl.from_1d[f32]([1.5, 0.0])!
	sign := vtl.from_1d[f32]([-1.0, -1.0])!
	result := vtl.copysign(magnitude, sign)!
	assert typeof(result.get_nth(0)).name == 'f32'
	assert result.to_array() == [f32(-1.5), f32(-0.0)]
}
