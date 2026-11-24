module core

import math
import vtl

fn test_modf_returns_fractional_then_integral_parts() ! {
	input := vtl.from_1d([1.25, -2.75, 3.0, -0.0, math.inf(1), math.nan()])!
	fractional, integral := input.modf()!
	assert fractional.shape == input.shape
	assert integral.shape == input.shape
	assert fractional.to_array()[0..4] == [0.25, -0.75, 0.0, -0.0]
	assert integral.to_array()[0..4] == [1.0, -2.0, 3.0, -0.0]
	assert math.is_nan(fractional.get_nth(4))
	assert math.is_nan(fractional.get_nth(5))
	assert integral.get_nth(4) == math.inf(1)
	assert math.is_nan(integral.get_nth(5))
	assert math.signbit(fractional.get_nth(3))
	assert math.signbit(integral.get_nth(3))
}

fn test_modf_promoted_accepts_integer_and_boolean_inputs() ! {
	integers := vtl.from_1d[i16]([1, -2, 0, 8])!
	fractional, integral := vtl.modf_promoted[f64, i16](integers)!
	assert fractional.dtype() == .float64
	assert fractional.to_array() == [0.0, -0.0, 0.0, 0.0]
	assert integral.to_array() == [1.0, -2.0, 0.0, 8.0]
	assert math.signbit(fractional.get_nth(1))

	booleans := vtl.from_1d([true, false])!
	bool_fractional, bool_integral := vtl.modf_promoted[f64, bool](booleans)!
	assert bool_fractional.to_array() == [0.0, 0.0]
	assert bool_integral.to_array() == [1.0, 0.0]
}

fn test_modf_promoted_preserves_f32_and_special_values() ! {
	input := vtl.from_1d[f32]([f32(-0.0), f32(math.inf(1)), f32(math.nan())])!
	fractional, integral := vtl.modf_promoted[f32, f32](input)!
	assert fractional.dtype() == .float32
	assert math.signbit(fractional.get_nth(0))
	assert math.signbit(integral.get_nth(0))
	assert math.is_nan(fractional.get_nth(1))
	assert integral.get_nth(1) == f32(math.inf(1))
	assert math.is_nan(fractional.get_nth(2))
	assert math.is_nan(integral.get_nth(2))

	empty := vtl.empty[i32]([0])
	empty_fractional, empty_integral := vtl.modf_promoted[f64, i32](empty)!
	assert empty_fractional.shape == [0]
	assert empty_integral.shape == [0]
}

fn test_modf_promoted_validates_output_dtype() ! {
	input := vtl.from_1d[i32]([1])!
	vtl.modf_promoted[f32, i32](input) or {
		assert err.msg().contains('output dtype')
		return
	}
	assert false, 'expected integer modf inputs to produce f64 outputs'
}

fn test_frexp_and_ldexp_round_trip_f64() ! {
	input := vtl.from_1d([0.0, -0.0, 0.75, -8.0, 1e-300, 1e300, math.inf(1), math.nan()])!
	mantissa, exponent := input.frexp()!
	assert mantissa.shape == input.shape
	assert exponent.shape == input.shape
	reconstructed := vtl.ldexp(mantissa, exponent)!
	for flat_index in 0 .. input.size {
		value := input.get_nth(flat_index)
		actual := reconstructed.get_nth(flat_index)
		if math.is_nan(value) {
			assert math.is_nan(actual)
		} else {
			assert actual == value
		}
	}
	assert math.signbit(reconstructed.get_nth(1))
	assert exponent.get_nth(6) == 0
}

fn test_frexp_preserves_f32_mantissa_dtype() ! {
	input := vtl.from_1d([f32(0.5), f32(-12.0), f32(1e-30), f32(-2.75)])!
	mantissa, exponent := input.frexp()!
	reconstructed := vtl.ldexp(mantissa, exponent)!
	assert typeof(mantissa.get_nth(0)).name == 'f32'
	assert typeof(reconstructed.get_nth(0)).name == 'f32'
	assert reconstructed.to_array() == input.to_array()
	fractional, integral := input.modf()!
	assert fractional.to_array() == [f32(0.5), f32(-0.0), f32(1e-30), f32(-0.75)]
	assert integral.to_array() == [f32(0.0), f32(-12.0), f32(0.0), f32(-2.0)]
}

fn test_frexp_promoted_accepts_integer_and_boolean_inputs() ! {
	integers := vtl.from_1d[i16]([1, -2, 0, 8])!
	mantissa, exponent := vtl.frexp_promoted[f64, i16](integers)!
	assert mantissa.dtype() == .float64
	assert mantissa.to_array() == [0.5, -0.5, 0.0, 0.5]
	assert exponent.to_array() == [1, 2, 0, 4]

	booleans := vtl.from_1d([true, false])!
	bool_mantissa, bool_exponent := vtl.frexp_promoted[f64, bool](booleans)!
	assert bool_mantissa.to_array() == [0.5, 0.0]
	assert bool_exponent.to_array() == [1, 0]
}

fn test_frexp_promoted_preserves_f32_and_ieee_special_values() ! {
	input := vtl.from_1d[f32]([f32(0.0), f32(-0.0), f32(math.inf(1)), f32(math.nan())])!
	mantissa, exponent := vtl.frexp_promoted[f32, f32](input)!
	assert mantissa.dtype() == .float32
	assert mantissa.get_nth(0) == f32(0.0) && !math.signbit(mantissa.get_nth(0))
	assert mantissa.get_nth(1) == f32(0.0) && math.signbit(mantissa.get_nth(1))
	assert mantissa.get_nth(2) == f32(math.inf(1))
	assert math.is_nan(mantissa.get_nth(3))
	assert exponent.to_array() == [0, 0, 0, 0]

	empty := vtl.empty[i32]([0])
	empty_mantissa, empty_exponent := vtl.frexp_promoted[f64, i32](empty)!
	assert empty_mantissa.shape == [0]
	assert empty_exponent.shape == [0]
}

fn test_frexp_promoted_validates_mantissa_dtype() ! {
	input := vtl.from_1d[i32]([1])!
	vtl.frexp_promoted[f32, i32](input) or {
		assert err.msg().contains('mantissa dtype')
		return
	}
	assert false, 'expected integer frexp inputs to produce f64 mantissas'
}

fn test_decomposition_uses_logical_order_for_strided_views() ! {
	base := vtl.from_2d([[1.5, 2.5], [3.5, 4.5]])!
	view := base.transpose([1, 0])!
	fractional, integral := view.modf()!
	assert fractional.shape == [2, 2]
	assert fractional.to_array() == [0.5, 0.5, 0.5, 0.5]
	assert integral.to_array() == [1.0, 3.0, 2.0, 4.0]
}

fn test_ldexp_broadcasts_mantissas_and_exponents() ! {
	mantissa := vtl.from_2d([[0.5], [-0.25]])!
	exponent := vtl.from_1d([1, 2, 3])!
	result := vtl.ldexp(mantissa, exponent)!
	assert result.shape == [2, 3]
	assert result.to_array() == [1.0, 2.0, 4.0, -0.5, -1.0, -2.0]
}

fn test_ldexp_accepts_scalar_exponents_and_preserves_special_values() ! {
	mantissa := vtl.from_1d[f32]([0.5, -0.0, math.inf(1), math.nan()])!
	result := vtl.ldexp_scalar[f32](mantissa, 2)!

	assert result.dtype() == .float32
	assert result.get_nth(0) == f32(2.0)
	assert math.signbit(result.get_nth(1))
	assert result.get_nth(2) == f32(math.inf(1))
	assert math.is_nan(result.get_nth(3))

	scalar_mantissa := vtl.from_array[f64]([0.5], [])!
	scalar_result := vtl.ldexp_scalar(scalar_mantissa, 1)!
	assert scalar_result.shape.len == 0
	assert scalar_result.get_nth(0) == 1.0

	empty := vtl.from_array[f64]([]f64{}, [0])!
	assert vtl.ldexp_scalar[f64](empty, 3)!.shape == [0]
}

fn test_ldexp_accepts_scalar_mantissas_and_empty_exponent_tensors() ! {
	exponent := vtl.from_1d([1, 2, -1])!
	result := vtl.ldexp_scalar_left[f64](0.5, exponent)!
	assert result.dtype() == .float64
	assert result.to_array() == [1.0, 2.0, 0.25]

	scalar_exponent := vtl.from_array[int]([2], [])!
	scalar_result := vtl.ldexp_scalar_left(0.5, scalar_exponent)!
	assert scalar_result.shape.len == 0
	assert scalar_result.get_nth(0) == 2.0

	empty_exponent := vtl.from_array[int]([]int{}, [0])!
	empty_result := vtl.ldexp_scalar_left[f64](0.5, empty_exponent)!
	assert empty_result.shape == [0]
	assert empty_result.size == 0
}

fn test_ldexp_promoted_broadcasts_integer_mantissas_and_exponents() ! {
	mantissa := vtl.from_2d[i16]([[1, 2], [3, 4]])!
	exponent := vtl.from_1d[i8]([0, 1])!
	result := vtl.ldexp_promoted[f64, i16, i8](mantissa, exponent)!

	assert result.dtype() == .float64
	assert result.shape == [2, 2]
	assert result.to_array() == [1.0, 4.0, 3.0, 8.0]

	empty_mantissa := vtl.from_array[f64]([]f64{}, [0])!
	empty_exponent := vtl.from_array[i16]([2], [])!
	empty_result := vtl.ldexp_promoted[f64, f64, i16](empty_mantissa, empty_exponent)!
	assert empty_result.shape == [0]
	assert empty_result.size == 0
}

fn test_ldexp_promoted_preserves_float32_and_special_values() ! {
	mantissa := vtl.from_1d[f32]([0.5, -0.0, math.inf(1), math.nan()])!
	exponent := vtl.from_array[bool]([true], [])!
	result := vtl.ldexp_promoted[f32, f32, bool](mantissa, exponent)!

	assert result.dtype() == .float32
	assert result.get_nth(0) == f32(1.0)
	assert math.signbit(result.get_nth(1))
	assert result.get_nth(2) == f32(math.inf(1))
	assert math.is_nan(result.get_nth(3))
}

fn test_ldexp_promoted_rejects_invalid_dtypes() ! {
	mantissa := vtl.from_1d[f64]([1.0])!
	float_exponent := vtl.from_1d[f32]([1.0])!
	vtl.ldexp_promoted[f64, f64, f32](mantissa, float_exponent) or {
		assert err.msg().contains('integer or boolean exponent')
		return
	}
	assert false, 'expected floating-point exponents to be rejected'

	exponent := vtl.from_1d([1])!
	if _ := vtl.ldexp_promoted[f32, f64, int](mantissa, exponent) {
		assert false, 'expected an f64 mantissa to require an f64 result'
	}
}

fn test_ldexp_scalar_rejects_non_floating_mantissas() ! {
	mantissa := vtl.from_1d[i32]([1, 2])!
	vtl.ldexp_scalar[i32](mantissa, 1) or {
		assert err.msg().contains('f32 or f64')
		return
	}
	assert false, 'expected ldexp to reject integer mantissas'
}

fn test_modf_rejects_non_floating_tensors() ! {
	input := vtl.from_1d([1, 2, 3])!
	input.modf() or {
		assert err.msg().contains('f32 or f64')
		return
	}
	assert false, 'expected modf to reject integer tensors'
}

fn test_ldexp_rejects_incompatible_shapes() ! {
	mantissa := vtl.from_1d([0.5, 0.25])!
	exponent := vtl.from_1d([1, 2, 3])!
	vtl.ldexp(mantissa, exponent) or {
		assert err.msg().contains('broadcast')
		return
	}
	assert false, 'expected incompatible operand shapes to be rejected'
}
