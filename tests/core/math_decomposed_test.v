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
