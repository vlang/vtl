module core

import math
import math.complex
import vtl

fn test_sqrt_promoted_selects_result_dtype_and_values() ! {
	integers := vtl.from_1d[i32]([0, 1, 2, 9])!
	integer_roots := vtl.sqrt_promoted[f64, i32](integers)!
	expected_integer_roots := vtl.from_1d[f64]([0.0, 1.0, math.sqrt(2.0), 3.0])!
	assert integer_roots.dtype() == .float64
	assert integer_roots.allclose(expected_integer_roots, rtol: 1e-14, atol: 1e-14)!

	small_integers := vtl.from_1d[i8]([0, 2])!
	small_integer_roots := vtl.sqrt_promoted[f32, i8](small_integers)!
	expected_small_roots := vtl.from_1d[f32]([0.0, f32(math.sqrt(2.0))])!
	assert small_integer_roots.dtype() == .float32
	assert small_integer_roots.allclose(expected_small_roots, rtol: 1e-6, atol: 1e-6)!

	floats := vtl.from_1d[f32]([1.0, 4.0])!
	float_roots := vtl.sqrt_promoted[f32, f32](floats)!
	assert float_roots.dtype() == .float32
	expected_float_roots := vtl.from_1d[f32]([1.0, 2.0])!
	assert float_roots.allclose(expected_float_roots, rtol: 0.0, atol: 0.0)!
}

fn test_sqrt_promoted_handles_boolean_special_and_empty_values() ! {
	booleans := vtl.from_1d([true, false])!
	boolean_roots := vtl.sqrt_promoted[f32, bool](booleans)!
	assert boolean_roots.dtype() == .float32
	expected_boolean_roots := vtl.from_1d[f32]([1.0, 0.0])!
	assert boolean_roots.allclose(expected_boolean_roots, rtol: 0.0, atol: 0.0)!

	special := vtl.from_1d[f64]([-0.0, -1.0, math.inf(1), math.nan()])!
	special_roots := vtl.sqrt_promoted[f64, f64](special)!
	values := special_roots.to_array()
	assert values[0] == 0.0 && math.signbit(values[0])
	assert math.is_nan(values[1])
	assert math.is_inf(values[2], 1)
	assert math.is_nan(values[3])

	empty := vtl.empty[i16]([2, 0])
	empty_roots := vtl.sqrt_promoted[f32, i16](empty)!
	assert empty_roots.shape == [2, 0]
	assert empty_roots.size == 0
}

fn test_sqrt_promoted_rejects_unsupported_inputs_and_result_dtype() {
	integers := vtl.from_1d[i32]([1])!
	if _ := vtl.sqrt_promoted[f32, i32](integers) {
		assert false, '32-bit integer sqrt must produce float64'
	}

	complex_values := vtl.from_1d[complex.Complex]([complex.complex(1.0, 2.0)])!
	if _ := vtl.sqrt_promoted[f64, complex.Complex](complex_values) {
		assert false, 'complex sqrt is unsupported by this real-valued API'
	}

	strings := vtl.from_1d(['1'])!
	if _ := vtl.sqrt_promoted[f64, string](strings) {
		assert false, 'string sqrt is unsupported'
	}
}
