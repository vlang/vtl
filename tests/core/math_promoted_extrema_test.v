module core

import math
import math.complex
import vtl

fn test_promoted_extrema_broadcast_without_losing_integer_precision() ! {
	left := vtl.from_array[i64]([9007199254740993, -4], [2, 1])!
	right := vtl.from_array[i16]([9000, -9000], [1, 2])!

	maximum := vtl.maximum_promoted[i64, i64, i16](left, right)!
	minimum := vtl.minimum_promoted[i64, i64, i16](left, right)!

	assert maximum.dtype() == .int64
	assert minimum.dtype() == .int64
	assert maximum.shape == [2, 2]
	assert maximum.to_array() == [i64(9007199254740993), 9007199254740993, 9000, -4]
	assert minimum.to_array() == [9000, -9000, -4, -9000]
}

fn test_promoted_extrema_match_nan_and_signed_zero_rules() ! {
	nan := math.nan()
	left := vtl.from_1d[f64]([nan, 1.0, nan, 0.0, -0.0])!
	right := vtl.from_1d[f32]([2.0, nan, nan, -0.0, 0.0])!

	maximum := vtl.maximum_promoted[f64, f64, f32](left, right)!
	minimum := vtl.minimum_promoted[f64, f64, f32](left, right)!
	fmaximum := vtl.fmax_promoted[f64, f64, f32](left, right)!
	fminimum := vtl.fmin_promoted[f64, f64, f32](left, right)!

	assert math.is_nan(maximum.get_nth(0))
	assert math.is_nan(maximum.get_nth(1))
	assert math.is_nan(maximum.get_nth(2))
	assert math.is_nan(minimum.get_nth(0))
	assert math.is_nan(minimum.get_nth(1))
	assert math.is_nan(minimum.get_nth(2))
	assert fmaximum.to_array()[0..2] == [2.0, 1.0]
	assert fminimum.to_array()[0..2] == [2.0, 1.0]
	assert math.is_nan(fmaximum.get_nth(2))
	assert math.is_nan(fminimum.get_nth(2))
	assert math.signbit(maximum.get_nth(3))
	assert !math.signbit(maximum.get_nth(4))
	assert math.signbit(minimum.get_nth(3))
	assert !math.signbit(minimum.get_nth(4))
	assert !math.signbit(fmaximum.get_nth(3))
	assert !math.signbit(fmaximum.get_nth(4))
	assert math.signbit(fminimum.get_nth(3))
	assert math.signbit(fminimum.get_nth(4))
}

fn test_promoted_extrema_support_boolean_and_complex_dtypes() ! {
	left_bool := vtl.from_2d([[true], [false]])!
	right_bool := vtl.from_1d([false, true])!
	boolean_maximum := vtl.maximum_promoted[bool, bool, bool](left_bool, right_bool)!
	assert boolean_maximum.dtype() == .boolean
	assert boolean_maximum.to_array() == [true, true, false, true]

	left_complex := vtl.from_1d[complex.Complex]([
		complex.Complex{ re: 1.0, im: 2.0 },
		complex.Complex{ re: math.nan(), im: 0.0 },
	])!
	right_integer := vtl.from_1d[i16]([1, 2])!
	complex_maximum := vtl.maximum_promoted[complex.Complex, complex.Complex, i16](left_complex,
		right_integer)!
	complex_fmaximum := vtl.fmax_promoted[complex.Complex, complex.Complex, i16](left_complex,
		right_integer)!
	assert complex_maximum.get_nth(0) == left_complex.get_nth(0)
	assert math.is_nan(complex_maximum.get_nth(1).re)
	assert complex_fmaximum.get_nth(1) == complex.Complex{ re: 2.0, im: 0.0 }
}

fn test_promoted_extrema_validate_output_dtype_and_keep_empty_shapes() ! {
	left := vtl.from_1d[i16]([1, 2])!
	right := vtl.from_1d[i8]([0])!
	if _ := vtl.maximum_promoted[i32, i16, i8](left, right) {
		assert false, 'promoted extrema must reject an output dtype that differs from promotion'
	}

	empty := vtl.from_array[f32]([], [0, 1])!
	other := vtl.from_array[f64]([1.0, 2.0], [1, 2])!
	result := vtl.fmax_promoted[f64, f32, f64](empty, other)!
	assert result.dtype() == .float64
	assert result.shape == [0, 2]
	assert result.size == 0
}

fn test_promoted_extrema_reject_string_tensors() ! {
	values := vtl.from_1d(['left', 'right'])!
	if _ := vtl.maximum_promoted[string, string, string](values, values) {
		assert false, 'promoted extrema must reject string tensors'
	}
}
