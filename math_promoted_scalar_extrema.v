module vtl

import math.complex

// maximum_scalar_promoted applies NumPy-style maximum to a tensor and weak
// scalar. R must match the inferred weak-scalar result dtype.
pub fn maximum_scalar_promoted[R, T, U](a &Tensor[T], scalar U) !&Tensor[R] {
	return scalar_extreme_promoted[R, T, U](a, scalar, true, false, false)
}

// maximum_scalar_left_promoted applies maximum with the scalar as the left
// operand, preserving operand order for ties and NaNs.
pub fn maximum_scalar_left_promoted[R, T, U](scalar U, a &Tensor[T]) !&Tensor[R] {
	return scalar_extreme_promoted[R, T, U](a, scalar, true, false, true)
}

// minimum_scalar_promoted applies NumPy-style minimum to a tensor and weak
// scalar. R must match the inferred weak-scalar result dtype.
pub fn minimum_scalar_promoted[R, T, U](a &Tensor[T], scalar U) !&Tensor[R] {
	return scalar_extreme_promoted[R, T, U](a, scalar, false, false, false)
}

// minimum_scalar_left_promoted applies minimum with the scalar as the left
// operand, preserving operand order for ties and NaNs.
pub fn minimum_scalar_left_promoted[R, T, U](scalar U, a &Tensor[T]) !&Tensor[R] {
	return scalar_extreme_promoted[R, T, U](a, scalar, false, false, true)
}

// fmax_scalar_promoted ignores a single NaN operand, like numpy.fmax.
pub fn fmax_scalar_promoted[R, T, U](a &Tensor[T], scalar U) !&Tensor[R] {
	return scalar_extreme_promoted[R, T, U](a, scalar, true, true, false)
}

// fmax_scalar_left_promoted ignores a single NaN operand with the scalar on
// the left.
pub fn fmax_scalar_left_promoted[R, T, U](scalar U, a &Tensor[T]) !&Tensor[R] {
	return scalar_extreme_promoted[R, T, U](a, scalar, true, true, true)
}

// fmin_scalar_promoted ignores a single NaN operand, like numpy.fmin.
pub fn fmin_scalar_promoted[R, T, U](a &Tensor[T], scalar U) !&Tensor[R] {
	return scalar_extreme_promoted[R, T, U](a, scalar, false, true, false)
}

// fmin_scalar_left_promoted ignores a single NaN operand with the scalar on
// the left.
pub fn fmin_scalar_left_promoted[R, T, U](scalar U, a &Tensor[T]) !&Tensor[R] {
	return scalar_extreme_promoted[R, T, U](a, scalar, false, true, true)
}

fn scalar_extreme_promoted[R, T, U](a &Tensor[T], scalar U, want_max bool, ignore_nan bool, scalar_on_left bool) !&Tensor[R] {
	result_dtype := weak_extrema_scalar_result_dtype[T, U](scalar)!
	if dtype_of[R]() != result_dtype {
		return error('output dtype ${dtype_of[R]()} does not match weak-scalar result dtype ${result_dtype}')
	}
	$if R is complex.Complex || R is bool || R is $int || R is $float {
		scalar_value := cast_scalar_extreme_value[U, R](scalar)
		mut result := empty[R](a.shape, memory: .row_major)
		for index in 0 .. a.size {
			tensor_value := cast_scalar_extreme_value[T, R](a.get_nth[T](index))
			result.data.data[index] = if scalar_on_left {
				numpy_extreme[R](scalar_value, tensor_value, want_max, ignore_nan)
			} else {
				numpy_extreme[R](tensor_value, scalar_value, want_max, ignore_nan)
			}
		}
		return result
	} $else {
		return error('promoted scalar extrema require real numeric, boolean, or complex128 values')
	}
}

fn weak_extrema_scalar_result_dtype[T, U](scalar U) !DType {
	array_dtype := dtype_of[T]()
	scalar_dtype := dtype_of[U]()
	if array_dtype == .complex128 || scalar_dtype == .complex128 {
		return weak_complex_scalar_result_dtype[T, U]()
	}
	if array_dtype == .boolean {
		if scalar_dtype == .boolean {
			return .boolean
		}
		if is_real_numeric_dtype(scalar_dtype) {
			return promote_types(.boolean, scalar_dtype)
		}
		return error('weak-scalar promoted extrema requires numeric or boolean operands')
	}
	return weak_scalar_result_dtype[T, U](array_dtype, scalar)
}

@[inline]
fn cast_scalar_extreme_value[T, R](value T) R {
	$if R is bool {
		$if T is bool {
			return value
		}
	} $else {
		return cast_promoted_value[T, R](value)
	}
	panic('promoted scalar extrema require real numeric, boolean, or complex128 values')
}
