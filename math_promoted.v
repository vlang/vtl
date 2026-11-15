module vtl

import math
import math.complex

enum PromotedBinaryOperation {
	add
	subtract
	multiply
	divide
	remainder
}

// add_promoted adds numeric tensors after converting both operands to
// the explicitly selected NumPy-style promoted dtype. The output type R must
// match promote_types(dtype_of[T](), dtype_of[U]()).
pub fn add_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_promoted[R, T, U](a, b, .add)
}

// subtract_promoted subtracts numeric tensors after converting both
// operands to the explicitly selected NumPy-style promoted dtype. The output
// type R must match promote_types(dtype_of[T](), dtype_of[U]()).
pub fn subtract_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_promoted[R, T, U](a, b, .subtract)
}

// multiply_promoted multiplies numeric tensors after converting both
// operands to the explicitly selected NumPy-style promoted dtype. The output
// type R must match promote_types(dtype_of[T](), dtype_of[U]()).
pub fn multiply_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_promoted[R, T, U](a, b, .multiply)
}

// divide_promoted performs NumPy-style true division for numeric tensors and
// booleans. Complex inputs produce complex128; real inputs produce float32 only
// when the promoted input dtype is float32, and float64 otherwise.
pub fn divide_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_promoted[R, T, U](a, b, .divide)
}

// sqrt_promoted computes elementwise square roots for real numeric or boolean
// tensors. It returns float64 for float64 and >=32-bit integer inputs, and
// float32 otherwise; NumPy float16 results are widened because VTL does not
// support float16.
pub fn sqrt_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.sqrt)
}

// cbrt_promoted computes elementwise cube roots using the same output dtype
// rules as sqrt_promoted.
pub fn cbrt_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.cbrt)
}

// exp_promoted computes elementwise exponential values using the same output
// dtype rules as sqrt_promoted.
pub fn exp_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.exp)
}

// exp2_promoted computes elementwise powers of two using the same output dtype
// rules as sqrt_promoted.
pub fn exp2_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.exp2)
}

// expm1_promoted computes exp(x)-1 accurately near zero using the same output
// dtype rules as sqrt_promoted.
pub fn expm1_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.expm1)
}

// log_promoted computes elementwise natural logarithms using the same output
// dtype rules as sqrt_promoted.
pub fn log_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.log)
}

// log2_promoted computes elementwise base-two logarithms using the same output
// dtype rules as sqrt_promoted.
pub fn log2_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.log2)
}

// log10_promoted computes elementwise base-ten logarithms using the same output
// dtype rules as sqrt_promoted.
pub fn log10_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.log10)
}

// log1p_promoted computes log(1+x) accurately near zero using the same output
// dtype rules as sqrt_promoted.
pub fn log1p_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.log1p)
}

// sin_promoted computes elementwise sine values using the same output dtype
// rules as sqrt_promoted.
pub fn sin_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.sin)
}

// cos_promoted computes elementwise cosine values using the same output dtype
// rules as sqrt_promoted.
pub fn cos_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.cos)
}

// tan_promoted computes elementwise tangent values using the same output dtype
// rules as sqrt_promoted.
pub fn tan_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.tan)
}

// asin_promoted computes elementwise inverse sine values using the same output
// dtype rules as sqrt_promoted.
pub fn asin_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.asin)
}

// acos_promoted computes elementwise inverse cosine values using the same
// output dtype rules as sqrt_promoted.
pub fn acos_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.acos)
}

// atan_promoted computes elementwise inverse tangent values using the same
// output dtype rules as sqrt_promoted.
pub fn atan_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.atan)
}

// sinh_promoted computes elementwise hyperbolic sine values using the same
// output dtype rules as sqrt_promoted.
pub fn sinh_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.sinh)
}

// cosh_promoted computes elementwise hyperbolic cosine values using the same
// output dtype rules as sqrt_promoted.
pub fn cosh_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.cosh)
}

// tanh_promoted computes elementwise hyperbolic tangent values using the same
// output dtype rules as sqrt_promoted.
pub fn tanh_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.tanh)
}

// asinh_promoted computes elementwise inverse hyperbolic sine using the same
// output dtype rules as sqrt_promoted.
pub fn asinh_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.asinh)
}

// acosh_promoted computes elementwise inverse hyperbolic cosine using the same
// output dtype rules as sqrt_promoted.
pub fn acosh_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.acosh)
}

// atanh_promoted computes elementwise inverse hyperbolic tangent using the
// same output dtype rules as sqrt_promoted.
pub fn atanh_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.atanh)
}

// rint_promoted rounds elementwise to the nearest integer using ties-to-even
// and the floating output dtype rules of sqrt_promoted.
pub fn rint_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.round_to_even)
}

// floor_promoted returns elementwise floors while preserving the input dtype.
pub fn floor_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_dtype_preserving_ufunc_promoted[R, T](t, math.floor)
}

// ceil_promoted returns elementwise ceilings while preserving the input dtype.
pub fn ceil_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_dtype_preserving_ufunc_promoted[R, T](t, math.ceil)
}

// trunc_promoted truncates elementwise toward zero while preserving the input
// dtype.
pub fn trunc_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_dtype_preserving_ufunc_promoted[R, T](t, math.trunc)
}

fn unary_dtype_preserving_ufunc_promoted[R, T](t &Tensor[T], operation fn (f64) f64) !&Tensor[R] {
	input_dtype := dtype_of[T]()
	if !(is_real_numeric_dtype(input_dtype) || input_dtype == .boolean) {
		return error('promoted rounding supports only real numeric and boolean tensors')
	}
	if dtype_of[R]() != input_dtype {
		return error('output dtype ${dtype_of[R]()} does not match input dtype ${input_dtype}')
	}

	mut result := empty[R](t.shape, memory: .row_major)
	$if R is bool {
		$if T is bool {
			for index in 0 .. t.size {
				result.data.data[index] = t.get_nth[T](index)
			}
		} $else {
			return error('rounding output dtype must match the input dtype')
		}
	} $else $if R is $int || R is $float {
		for index in 0 .. t.size {
			value := t.get_nth[T](index)
			converted := cast_promoted_value[T, R](value)
			if is_integer_dtype(input_dtype) || input_dtype == .boolean {
				result.data.data[index] = converted
			} else {
				result.data.data[index] = R(operation(f64(converted)))
			}
		}
	} $else {
		return error('promoted rounding output must be real numeric or boolean')
	}
	return result
}

fn unary_real_ufunc_promoted[R, T](t &Tensor[T], operation fn (f64) f64) !&Tensor[R] {
	input_dtype := dtype_of[T]()
	if !(is_real_numeric_dtype(input_dtype) || input_dtype == .boolean) {
		return error('promoted unary math supports only real numeric and boolean tensors')
	}
	mut expected_dtype := DType.float32
	if numpy_float_ufunc_uses_f64(input_dtype) {
		expected_dtype = .float64
	}
	if dtype_of[R]() != expected_dtype {
		return error('output dtype ${dtype_of[R]()} does not match unary math result dtype ${expected_dtype}')
	}

	mut result := empty[R](t.shape, memory: .row_major)
	for index in 0 .. t.size {
		value := cast_promoted_value[T, R](t.get_nth[T](index))
		result.data.data[index] = R(operation(f64(value)))
	}
	return result
}

// atan2_promoted computes elementwise atan2 for real numeric or boolean
// tensors. It returns float64 when either input is float64 or a >=32-bit
// integer; other results use float32. NumPy float16 results for booleans and
// <=8-bit integers are widened because VTL does not support float16.
pub fn atan2_promoted[R, T, U](y &Tensor[T], x &Tensor[U]) !&Tensor[R] {
	return binary_real_ufunc_promoted[R, T, U](y, x, math.atan2, 'atan2')
}

// copysign_promoted applies the sign bit of a real numeric or boolean tensor
// to another tensor's magnitude, with NumPy-style float output promotion.
pub fn copysign_promoted[R, T, U](magnitude &Tensor[T], sign &Tensor[U]) !&Tensor[R] {
	return binary_real_ufunc_promoted[R, T, U](magnitude, sign, math.copysign, 'copysign')
}

// hypot_promoted computes elementwise hypotenuses with NumPy-style float
// output promotion and broadcasting.
pub fn hypot_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_real_ufunc_promoted[R, T, U](a, b, math.hypot, 'hypot')
}

// logaddexp_promoted computes log(exp(a)+exp(b)) stably with NumPy-style
// floating output promotion and broadcasting.
pub fn logaddexp_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_real_ufunc_promoted[R, T, U](a, b, logaddexp_value, 'logaddexp')
}

// logaddexp2_promoted computes log2(2^a+2^b) stably with NumPy-style
// floating output promotion and broadcasting.
pub fn logaddexp2_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_real_ufunc_promoted[R, T, U](a, b, logaddexp2_value, 'logaddexp2')
}

fn binary_real_ufunc_promoted[R, T, U](a &Tensor[T], b &Tensor[U], operation fn (f64, f64) f64, name string) !&Tensor[R] {
	a_dtype := dtype_of[T]()
	b_dtype := dtype_of[U]()
	if !(is_real_numeric_dtype(a_dtype) || a_dtype == .boolean)
		|| !(is_real_numeric_dtype(b_dtype) || b_dtype == .boolean) {
		return error('promoted ${name} supports only real numeric and boolean tensors')
	}
	mut expected_dtype := DType.float32
	if numpy_float_ufunc_uses_f64(a_dtype) || numpy_float_ufunc_uses_f64(b_dtype) {
		expected_dtype = .float64
	}
	if dtype_of[R]() != expected_dtype {
		return error('output dtype ${dtype_of[R]()} does not match ${name} result dtype ${expected_dtype}')
	}

	shape := broadcast_shapes(a.shape, b.shape)!
	left := a.broadcast_to(shape)!
	right := b.broadcast_to(shape)!
	mut result := empty[R](shape, memory: .row_major)
	left_is_flat := left.size == result.size && left.is_row_major_contiguous()
	right_is_flat := right.size == result.size && right.is_row_major_contiguous()
	for index in 0 .. result.size {
		left_offset := if left_is_flat {
			index
		} else if a.size == 1 {
			0
		} else {
			broadcast_tensor_offset(index, shape, left.strides)
		}
		right_offset := if right_is_flat {
			index
		} else if b.size == 1 {
			0
		} else {
			broadcast_tensor_offset(index, shape, right.strides)
		}
		left_value := cast_promoted_value[T, R](left.data.data[left_offset])
		right_value := cast_promoted_value[U, R](right.data.data[right_offset])
		result.data.data[index] = R(operation(f64(left_value), f64(right_value)))
	}
	return result
}

fn numpy_float_ufunc_uses_f64(dtype DType) bool {
	return dtype == .float64 || (is_integer_dtype(dtype) && dtype_bits(dtype) >= 32)
}

// remainder_promoted returns the element-wise NumPy remainder of real numeric
// tensors after dtype promotion and broadcasting. Complex inputs are rejected.
// Integer zero divisors produce zero; floating-point zero divisors produce NaN.
pub fn remainder_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_promoted[R, T, U](a, b, .remainder)
}

// add_scalar_promoted adds a weak numeric or complex scalar using NumPy-style
// promotion.
// R must be the inferred result dtype; this explicit type is required by V's
// statically typed generic return values.
pub fn add_scalar_promoted[R, T, U](a &Tensor[T], scalar U) !&Tensor[R] {
	return scalar_promoted[R, T, U](a, scalar, .add)
}

// subtract_scalar_promoted subtracts a weak numeric scalar using NumPy-style
// promotion. R must be the inferred result dtype.
pub fn subtract_scalar_promoted[R, T, U](a &Tensor[T], scalar U) !&Tensor[R] {
	return scalar_promoted[R, T, U](a, scalar, .subtract)
}

// multiply_scalar_promoted multiplies by a weak numeric scalar using NumPy-style
// promotion. R must be the inferred result dtype.
pub fn multiply_scalar_promoted[R, T, U](a &Tensor[T], scalar U) !&Tensor[R] {
	return scalar_promoted[R, T, U](a, scalar, .multiply)
}

// divide_scalar_promoted performs true division by a weak numeric scalar using
// NumPy-style promotion. R must be the inferred result dtype.
pub fn divide_scalar_promoted[R, T, U](a &Tensor[T], scalar U) !&Tensor[R] {
	return scalar_promoted[R, T, U](a, scalar, .divide)
}

// remainder_scalar_promoted returns the NumPy-style remainder of a tensor and
// weak numeric scalar. Complex operands are unsupported; R must be the inferred
// result dtype.
pub fn remainder_scalar_promoted[R, T, U](a &Tensor[T], scalar U) !&Tensor[R] {
	return scalar_promoted[R, T, U](a, scalar, .remainder)
}

// add_scalar_left_promoted adds a tensor to a weak numeric or complex scalar
// using NumPy-style promotion.
pub fn add_scalar_left_promoted[R, T, U](scalar U, a &Tensor[T]) !&Tensor[R] {
	return scalar_left_promoted[R, T, U](scalar, a, .add)
}

// subtract_scalar_left_promoted subtracts a tensor from a weak scalar using
// NumPy-style promotion.
pub fn subtract_scalar_left_promoted[R, T, U](scalar U, a &Tensor[T]) !&Tensor[R] {
	return scalar_left_promoted[R, T, U](scalar, a, .subtract)
}

// multiply_scalar_left_promoted multiplies a tensor by a weak scalar using
// NumPy-style promotion.
pub fn multiply_scalar_left_promoted[R, T, U](scalar U, a &Tensor[T]) !&Tensor[R] {
	return scalar_left_promoted[R, T, U](scalar, a, .multiply)
}

// divide_scalar_left_promoted performs NumPy-style true division with a weak
// scalar as the numerator.
pub fn divide_scalar_left_promoted[R, T, U](scalar U, a &Tensor[T]) !&Tensor[R] {
	return scalar_left_promoted[R, T, U](scalar, a, .divide)
}

// remainder_scalar_left_promoted returns the NumPy-style remainder of a weak
// scalar and tensor. Complex operands are unsupported.
pub fn remainder_scalar_left_promoted[R, T, U](scalar U, a &Tensor[T]) !&Tensor[R] {
	return scalar_left_promoted[R, T, U](scalar, a, .remainder)
}

@[inline]
fn binary_promoted[R, T, U](a &Tensor[T], b &Tensor[U], operation PromotedBinaryOperation) !&Tensor[R] {
	$if R is complex.Complex {
		if operation == .remainder {
			return error('promoted remainder does not support complex values')
		}
		left_dtype := dtype_of[T]()
		right_dtype := dtype_of[U]()
		if !is_complex_numeric_dtype(left_dtype) || !is_complex_numeric_dtype(right_dtype) {
			return error('complex promoted arithmetic requires numeric or boolean tensors')
		}
		promoted_dtype := promote_types(left_dtype, right_dtype)!
		expected := if operation == .divide && promoted_dtype != .complex128 {
			division_result_dtype(promoted_dtype)
		} else {
			promoted_dtype
		}
		if dtype_of[R]() != expected {
			return error('output dtype ${dtype_of[R]()} does not match operation result dtype ${expected}')
		}
		shape := broadcast_shapes(a.shape, b.shape)!
		left := a.broadcast_to(shape)!
		right := b.broadcast_to(shape)!
		mut result := empty[R](shape, memory: .row_major)
		left_is_singleton := a.size == 1
		right_is_singleton := b.size == 1
		left_is_flat := left.size == result.size && left.is_row_major_contiguous()
		right_is_flat := right.size == result.size && right.is_row_major_contiguous()
		for index in 0 .. result.size {
			left_offset := if left_is_flat {
				index
			} else if left_is_singleton {
				0
			} else {
				broadcast_tensor_offset(index, shape, left.strides)
			}
			right_offset := if right_is_flat {
				index
			} else if right_is_singleton {
				0
			} else {
				broadcast_tensor_offset(index, shape, right.strides)
			}
			x := cast_promoted_value[T, R](left.data.data[left_offset])
			y := cast_promoted_value[U, R](right.data.data[right_offset])
			result.data.data[index] = complex_promoted_value(x, y, operation)!
		}
		return result
	} $else $if R is $int || R is $float {
		left_dtype := dtype_of[T]()
		right_dtype := dtype_of[U]()
		left_supported := is_real_numeric_dtype(left_dtype)
			|| (operation == .divide && left_dtype == .boolean)
		right_supported := is_real_numeric_dtype(right_dtype)
			|| (operation == .divide && right_dtype == .boolean)
		if !left_supported || !right_supported {
			return error('promoted arithmetic supports only real numeric tensors and division of booleans')
		}
		promoted_dtype := promote_types(left_dtype, right_dtype)!
		expected := if operation == .divide {
			division_result_dtype(promoted_dtype)
		} else {
			promoted_dtype
		}
		if dtype_of[R]() != expected {
			return error('output dtype ${dtype_of[R]()} does not match operation result dtype ${expected}')
		}
		if operation == .divide && !is_float_dtype(dtype_of[R]()) {
			return error('promoted division requires a floating-point output dtype')
		}
		shape := broadcast_shapes(a.shape, b.shape)!
		left := a.broadcast_to(shape)!
		right := b.broadcast_to(shape)!
		mut result := empty[R](shape, memory: .row_major)
		left_is_singleton := a.size == 1
		right_is_singleton := b.size == 1
		left_is_flat := left.size == result.size && left.is_row_major_contiguous()
		right_is_flat := right.size == result.size && right.is_row_major_contiguous()
		if operation == .remainder {
			if left_is_flat && right_is_singleton {
				divisor := cast_promoted_value[U, R](b.data.data[0])
				if divisor == R(0) && is_integer_dtype(dtype_of[R]()) {
					return result
				}
				for index in 0 .. result.size {
					value := cast_promoted_value[T, R](left.data.data[index])
					result.data.data[index] = numpy_remainder(value, divisor)
				}
				return result
			}
			for index in 0 .. result.size {
				left_offset := if left_is_flat {
					index
				} else if left_is_singleton {
					0
				} else {
					broadcast_tensor_offset(index, shape, left.strides)
				}
				right_offset := if right_is_flat {
					index
				} else if right_is_singleton {
					0
				} else {
					broadcast_tensor_offset(index, shape, right.strides)
				}
				x := cast_promoted_value[T, R](left.data.data[left_offset])
				y := cast_promoted_value[U, R](right.data.data[right_offset])
				result.data.data[index] = if y == R(0) && is_integer_dtype(dtype_of[R]()) {
					R(0)
				} else {
					numpy_remainder(x, y)
				}
			}
			return result
		}
		for index in 0 .. result.size {
			left_offset := if left_is_flat {
				index
			} else if left_is_singleton {
				0
			} else {
				broadcast_tensor_offset(index, shape, left.strides)
			}
			right_offset := if right_is_flat {
				index
			} else if right_is_singleton {
				0
			} else {
				broadcast_tensor_offset(index, shape, right.strides)
			}
			x := cast_promoted_value[T, R](left.data.data[left_offset])
			y := cast_promoted_value[U, R](right.data.data[right_offset])
			result.data.data[index] = match operation {
				.add { x + y }
				.subtract { x - y }
				.multiply { x * y }
				.divide { x / y }
			}
		}
		return result
	} $else {
		return error('promoted arithmetic requires a real numeric output dtype')
	}
}

fn scalar_promoted[R, T, U](a &Tensor[T], scalar U, operation PromotedBinaryOperation) !&Tensor[R] {
	return scalar_promoted_impl[R, T, U](a, scalar, operation, false)
}

fn scalar_left_promoted[R, T, U](scalar U, a &Tensor[T], operation PromotedBinaryOperation) !&Tensor[R] {
	return scalar_promoted_impl[R, T, U](a, scalar, operation, true)
}

fn scalar_promoted_impl[R, T, U](a &Tensor[T], scalar U, operation PromotedBinaryOperation, scalar_on_left bool) !&Tensor[R] {
	$if R is complex.Complex {
		if operation == .remainder {
			return error('promoted remainder does not support complex values')
		}
		result_dtype := weak_complex_scalar_result_dtype[T, U]()!
		if dtype_of[R]() != result_dtype {
			return error('output dtype ${dtype_of[R]()} does not match weak-scalar result dtype ${result_dtype}')
		}
		left := cast_promoted_tensor_values[T, R](a)!
		right_scalar := cast_promoted_value[U, R](scalar)
		mut result := empty[R](a.shape, memory: .row_major)
		for index in 0 .. a.size {
			if scalar_on_left {
				result.data.data[index] = complex_promoted_value(right_scalar, left.data.data[index],
					operation)!
			} else {
				result.data.data[index] = complex_promoted_value(left.data.data[index],
					right_scalar, operation)!
			}
		}
		return result
	} $else $if R is $int || R is $float {
		array_dtype := dtype_of[T]()
		result_dtype := weak_scalar_result_dtype[T, U](array_dtype, scalar)!
		expected := if operation == .divide {
			division_result_dtype(result_dtype)
		} else {
			result_dtype
		}
		if dtype_of[R]() != expected {
			return error('output dtype ${dtype_of[R]()} does not match weak-scalar result dtype ${expected}')
		}
		left := cast_promoted_tensor_values[T, R](a)!
		right_scalar := cast_promoted_value[U, R](scalar)
		scalar_tensor := from_array[R]([right_scalar], [])!
		if scalar_on_left {
			return binary_promoted[R, R, R](scalar_tensor, left, operation)
		}
		return binary_promoted[R, R, R](left, scalar_tensor, operation)
	} $else {
		return error('weak-scalar promoted arithmetic requires a real numeric output dtype')
	}
}

fn weak_complex_scalar_result_dtype[T, U]() !DType {
	array_dtype := dtype_of[T]()
	scalar_dtype := dtype_of[U]()
	array_supported := array_dtype == .complex128 || is_real_numeric_dtype(array_dtype)
		|| array_dtype == .boolean
	scalar_supported := scalar_dtype == .complex128 || is_real_numeric_dtype(scalar_dtype)
		|| scalar_dtype == .boolean
	if !array_supported || !scalar_supported {
		return error('complex weak-scalar arithmetic supports numeric and boolean operands')
	}
	return .complex128
}

fn complex_promoted_value(x complex.Complex, y complex.Complex, operation PromotedBinaryOperation) !complex.Complex {
	return match operation {
		.add { x + y }
		.subtract { x - y }
		.multiply { x * y }
		.divide { x / y }
		.remainder { return error('promoted remainder does not support complex values') }
	}
}

fn weak_scalar_result_dtype[T, U](array_dtype DType, scalar U) !DType {
	if !is_real_numeric_dtype(array_dtype) {
		return error('weak-scalar promoted arithmetic supports real numeric tensors')
	}
	scalar_dtype := dtype_of[U]()
	if !is_real_numeric_dtype(scalar_dtype) && scalar_dtype != .boolean {
		return error('weak-scalar promoted arithmetic supports real numeric and boolean scalars')
	}
	if is_integer_dtype(array_dtype) {
		if is_integer_dtype(scalar_dtype) || scalar_dtype == .boolean {
			if !integer_scalar_fits_dtype[U](scalar, array_dtype) {
				return error('integer scalar is out of range for array dtype ${array_dtype}')
			}
			return array_dtype
		}
		return .float64
	}
	return array_dtype
}

fn integer_scalar_fits_dtype[U](scalar U, dtype DType) bool {
	$if U is bool {
		return true
	} $else $if U is i8 || U is i16 || U is i32 || U is i64 || U is int {
		return signed_scalar_fits_dtype(i64(scalar), dtype)
	} $else $if U is u8 || U is u16 || U is u32 || U is u64 {
		return unsigned_scalar_fits_dtype(u64(scalar), dtype)
	} $else {
		return false
	}
}

fn signed_scalar_fits_dtype(value i64, dtype DType) bool {
	return match dtype {
		.int8 { value >= -128 && value <= 127 }
		.int16 { value >= -32768 && value <= 32767 }
		.int32 { value >= -2147483648 && value <= 2147483647 }
		.int64 { true }
		.native_int {
			if sizeof(int) == 4 {
				value >= -2147483648 && value <= 2147483647
			} else {
				true
			}
		}
		.uint8, .uint16, .uint32, .uint64 {
			value >= 0 && unsigned_scalar_fits_dtype(u64(value), dtype)
		}
		else { false }
	}
}

fn unsigned_scalar_fits_dtype(value u64, dtype DType) bool {
	return match dtype {
		.int8 { value <= 127 }
		.int16 { value <= 32767 }
		.int32 { value <= 2147483647 }
		.int64 { value <= 9223372036854775807 }
		.native_int {
			if sizeof(int) == 4 {
				value <= 2147483647
			} else {
				value <= 9223372036854775807
			}
		}
		.uint8 { value <= 255 }
		.uint16 { value <= 65535 }
		.uint32 { value <= 4294967295 }
		.uint64 { true }
		else { false }
	}
}

fn cast_promoted_tensor_values[T, R](t &Tensor[T]) !&Tensor[R] {
	mut values := []R{len: t.size}
	for index in 0 .. t.size {
		values[index] = cast_promoted_value[T, R](t.get_nth[T](index))
	}
	return from_array[R](values, t.shape)
}

fn division_result_dtype(promoted_dtype DType) DType {
	return match promoted_dtype {
		.float32, .float64 { promoted_dtype }
		else { .float64 }
	}
}

fn is_float_dtype(dtype DType) bool {
	return dtype == .float32 || dtype == .float64
}

fn is_complex_numeric_dtype(dtype DType) bool {
	return dtype == .complex128 || is_real_numeric_dtype(dtype) || dtype == .boolean
}

fn is_integer_dtype(dtype DType) bool {
	return match dtype {
		.int8, .int16, .int32, .int64, .native_int, .uint8, .uint16, .uint32, .uint64 { true }
		else { false }
	}
}

@[inline]
fn numpy_remainder[T](x T, y T) T {
	$if T is $int {
		$if T is i8 || T is i16 || T is i32 || T is i64 || T is int {
			// Signed minimum % -1 overflows in machine arithmetic, although
			// NumPy's remainder result is exactly zero for every value modulo -1.
			if y == T(-1) {
				return T(0)
			}
		}
		mut result := x % y
		if result != T(0) && (result < T(0)) != (y < T(0)) {
			result += y
		}
		return result
	} $else $if T is $float {
		x_f64 := f64(x)
		y_f64 := f64(y)
		mut result := math.fmod(x_f64, y_f64)
		if result == 0.0 {
			result = math.copysign(0.0, y_f64)
		} else if math.signbit(result) != math.signbit(y_f64) {
			result += y_f64
		}
		return T(result)
	} $else {
		panic('promoted remainder requires an integer or floating-point dtype')
	}
}

@[inline]
fn cast_promoted_value[T, R](value T) R {
	$if R is complex.Complex {
		$if T is complex.Complex {
			return value
		} $else $if T is bool {
			return complex.Complex{
				re: if value { 1.0 } else { 0.0 }
				im: 0.0
			}
		} $else $if T is $int || T is $float {
			return complex.Complex{
				re: f64(value)
				im: 0.0
			}
		}
	} $else $if R is $int || R is $float {
		$if T is bool {
			return if value { R(1) } else { R(0) }
		} $else $if T is $int || T is $float {
			return R(value)
		}
	}
	panic('promoted arithmetic supports only real numeric tensors')
}

fn is_real_numeric_dtype(dtype DType) bool {
	return match dtype {
		.int8, .int16, .int32, .int64, .native_int, .uint8, .uint16, .uint32, .uint64,
		.float32, .float64 {
			true
		}
		else { false }
	}
}
