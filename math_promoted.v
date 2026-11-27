module vtl

import math
import math.complex

pub enum PromotedArithmetic {
	add
	subtract
	multiply
	divide
	floor_divide
	fmod
	remainder
	power
	float_power
}

// PromotedTensor is the dynamically selected result of binary_promoted_auto.
pub type PromotedTensor = &Tensor[bool]
	| &Tensor[i8]
	| &Tensor[i16]
	| &Tensor[i32]
	| &Tensor[i64]
	| &Tensor[int]
	| &Tensor[u8]
	| &Tensor[u16]
	| &Tensor[u32]
	| &Tensor[u64]
	| &Tensor[f32]
	| &Tensor[f64]
	| &Tensor[complex.Complex]

// promoted_tensor_dtype returns the runtime-selected dtype of a promoted tensor.
pub fn promoted_tensor_dtype(t PromotedTensor) DType {
	return match t {
		&Tensor[bool] { .boolean }
		&Tensor[i8] { .int8 }
		&Tensor[i16] { .int16 }
		&Tensor[i32] { .int32 }
		&Tensor[i64] { .int64 }
		&Tensor[int] { .native_int }
		&Tensor[u8] { .uint8 }
		&Tensor[u16] { .uint16 }
		&Tensor[u32] { .uint32 }
		&Tensor[u64] { .uint64 }
		&Tensor[f32] { .float32 }
		&Tensor[f64] { .float64 }
		&Tensor[complex.Complex] { .complex128 }
	}
}

// binary_promoted_auto selects the NumPy-style result dtype at runtime.
// The returned sum type preserves the concrete tensor dtype for callers that
// need statically typed values; use promoted_tensor_dtype or match on the result.
pub fn binary_promoted_auto[T, U](a &Tensor[T], b &Tensor[U], operation PromotedArithmetic) !PromotedTensor {
	result_dtype := promoted_operation_dtype(dtype_of[T](), dtype_of[U](), operation)!
	return match result_dtype {
		.boolean { PromotedTensor(binary_promoted[bool, T, U](a, b, operation)!) }
		.int8 { PromotedTensor(binary_promoted[i8, T, U](a, b, operation)!) }
		.int16 { PromotedTensor(binary_promoted[i16, T, U](a, b, operation)!) }
		.int32 { PromotedTensor(binary_promoted[i32, T, U](a, b, operation)!) }
		.int64 { PromotedTensor(binary_promoted[i64, T, U](a, b, operation)!) }
		.native_int { PromotedTensor(binary_promoted[int, T, U](a, b, operation)!) }
		.uint8 { PromotedTensor(binary_promoted[u8, T, U](a, b, operation)!) }
		.uint16 { PromotedTensor(binary_promoted[u16, T, U](a, b, operation)!) }
		.uint32 { PromotedTensor(binary_promoted[u32, T, U](a, b, operation)!) }
		.uint64 { PromotedTensor(binary_promoted[u64, T, U](a, b, operation)!) }
		.float32 { PromotedTensor(binary_promoted[f32, T, U](a, b, operation)!) }
		.float64 { PromotedTensor(binary_promoted[f64, T, U](a, b, operation)!) }
		.complex128 {
			PromotedTensor(binary_promoted[complex.Complex, T, U](a, b, operation)!)
		}
		.string { error('promoted arithmetic does not support string tensors') }
	}
}

// add_promoted adds real numeric or boolean tensors after converting both
// operands to the explicitly selected NumPy-style result dtype. The output
// type R must match the operation's NumPy result dtype.
pub fn add_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_promoted[R, T, U](a, b, .add)
}

// subtract_promoted subtracts real numeric or boolean tensors after converting
// both operands to the explicitly selected NumPy-style result dtype. Boolean
// subtraction is unsupported.
pub fn subtract_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_promoted[R, T, U](a, b, .subtract)
}

// multiply_promoted multiplies real numeric or boolean tensors after converting
// both operands to the explicitly selected NumPy-style result dtype.
pub fn multiply_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_promoted[R, T, U](a, b, .multiply)
}

// pow_promoted computes elementwise powers after broadcasting. Numeric pairs
// use promote_types(dtype_of[T](), dtype_of[U]()); boolean pairs return int8,
// matching NumPy. Integer powers keep exact-width wrapping arithmetic and
// reject negative integer exponents. Complex pairs use the principal branch;
// 64-bit integers cannot be combined with a floating output because conversion
// could lose integer precision.
pub fn pow_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_promoted[R, T, U](a, b, .power)
}

// float_power_promoted computes elementwise powers as float64 for real numeric
// or boolean inputs, allowing negative integer exponents like NumPy.
pub fn float_power_promoted[T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[f64] {
	return binary_promoted[f64, T, U](a, b, .float_power)
}

// float_power_scalar_promoted raises tensor values to a weak real numeric or
// boolean scalar and always returns float64.
pub fn float_power_scalar_promoted[T, U](a &Tensor[T], scalar U) !&Tensor[f64] {
	return float_power_scalar_impl[T, U](a, scalar, false)
}

// float_power_scalar_left_promoted raises a weak real numeric or boolean scalar
// to tensor values and always returns float64.
pub fn float_power_scalar_left_promoted[T, U](scalar U, a &Tensor[T]) !&Tensor[f64] {
	return float_power_scalar_impl[T, U](a, scalar, true)
}

// divide_promoted performs NumPy-style true division for numeric tensors and
// booleans. Complex inputs produce complex128; real inputs produce float32 only
// when the promoted input dtype is float32, and float64 otherwise.
pub fn divide_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_promoted[R, T, U](a, b, .divide)
}

// floor_divide_promoted performs NumPy-style floor division after promoting
// real numeric tensor operands to R. Complex inputs are unsupported.
pub fn floor_divide_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_promoted[R, T, U](a, b, .floor_divide)
}

// fmod_promoted returns the remainder with the sign of the dividend after
// NumPy-style dtype promotion and broadcasting. Boolean and complex inputs are
// unsupported; R must match promote_types(dtype_of[T](), dtype_of[U]()).
pub fn fmod_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_promoted[R, T, U](a, b, .fmod)
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

// degrees_promoted converts radians to degrees with NumPy-style float output
// promotion.
pub fn degrees_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.degrees)
}

// radians_promoted converts degrees to radians with NumPy-style float output
// promotion.
pub fn radians_promoted[R, T](t &Tensor[T]) !&Tensor[R] {
	return unary_real_ufunc_promoted[R, T](t, math.radians)
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

// nextafter_promoted steps one representable output value from x toward y.
// The f32 path uses f32 nextafter semantics instead of rounding an f64 step.
pub fn nextafter_promoted[R, T, U](x &Tensor[T], y &Tensor[U]) !&Tensor[R] {
	$if R is f32 {
		return binary_real_ufunc_promoted[R, T, U](x, y, nextafter32_as_f64, 'nextafter')
	} $else {
		return binary_real_ufunc_promoted[R, T, U](x, y, math.nextafter, 'nextafter')
	}
}

// heaviside_promoted returns 0 for negative x1, 1 for positive x1, and x2
// for zero x1, using NumPy-style float output promotion.
pub fn heaviside_promoted[R, T, U](x1 &Tensor[T], x2 &Tensor[U]) !&Tensor[R] {
	return binary_real_ufunc_promoted[R, T, U](x1, x2, heaviside_value, 'heaviside')
}

// hypot_promoted computes elementwise hypotenuses with NumPy-style float
// output promotion and broadcasting.
pub fn hypot_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_real_ufunc_promoted[R, T, U](a, b, math.hypot, 'hypot')
}

// maximum_promoted returns the broadcasted maximum after NumPy-style dtype
// promotion. NaNs propagate, and equal values follow NumPy's operand ordering.
pub fn maximum_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_extrema_promoted[R, T, U](a, b, true, false, 'maximum')
}

// minimum_promoted returns the broadcasted minimum after NumPy-style dtype
// promotion. NaNs propagate, and equal values follow NumPy's operand ordering.
pub fn minimum_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_extrema_promoted[R, T, U](a, b, false, false, 'minimum')
}

// fmax_promoted returns the broadcasted maximum while ignoring one NaN operand.
// The output dtype follows NumPy-style promotion.
pub fn fmax_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_extrema_promoted[R, T, U](a, b, true, true, 'fmax')
}

// fmin_promoted returns the broadcasted minimum while ignoring one NaN operand.
// The output dtype follows NumPy-style promotion.
pub fn fmin_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_extrema_promoted[R, T, U](a, b, false, true, 'fmin')
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

fn binary_extrema_promoted[R, T, U](a &Tensor[T], b &Tensor[U], want_max bool, ignore_nan bool, name string) !&Tensor[R] {
	left_dtype := dtype_of[T]()
	right_dtype := dtype_of[U]()
	if !is_complex_numeric_dtype(left_dtype) || !is_complex_numeric_dtype(right_dtype) {
		return error('promoted ${name} supports only numeric and boolean tensors')
	}
	expected_dtype := promote_types(left_dtype, right_dtype)!
	if dtype_of[R]() != expected_dtype {
		return error('output dtype ${dtype_of[R]()} does not match ${name} result dtype ${expected_dtype}')
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
		result.data.data[index] = numpy_extreme[R](x, y, want_max, ignore_nan)
	}
	return result
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

fn nextafter32_as_f64(x f64, y f64) f64 {
	return f64(math.nextafter32(f32(x), f32(y)))
}

fn heaviside_value(x1 f64, x2 f64) f64 {
	if math.is_nan(x1) {
		return x1
	}
	if x1 < 0.0 {
		return 0.0
	}
	if x1 > 0.0 {
		return 1.0
	}
	return x2
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

// pow_scalar_promoted raises each tensor value to a weak numeric or complex
// scalar exponent using NumPy-style promotion. R must match the inferred output
// dtype; negative integer exponents are rejected.
pub fn pow_scalar_promoted[R, T, U](a &Tensor[T], scalar U) !&Tensor[R] {
	return scalar_promoted[R, T, U](a, scalar, .power)
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

// floor_divide_scalar_promoted performs NumPy-style floor division by a weak
// real numeric scalar using the inferred promoted dtype R.
pub fn floor_divide_scalar_promoted[R, T, U](a &Tensor[T], scalar U) !&Tensor[R] {
	return scalar_promoted[R, T, U](a, scalar, .floor_divide)
}

// fmod_scalar_promoted applies fmod with a weak real numeric scalar.
pub fn fmod_scalar_promoted[R, T, U](a &Tensor[T], scalar U) !&Tensor[R] {
	return scalar_promoted[R, T, U](a, scalar, .fmod)
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

// pow_scalar_left_promoted raises a weak numeric or complex scalar to each
// tensor value using NumPy-style promotion. R must match the inferred output
// dtype; negative integer exponents are rejected.
pub fn pow_scalar_left_promoted[R, T, U](scalar U, a &Tensor[T]) !&Tensor[R] {
	return scalar_left_promoted[R, T, U](scalar, a, .power)
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

// floor_divide_scalar_left_promoted performs NumPy-style floor division with a
// weak real numeric scalar as the numerator using the inferred dtype R.
pub fn floor_divide_scalar_left_promoted[R, T, U](scalar U, a &Tensor[T]) !&Tensor[R] {
	return scalar_left_promoted[R, T, U](scalar, a, .floor_divide)
}

// fmod_scalar_left_promoted applies fmod with a weak real numeric scalar as the
// dividend.
pub fn fmod_scalar_left_promoted[R, T, U](scalar U, a &Tensor[T]) !&Tensor[R] {
	return scalar_left_promoted[R, T, U](scalar, a, .fmod)
}

fn promoted_operation_dtype(left DType, right DType, operation PromotedArithmetic) !DType {
	promoted := promote_types(left, right)!
	if operation == .float_power {
		return .float64
	}
	if operation == .divide {
		return if promoted == .complex128 { .complex128 } else { division_result_dtype(promoted) }
	}
	boolean_pair_integer_result := left == .boolean && right == .boolean
		&& (operation == .power || operation == .floor_divide || operation == .fmod
			|| operation == .remainder)
	if boolean_pair_integer_result {
		return .int8
	}
	return promoted
}

@[inline]
fn binary_promoted[R, T, U](a &Tensor[T], b &Tensor[U], operation PromotedArithmetic) !&Tensor[R] {
	$if R is bool {
		if dtype_of[T]() != .boolean || dtype_of[U]() != .boolean {
			return error('boolean promoted output requires two boolean tensor inputs')
		}
		if operation == .subtract {
			return error('promoted boolean subtraction is not supported')
		}
		if operation != .add && operation != .multiply {
			return error('promoted boolean output supports only addition and multiplication')
		}
		if promoted_operation_dtype(.boolean, .boolean, operation)! != .boolean {
			return error('output dtype ${dtype_of[R]()} does not match operation result dtype')
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
			result.data.data[index] = if operation == .add { x || y } else { x && y }
		}
		return result
	} $else $if R is complex.Complex {
		if operation == .float_power {
			return error('promoted float_power does not support complex values')
		}
		if operation == .remainder {
			return error('promoted remainder does not support complex values')
		}
		if operation == .floor_divide {
			return error('promoted floor division does not support complex values')
		}
		if operation == .fmod {
			return error('promoted fmod does not support complex values')
		}
		left_dtype := dtype_of[T]()
		right_dtype := dtype_of[U]()
		if !is_complex_numeric_dtype(left_dtype) || !is_complex_numeric_dtype(right_dtype) {
			return error('complex promoted arithmetic requires numeric or boolean tensors')
		}
		expected := promoted_operation_dtype(left_dtype, right_dtype, operation)!
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
		left_supported := is_real_numeric_dtype(left_dtype) || left_dtype == .boolean
		right_supported := is_real_numeric_dtype(right_dtype) || right_dtype == .boolean
		if !left_supported || !right_supported {
			return error('promoted arithmetic supports only real numeric and boolean tensors')
		}
		expected := promoted_operation_dtype(left_dtype, right_dtype, operation)!
		if dtype_of[R]() != expected {
			return error('output dtype ${dtype_of[R]()} does not match operation result dtype ${expected}')
		}
		if operation == .power && is_float_dtype(expected)
			&& ((is_integer_dtype(left_dtype) && dtype_bits(left_dtype) >= 64)
				|| (is_integer_dtype(right_dtype) && dtype_bits(right_dtype) >= 64)) {
			return error('promoted power rejects 64-bit integer inputs with floating output to avoid precision loss')
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
			if operation == .power {
				result.data.data[index] = numpy_power(x, y)!
				continue
			}
			if operation == .float_power {
				result.data.data[index] = R(math.pow(f64(x), f64(y)))
				continue
			}
			result.data.data[index] = match operation {
				.add { x + y }
				.subtract { x - y }
				.multiply { x * y }
				.divide { x / y }
				.floor_divide { numpy_floor_divide[R](x, y) }
				.fmod { numpy_fmod[R](x, y) }
				.remainder { return error('promoted remainder dispatch failed') }
				.power { return error('promoted power dispatch failed') }
				.float_power { return error('promoted float_power dispatch failed') }
			}
		}
		return result
	} $else {
		return error('promoted arithmetic requires a real numeric output dtype')
	}
}

fn scalar_promoted[R, T, U](a &Tensor[T], scalar U, operation PromotedArithmetic) !&Tensor[R] {
	return scalar_promoted_impl[R, T, U](a, scalar, operation, false)
}

fn scalar_left_promoted[R, T, U](scalar U, a &Tensor[T], operation PromotedArithmetic) !&Tensor[R] {
	return scalar_promoted_impl[R, T, U](a, scalar, operation, true)
}

fn float_power_scalar_impl[T, U](a &Tensor[T], scalar U, scalar_on_left bool) !&Tensor[f64] {
	array_dtype := dtype_of[T]()
	scalar_dtype := dtype_of[U]()
	if !(is_real_numeric_dtype(array_dtype) || array_dtype == .boolean)
		|| !(is_real_numeric_dtype(scalar_dtype) || scalar_dtype == .boolean) {
		return error('promoted float_power supports only real numeric and boolean scalars and tensors')
	}
	left := cast_promoted_tensor_values[T, f64](a)!
	right := from_array[f64]([cast_promoted_value[U, f64](scalar)], [])!
	if scalar_on_left {
		return binary_promoted[f64, f64, f64](right, left, .float_power)
	}
	return binary_promoted[f64, f64, f64](left, right, .float_power)
}

fn scalar_promoted_impl[R, T, U](a &Tensor[T], scalar U, operation PromotedArithmetic, scalar_on_left bool) !&Tensor[R] {
	$if R is complex.Complex {
		if operation == .remainder {
			return error('promoted remainder does not support complex values')
		}
		if operation == .floor_divide {
			return error('promoted floor division does not support complex values')
		}
		if operation == .fmod {
			return error('promoted fmod does not support complex values')
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
	} $else $if R is bool {
		if dtype_of[T]() != .boolean || dtype_of[U]() != .boolean {
			return error('boolean weak-scalar output requires a boolean tensor and scalar')
		}
		if operation != .add && operation != .multiply {
			return error('promoted boolean scalar output supports only addition and multiplication')
		}
		if weak_scalar_result_dtype[T, U](dtype_of[T](), scalar)! != .boolean {
			return error('output dtype ${dtype_of[R]()} does not match weak-scalar result dtype')
		}
		left := cast_promoted_tensor_values[T, R](a)!
		right_scalar := cast_promoted_value[U, R](scalar)
		scalar_tensor := from_array[R]([right_scalar], [])!
		if scalar_on_left {
			return binary_promoted[R, R, R](scalar_tensor, left, operation)
		}
		return binary_promoted[R, R, R](left, scalar_tensor, operation)
	} $else $if R is $int || R is $float {
		array_dtype := dtype_of[T]()
		result_dtype := weak_scalar_result_dtype[T, U](array_dtype, scalar)!
		boolean_pair_integer_result := array_dtype == .boolean && dtype_of[U]() == .boolean
			&& (operation == .power || operation == .floor_divide || operation == .fmod
				|| operation == .remainder)
		expected := if operation == .divide {
			division_result_dtype(result_dtype)
		} else if boolean_pair_integer_result {
			DType.int8
		} else {
			result_dtype
		}
		if dtype_of[R]() != expected {
			return error('output dtype ${dtype_of[R]()} does not match weak-scalar result dtype ${expected}')
		}
		if operation == .power && is_float_dtype(expected)
			&& ((is_integer_dtype(array_dtype) && dtype_bits(array_dtype) >= 64)
				|| (is_integer_dtype(dtype_of[U]()) && dtype_bits(dtype_of[U]()) >= 64)) {
			return error('promoted power rejects 64-bit integer inputs with floating output to avoid precision loss')
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

fn complex_promoted_value(x complex.Complex, y complex.Complex, operation PromotedArithmetic) !complex.Complex {
	return match operation {
		.add { x + y }
		.subtract { x - y }
		.multiply { x * y }
		.divide { x / y }
		.power { complex_power_value(x, y) }
		.floor_divide { return error('promoted floor division does not support complex values') }
		.fmod { return error('promoted fmod does not support complex values') }
		.remainder { return error('promoted remainder does not support complex values') }
		.float_power { return error('promoted float_power does not support complex values') }
	}
}

@[inline]
fn numpy_power[T](base T, exponent T) !T {
	$if T is bool {
		return if exponent { base } else { true }
	} $else $if T is $int {
		if is_signed_integer_dtype(dtype_of[T]()) && exponent < T(0) {
			return error('negative exponents are not allowed for integer power')
		}
		mut result := T(1)
		mut remaining := exponent
		mut factor := base
		for remaining > T(0) {
			if remaining % T(2) != T(0) {
				result *= factor
			}
			remaining /= T(2)
			if remaining > T(0) {
				factor *= factor
			}
		}
		return result
	} $else $if T is $float {
		return T(math.pow(f64(base), f64(exponent)))
	} $else {
		return error('promoted power supports only boolean, real integer, and floating-point values')
	}
}

fn weak_scalar_result_dtype[T, U](array_dtype DType, scalar U) !DType {
	scalar_dtype := dtype_of[U]()
	if array_dtype == .boolean {
		if scalar_dtype == .boolean || is_real_numeric_dtype(scalar_dtype) {
			return scalar_dtype
		}
		return error('weak-scalar promoted arithmetic supports real numeric and boolean scalars')
	}
	if !is_real_numeric_dtype(array_dtype) {
		return error('weak-scalar promoted arithmetic supports real numeric or boolean tensors')
	}
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
fn numpy_fmod[T](x T, y T) T {
	$if T is $int {
		if y == T(0) {
			return T(0)
		}
		$if T is i8 || T is i16 || T is i32 || T is i64 || T is int {
			if y == T(-1) {
				return T(0)
			}
		}
		return x % y
	} $else $if T is $float {
		return T(math.fmod(f64(x), f64(y)))
	} $else {
		panic('promoted fmod requires an integer or floating-point dtype')
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
	} $else $if R is bool {
		$if T is bool {
			return value
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
