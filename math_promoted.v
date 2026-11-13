module vtl

import math

enum PromotedBinaryOperation {
	add
	subtract
	multiply
	divide
	remainder
}

// add_promoted adds real numeric tensors after converting both operands to
// the explicitly selected NumPy-style promoted dtype. The output type R must
// match promote_types(dtype_of[T](), dtype_of[U]()).
pub fn add_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_promoted[R, T, U](a, b, .add)
}

// subtract_promoted subtracts real numeric tensors after converting both
// operands to the explicitly selected NumPy-style promoted dtype. The output
// type R must match promote_types(dtype_of[T](), dtype_of[U]()).
pub fn subtract_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_promoted[R, T, U](a, b, .subtract)
}

// multiply_promoted multiplies real numeric tensors after converting both
// operands to the explicitly selected NumPy-style promoted dtype. The output
// type R must match promote_types(dtype_of[T](), dtype_of[U]()).
pub fn multiply_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_promoted[R, T, U](a, b, .multiply)
}

// divide_promoted performs NumPy-style true division for real numeric tensors
// and booleans. The output dtype is float32 only when the promoted input dtype
// is float32; otherwise it is float64. Inputs are broadcast before division.
pub fn divide_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_promoted[R, T, U](a, b, .divide)
}

// remainder_promoted returns the element-wise NumPy remainder of real numeric
// tensors after dtype promotion and broadcasting. Integer zero divisors produce
// zero; floating-point zero divisors produce NaN.
pub fn remainder_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_promoted[R, T, U](a, b, .remainder)
}

// add_scalar_promoted adds a weak numeric scalar using NumPy-style promotion.
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
// weak numeric scalar. R must be the inferred result dtype.
pub fn remainder_scalar_promoted[R, T, U](a &Tensor[T], scalar U) !&Tensor[R] {
	return scalar_promoted[R, T, U](a, scalar, .remainder)
}

@[inline]
fn binary_promoted[R, T, U](a &Tensor[T], b &Tensor[U], operation PromotedBinaryOperation) !&Tensor[R] {
	$if R is $int || R is $float {
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
	$if R is $int || R is $float {
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
		right := from_array[R]([right_scalar], [])!
		return binary_promoted[R, R, R](left, right, operation)
	} $else {
		return error('weak-scalar promoted arithmetic requires a real numeric output dtype')
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
	$if R is $int || R is $float {
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
