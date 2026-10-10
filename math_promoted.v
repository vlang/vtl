module vtl

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

// remainder_promoted returns the element-wise NumPy remainder of integer
// tensors, after dtype promotion and broadcasting. A zero divisor produces 0.
pub fn remainder_promoted[R, T, U](a &Tensor[T], b &Tensor[U]) !&Tensor[R] {
	return binary_promoted[R, T, U](a, b, .remainder)
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
		if operation == .remainder && (!is_integer_dtype(left_dtype) || !is_integer_dtype(right_dtype)) {
			return error('promoted remainder supports only integer tensors')
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
				if divisor == R(0) {
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
				result.data.data[index] = if y == R(0) { R(0) } else { numpy_remainder(x, y) }
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
		mut result := x % y
		if result != T(0) && (result < T(0)) != (y < T(0)) {
			result += y
		}
		return result
	} $else {
		panic('promoted remainder requires an integer dtype')
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
