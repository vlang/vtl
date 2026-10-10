module vtl

enum PromotedBinaryOperation {
	add
	subtract
	multiply
	divide
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
		for index in 0 .. result.size {
			x := cast_promoted_value[T, R](left.get_nth(index))
			y := cast_promoted_value[U, R](right.get_nth(index))
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

fn cast_promoted_value[T, R](value T) R {
	$if T is bool || T is $int || T is $float {
		$if R is $int || R is $float {
			return cast[R](td[T](value))
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
