module vtl

import math

// modf returns fractional and integral parts, in that order, matching NumPy.
// Only f32 and f64 tensors are supported.
pub fn (t &Tensor[T]) modf[T]() !(&Tensor[T], &Tensor[T]) {
	$if T is f32 || T is f64 {
		mut fractional := empty[T](t.shape.clone(), memory: .row_major)
		mut integral := empty[T](t.shape.clone(), memory: .row_major)
		for flat_index in 0 .. t.size {
			integer_part, fractional_part := math.modf(f64(t.get_nth(flat_index)))
			fractional.data.data[flat_index] = T(fractional_part)
			integral.data.data[flat_index] = T(integer_part)
		}
		return fractional, integral
	} $else {
		return error('modf requires an f32 or f64 tensor')
	}
}

// frexp decomposes each f32/f64 value into a mantissa and base-2 exponent.
// The mantissa preserves the input floating-point type; exponents use int.
pub fn (t &Tensor[T]) frexp[T]() !(&Tensor[T], &Tensor[int]) {
	$if T is f32 || T is f64 {
		mut mantissa := empty[T](t.shape.clone(), memory: .row_major)
		mut exponent := empty[int](t.shape.clone(), memory: .row_major)
		for flat_index in 0 .. t.size {
			fraction, power := math.frexp(f64(t.get_nth(flat_index)))
			mantissa.data.data[flat_index] = T(fraction)
			exponent.data.data[flat_index] = power
		}
		return mantissa, exponent
	} $else {
		return error('frexp requires an f32 or f64 tensor')
	}
}

// frexp_promoted decomposes real numeric or boolean values into a floating
// mantissa and an integer base-2 exponent. f32 inputs preserve f32 mantissas;
// all other supported inputs produce f64 mantissas.
pub fn frexp_promoted[R, T](t &Tensor[T]) !(&Tensor[R], &Tensor[int]) {
	input_dtype := dtype_of[T]()
	if !is_real_numeric_dtype(input_dtype) && input_dtype != .boolean {
		return error('frexp requires a real numeric or boolean tensor')
	}
	expected_dtype := if input_dtype == .float32 { DType.float32 } else { DType.float64 }
	if dtype_of[R]() != expected_dtype {
		return error('mantissa dtype ${dtype_of[R]()} does not match frexp result dtype ${expected_dtype}')
	}
	mut mantissa := empty[R](t.shape.clone(), memory: .row_major)
	mut exponent := empty[int](t.shape.clone(), memory: .row_major)
	for flat_index in 0 .. t.size {
		value := cast_promoted_value[T, f64](t.get_nth(flat_index))
		fraction, power := math.frexp(value)
		mantissa.data.data[flat_index] = cast_promoted_value[f64, R](fraction)
		exponent.data.data[flat_index] = power
	}
	return mantissa, exponent
}

// ldexp scales each floating-point mantissa by 2 raised to the corresponding
// integer exponent. The operands broadcast to a common shape.
pub fn ldexp[T](mantissa &Tensor[T], exponent &Tensor[int]) !&Tensor[T] {
	$if T is f32 || T is f64 {
		shape := broadcast_shapes(mantissa.shape, exponent.shape)!
		mantissas := mantissa.broadcast_to(shape)!
		exponents := exponent.broadcast_to(shape)!
		mut result := empty[T](shape, memory: .row_major)
		for flat_index in 0 .. result.size {
			mantissa_offset := broadcast_tensor_offset(flat_index, shape, mantissas.strides)
			exponent_offset := broadcast_tensor_offset(flat_index, shape, exponents.strides)
			value := math.ldexp(f64(mantissas.data.data[mantissa_offset]),
				exponents.data.data[exponent_offset])
			result.data.data[flat_index] = T(value)
		}
		return result
	} $else {
		return error('ldexp requires an f32 or f64 mantissa tensor')
	}
}

// ldexp_promoted applies NumPy-style dtype selection to real mantissas and
// integer exponents. Floating mantissas preserve f32/f64; integer and boolean
// mantissas produce f64.
pub fn ldexp_promoted[R, T, U](mantissa &Tensor[T], exponent &Tensor[U]) !&Tensor[R] {
	mantissa_dtype := dtype_of[T]()
	exponent_dtype := dtype_of[U]()
	if !is_real_numeric_dtype(mantissa_dtype) && mantissa_dtype != .boolean {
		return error('ldexp requires a real numeric or boolean mantissa tensor')
	}
	if !is_integer_dtype(exponent_dtype) && exponent_dtype != .boolean {
		return error('ldexp requires an integer or boolean exponent tensor')
	}
	expected_dtype := if mantissa_dtype == .float32 { DType.float32 } else { DType.float64 }
	if dtype_of[R]() != expected_dtype {
		return error('output dtype ${dtype_of[R]()} does not match ldexp result dtype ${expected_dtype}')
	}
	shape := broadcast_shapes(mantissa.shape, exponent.shape)!
	mantissas := mantissa.broadcast_to(shape)!
	exponents := exponent.broadcast_to(shape)!
	mut result := empty[R](shape, memory: .row_major)
	for flat_index in 0 .. result.size {
		mantissa_offset := broadcast_tensor_offset(flat_index, shape, mantissas.strides)
		exponent_offset := broadcast_tensor_offset(flat_index, shape, exponents.strides)
		mantissa_value := cast_promoted_value[T, f64](mantissas.data.data[mantissa_offset])
		exponent_value := ldexp_exponent_to_int[U](exponents.data.data[exponent_offset])
		result.data.data[flat_index] = cast_promoted_value[f64, R](math.ldexp(mantissa_value, exponent_value))
	}
	return result
}

@[inline]
fn ldexp_exponent_to_int[T](value T) int {
	$if T is bool {
		return if value { 1 } else { 0 }
	} $else $if T is $int {
		return int(value)
	} $else {
		panic('ldexp requires an integer or boolean exponent tensor')
	}
}

// ldexp_scalar scales a floating-point mantissa tensor by one integer exponent.
pub fn ldexp_scalar[T](mantissa &Tensor[T], exponent int) !&Tensor[T] {
	exponent_tensor := from_array[int]([exponent], [])!
	return ldexp[T](mantissa, exponent_tensor)
}

// ldexp_scalar_left scales one floating-point mantissa by an exponent tensor.
pub fn ldexp_scalar_left[T](mantissa T, exponent &Tensor[int]) !&Tensor[T] {
	mantissa_tensor := from_array[T]([mantissa], [])!
	return ldexp[T](mantissa_tensor, exponent)
}
