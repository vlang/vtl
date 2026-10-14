module vtl

import math

// signbit returns true for values with a negative sign bit, including -0 and
// negative NaNs. Signed integers are true when negative; unsigned and boolean
// values are always false.
pub fn (t &Tensor[T]) signbit[T]() !&Tensor[bool] {
	$if T is bool || T is int || T is i8 || T is i16 || T is i32 || T is i64 || T is u8 || T is u16 || T is u32 || T is u64 || T is f32 || T is f64 {
		mut result := empty[bool](t.shape.clone(), memory: .row_major)
		for flat_index in 0 .. t.size {
			$if T is f32 || T is f64 {
				result.data.data[flat_index] = math.signbit(f64(t.get_nth(flat_index)))
			} $else $if T is int || T is i8 || T is i16 || T is i32 || T is i64 {
				result.data.data[flat_index] = t.get_nth(flat_index) < 0
			} $else {
				result.data.data[flat_index] = false
			}
		}
		return result
	} $else {
		return error('signbit requires a real numeric tensor')
	}
}

fn logaddexp_value(x f64, y f64) f64 {
	if math.is_nan(x) || math.is_nan(y) {
		return math.nan()
	}
	if x == y && math.is_inf(x, 0) {
		return x
	}
	maximum := math.max(x, y)
	if math.is_inf(maximum, 1) {
		return maximum
	}
	minimum := math.min(x, y)
	return maximum + math.log1p(math.exp(minimum - maximum))
}

fn logaddexp2_value(x f64, y f64) f64 {
	if math.is_nan(x) || math.is_nan(y) {
		return math.nan()
	}
	if x == y && math.is_inf(x, 0) {
		return x
	}
	maximum := math.max(x, y)
	if math.is_inf(maximum, 1) {
		return maximum
	}
	minimum := math.min(x, y)
	return maximum + math.log2(1.0 + math.exp2(minimum - maximum))
}

// copysign returns the magnitude of `magnitude` with the sign bit of `sign`.
// Both f32 and f64 inputs broadcast to a common output shape.
pub fn copysign[T](magnitude &Tensor[T], sign &Tensor[T]) !&Tensor[T] {
	$if T is f32 || T is f64 {
		shape := broadcast_shapes(magnitude.shape, sign.shape)!
		absolute := magnitude.broadcast_to(shape)!
		signs := sign.broadcast_to(shape)!
		mut result := empty[T](shape, memory: .row_major)
		for flat_index in 0 .. result.size {
			magnitude_offset := broadcast_tensor_offset(flat_index, shape, absolute.strides)
			sign_offset := broadcast_tensor_offset(flat_index, shape, signs.strides)
			result.data.data[flat_index] = T(math.copysign(f64(absolute.data.data[magnitude_offset]),
				f64(signs.data.data[sign_offset])))
		}
		return result
	} $else {
		return error('copysign requires f32 or f64 tensors of the same type')
	}
}

// logaddexp computes log(exp(a) + exp(b)) elementwise using a stable form.
// The f32/f64 operands broadcast to a common output shape.
pub fn logaddexp[T](a &Tensor[T], b &Tensor[T]) !&Tensor[T] {
	$if T is f32 || T is f64 {
		shape := broadcast_shapes(a.shape, b.shape)!
		left := a.broadcast_to(shape)!
		right := b.broadcast_to(shape)!
		mut result := empty[T](shape, memory: .row_major)
		for flat_index in 0 .. result.size {
			left_offset := broadcast_tensor_offset(flat_index, shape, left.strides)
			right_offset := broadcast_tensor_offset(flat_index, shape, right.strides)
			result.data.data[flat_index] = T(logaddexp_value(f64(left.data.data[left_offset]),
				f64(right.data.data[right_offset])))
		}
		return result
	} $else {
		return error('logaddexp requires f32 or f64 tensors of the same type')
	}
}

// logaddexp2 computes log2(2^a + 2^b) elementwise using a stable form.
// The f32/f64 operands broadcast to a common output shape.
pub fn logaddexp2[T](a &Tensor[T], b &Tensor[T]) !&Tensor[T] {
	$if T is f32 || T is f64 {
		shape := broadcast_shapes(a.shape, b.shape)!
		left := a.broadcast_to(shape)!
		right := b.broadcast_to(shape)!
		mut result := empty[T](shape, memory: .row_major)
		for flat_index in 0 .. result.size {
			left_offset := broadcast_tensor_offset(flat_index, shape, left.strides)
			right_offset := broadcast_tensor_offset(flat_index, shape, right.strides)
			result.data.data[flat_index] = T(logaddexp2_value(f64(left.data.data[left_offset]),
				f64(right.data.data[right_offset])))
		}
		return result
	} $else {
		return error('logaddexp2 requires f32 or f64 tensors of the same type')
	}
}
