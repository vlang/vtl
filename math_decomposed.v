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
