module vtl

import math
import math.complex

// real returns the real components of a complex tensor as a float64 tensor.
pub fn real(input &Tensor[complex.Complex]) !&Tensor[f64] {
	mut values := []f64{len: input.size}
	for i in 0 .. input.size {
		values[i] = input.get_nth(i).re
	}
	return from_array[f64](values, input.shape)
}

// imag returns the imaginary components of a complex tensor as a float64 tensor.
pub fn imag(input &Tensor[complex.Complex]) !&Tensor[f64] {
	mut values := []f64{len: input.size}
	for i in 0 .. input.size {
		values[i] = input.get_nth(i).im
	}
	return from_array[f64](values, input.shape)
}

// conj returns the complex conjugate of every element in a complex tensor.
pub fn conj(input &Tensor[complex.Complex]) !&Tensor[complex.Complex] {
	mut values := []complex.Complex{len: input.size}
	for i in 0 .. input.size {
		value := input.get_nth(i)
		values[i] = complex.Complex{
			re: value.re
			im: -value.im
		}
	}
	return from_array[complex.Complex](values, input.shape)
}

// absolute returns the magnitude of every complex element as a float64 tensor.
// math.hypot scales its inputs to avoid unnecessary overflow and underflow.
pub fn absolute(input &Tensor[complex.Complex]) !&Tensor[f64] {
	mut values := []f64{len: input.size}
	for i in 0 .. input.size {
		value := input.get_nth(i)
		values[i] = math.hypot(value.re, value.im)
	}
	return from_array[f64](values, input.shape)
}

// abs is the NumPy-style alias for absolute.
pub fn abs(input &Tensor[complex.Complex]) !&Tensor[f64] {
	return absolute(input)
}
