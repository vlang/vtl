module vtl

import math
import math.complex

enum ComplexUnaryOperation {
	exp
	log
	sqrt
	sin
	cos
}

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

// exp applies the complex exponential to every complex tensor element.
pub fn exp(input &Tensor[complex.Complex]) !&Tensor[complex.Complex] {
	return map_complex_unary(input, .exp)
}

// log applies the principal complex natural logarithm to every element.
pub fn log(input &Tensor[complex.Complex]) !&Tensor[complex.Complex] {
	return map_complex_unary(input, .log)
}

// sqrt applies the principal complex square root to every element.
pub fn sqrt(input &Tensor[complex.Complex]) !&Tensor[complex.Complex] {
	return map_complex_unary(input, .sqrt)
}

// sin applies the complex sine to every element.
pub fn sin(input &Tensor[complex.Complex]) !&Tensor[complex.Complex] {
	return map_complex_unary(input, .sin)
}

// cos applies the complex cosine to every element.
pub fn cos(input &Tensor[complex.Complex]) !&Tensor[complex.Complex] {
	return map_complex_unary(input, .cos)
}

fn map_complex_unary(input &Tensor[complex.Complex], operation ComplexUnaryOperation) !&Tensor[complex.Complex] {
	mut values := []complex.Complex{len: input.size}
	for i in 0 .. input.size {
		value := input.get_nth(i)
		values[i] = match operation {
			.exp { value.exp() }
			.log { value.ln() }
			.sqrt { complex_sqrt(value) }
			.sin { value.sin() }
			.cos { value.cos() }
		}
	}
	return from_array[complex.Complex](values, input.shape)
}

fn complex_sqrt(value complex.Complex) complex.Complex {
	magnitude := math.hypot(value.re, value.im)
	if value.re >= 0 {
		real_part := math.sqrt(magnitude * 0.5 + value.re * 0.5)
		imaginary := if real_part == 0 { 0.0 } else { (value.im * 0.5) / real_part }
		return complex.Complex{
			re: real_part
			im: imaginary
		}
	}
	imaginary_magnitude := math.sqrt(magnitude * 0.5 - value.re * 0.5)
	real_part := if imaginary_magnitude == 0 {
		0.0
	} else {
		math.abs(value.im * 0.5) / imaginary_magnitude
	}
	imaginary := if value.im < 0 { -imaginary_magnitude } else { imaginary_magnitude }
	return complex.Complex{
		re: real_part
		im: imaginary
	}
}
