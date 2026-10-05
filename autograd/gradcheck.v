module autograd

import math
import vtl

// grad_check compares the analytical gradient with central finite differences
// for every element in input. The forward function must build a fresh result
// from input on each call. For non-scalar results, the checked objective is the
// sum of all output elements, matching backprop's all-ones initial gradient.
//
// The input's gradient is replaced by the analytical gradient produced by this
// check. The input value is restored before the function returns.
pub fn grad_check[T](input &Variable[T], forward fn (&Variable[T]) !&Variable[T], eps f64, tolerance f64) !bool {
	if !gradcheck_is_float[T]() {
		return error('grad_check: input elements must use f32 or f64')
	}
	if eps <= 0.0 || !math.is_finite(eps) || tolerance < 0.0 || !math.is_finite(tolerance) {
		return error('grad_check: eps must be positive and finite, and tolerance must be non-negative and finite')
	}
	if input.value.size == 0 {
		return error('grad_check: input tensor must not be empty')
	}
	if input.context.nodes.len != 0 {
		return error('grad_check: input context must have an empty graph before checking')
	}

	mut input_value := input.value
	original_values := []T{len: input.value.size}
	for i in 0 .. input.value.size {
		original_values[i] = input.value.get_nth(i)
	}
	input.grad = vtl.zeros_like[T](input.value)

	mut analytical_output := forward(input) or {
		input.context.nodes = []&Node[T]{}
		return error('grad_check: forward evaluation failed: ${err}')
	}
	if analytical_output.context != input.context || analytical_output.value.size == 0 {
		input.context.nodes = []&Node[T]{}
		return error('grad_check: forward function must return a non-empty value from the input context')
	}
	analytical_output.backprop() or {
		input.context.nodes = []&Node[T]{}
		return error('grad_check: backpropagation failed: ${err}')
	}

	for i in 0 .. input.value.size {
		original := f64(original_values[i])
		input_value.set_nth(i, vtl.cast[T](original + eps))
		plus_output := forward(input) or {
			restore_gradcheck_input[T](mut input_value, original_values)
			input.context.nodes = []&Node[T]{}
			return error('grad_check: positive perturbation failed: ${err}')
		}
		if plus_output.context != input.context || plus_output.value.size == 0 {
			restore_gradcheck_input[T](mut input_value, original_values)
			input.context.nodes = []&Node[T]{}
			return error('grad_check: forward function must return a non-empty value from the input context')
		}
		plus := tensor_sum_f64[T](plus_output.value)
		input.context.nodes = []&Node[T]{}

		input_value.set_nth(i, vtl.cast[T](original - eps))
		minus_output := forward(input) or {
			restore_gradcheck_input[T](mut input_value, original_values)
			input.context.nodes = []&Node[T]{}
			return error('grad_check: negative perturbation failed: ${err}')
		}
		if minus_output.context != input.context || minus_output.value.size == 0 {
			restore_gradcheck_input[T](mut input_value, original_values)
			input.context.nodes = []&Node[T]{}
			return error('grad_check: forward function must return a non-empty value from the input context')
		}
		minus := tensor_sum_f64[T](minus_output.value)
		input.context.nodes = []&Node[T]{}
		numerical := (plus - minus) / (2.0 * eps)
		analytic := f64(input.grad.get_nth(i))
		if math.abs(analytic - numerical) > tolerance * max_f64(1.0, math.abs(analytic),
			math.abs(numerical)) {
			restore_gradcheck_input[T](mut input_value, original_values)
			return false
		}
	}

	restore_gradcheck_input[T](mut input_value, original_values)
	return true
}

fn gradcheck_is_float[T]() bool {
	$if T is f32 || T is f64 {
		return true
	} $else {
		return false
	}
}

fn tensor_sum_f64[T](tensor &vtl.Tensor[T]) f64 {
	mut result := 0.0
	for i in 0 .. tensor.size {
		result += f64(tensor.get_nth(i))
	}
	return result
}

fn restore_gradcheck_input[T](mut input_value &vtl.Tensor[T], values []T) {
	for i, value in values {
		input_value.set_nth(i, value)
	}
}

fn max_f64(a f64, b f64, c f64) f64 {
	return if a > b {
		if a > c { a } else { c }
	} else {
		if b > c { b } else { c }
	}
}
