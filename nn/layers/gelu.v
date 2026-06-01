module layers

import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.activation
import vtl.nn.types

// GELULayer applies the Gaussian Error Linear Unit (GELU) activation.
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Approximation used: 0.7978845608 * (x + 0.044715 * x^3)  (lower-cost tanh approximation)
pub struct GELULayer[T] {
	output_shape []int
}

// gelu_layer exposes this operation as part of the public API.
pub fn gelu_layer[T](ctx &autograd.Context[T], output_shape []int) types.Layer[T] {
	layer := &GELULayer[T]{
		output_shape: output_shape.clone()
	}
	return types.layer[T](voidptr(layer), gelu_layer_output_shape_dispatch[T],
		gelu_layer_variables_dispatch[T], gelu_layer_forward_dispatch[T])
}

// output_shape exposes this operation as part of the public API.
pub fn (layer &GELULayer[T]) output_shape() []int {
	return layer.output_shape
}

// variables exposes this operation as part of the public API.
pub fn (_ &GELULayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

// forward exposes this operation as part of the public API.
pub fn (layer &GELULayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := internal.gelu[T](input.value)
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := activation.gelu_gate[T](input.value)
		gate.cache(mut result, input)!
	}
	return result
}

fn gelu_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&GELULayer[T](layer)).output_shape() }
}

fn gelu_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&GELULayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn gelu_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&GELULayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}
