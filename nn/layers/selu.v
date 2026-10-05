module layers

import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.activation
import vtl.nn.types

pub struct SeluLayer[T] {
	output_shape []int
}

pub fn selu_layer[T](ctx &autograd.Context[T], output_shape []int) types.Layer[T] {
	layer := &SeluLayer[T]{
		output_shape: output_shape.clone()
	}
	return types.layer[T](voidptr(layer), selu_layer_output_shape_dispatch[T],
		selu_layer_variables_dispatch[T], selu_layer_forward_dispatch[T])
}

pub fn (layer &SeluLayer[T]) output_shape() []int {
	return layer.output_shape
}

pub fn (_ &SeluLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

pub fn (layer &SeluLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := internal.selu[T](input.value)
	mut result := input.context.variable(output)
	if input.requires_grad {
		gate := activation.selu_gate[T](input.value)
		gate.cache(mut result, input)!
	}
	return result
}

fn selu_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&SeluLayer[T](layer)).output_shape() }
}

fn selu_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&SeluLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn selu_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&SeluLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}
