module layers

import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.activation
import vtl.nn.types

pub struct HardSwishLayer[T] {
	output_shape []int
}

pub fn hardswish_layer[T](ctx &autograd.Context[T], output_shape []int) types.Layer[T] {
	layer := &HardSwishLayer[T]{
		output_shape: output_shape.clone()
	}
	return types.layer[T](voidptr(layer), hardswish_layer_output_shape_dispatch[T],
		hardswish_layer_variables_dispatch[T], hardswish_layer_forward_dispatch[T])
}

pub fn (layer &HardSwishLayer[T]) output_shape() []int {
	return layer.output_shape
}

pub fn (_ &HardSwishLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

pub fn (layer &HardSwishLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	output := internal.hardswish[T](input.value)
	mut result := input.context.variable(output)
	if input.requires_grad {
		gate := activation.hardswish_gate[T](input.value)
		gate.cache(mut result, input)!
	}
	return result
}

fn hardswish_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&HardSwishLayer[T](layer)).output_shape() }
}

fn hardswish_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&HardSwishLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn hardswish_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&HardSwishLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}
