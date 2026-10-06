module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.activation
import vtl.nn.types

pub struct SoftplusLayer[T] {
	output_shape []int
}

pub fn softplus_layer[T](ctx &autograd.Context[T], output_shape []int) types.Layer[T] {
	layer := &SoftplusLayer[T]{
		output_shape: output_shape.clone()
	}
	return types.layer[T](voidptr(layer), softplus_layer_output_shape_dispatch[T],
		softplus_layer_variables_dispatch[T], softplus_layer_forward_dispatch[T])
}

pub fn (layer &SoftplusLayer[T]) output_shape() []int {
	return layer.output_shape
}

pub fn (_ &SoftplusLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

pub fn (layer &SoftplusLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	mut output := &vtl.Tensor[T](unsafe { nil })
	$if T is f32 {
		out_f32 := softplus_forward_f32(unsafe { &vtl.Tensor[f32](input.value) })!
		output = unsafe { &vtl.Tensor[T](out_f32) }
	} $else {
		output = internal.softplus[T](input.value)
	}
	mut result := input.context.variable(output)
	if input.requires_grad {
		gate := activation.softplus_gate[T](input.value)
		gate.cache(mut result, input)!
	}
	return result
}

fn softplus_layer_output_shape_dispatch[T](layer voidptr) []int {
	return unsafe { (&SoftplusLayer[T](layer)).output_shape() }
}

fn softplus_layer_variables_dispatch[T](layer voidptr) []voidptr {
	vars := unsafe { (&SoftplusLayer[T](layer)).variables() }
	return types.variable_ptrs_to_voidptrs[T](vars)
}

fn softplus_layer_forward_dispatch[T](layer voidptr, input voidptr) !voidptr {
	typed_input := unsafe { &autograd.Variable[T](input) }
	result := unsafe { (&SoftplusLayer[T](layer)).forward(typed_input)! }
	return voidptr(result)
}
