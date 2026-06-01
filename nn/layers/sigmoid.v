module layers

import vtl
import vtl.autograd
import vtl.nn.internal
import vtl.nn.gates.activation
import vtl.nn.types

// SigmoidLayer is a layer that applies the sigmoid function to its input.
pub struct SigmoidLayer[T] {
	output_shape []int
}

pub fn sigmoid_layer[T](ctx &autograd.Context[T], output_shape []int) types.Layer[T] {
	return types.Layer[T](&SigmoidLayer[T]{
		output_shape: output_shape.clone()
	})
}

pub fn (layer &SigmoidLayer[T]) output_shape() []int {
	return layer.output_shape
}

pub fn (_ &SigmoidLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

pub fn (layer &SigmoidLayer[T]) forward(input &autograd.Variable[T]) !&autograd.Variable[T] {
	mut output := &vtl.Tensor[T](unsafe { nil })
	$if sizeof(T) == 4 {
		out_f32 := sigmoid_forward_f32(unsafe { &vtl.Tensor[f32](input.value) })!
		output = unsafe { &vtl.Tensor[T](out_f32) }
	} $else {
		output = internal.sigmoid[T](input.value)
	}
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := activation.sigmoid_gate[T](input.value)
		gate.cache(mut result, input)!
	}
	return result
}
