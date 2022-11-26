module layers

import vtl
import vtl.autograd
import vtl.nn.gates.layers
import vtl.nn.types
import arrays

// FlattenLayer is a layer
pub struct FlattenLayer[T] {
	shape []int
}

pub fn flatten_layer[T](ctx &autograd.Context[T], shape []int) types.Layer {
	return types.Layer(&FlattenLayer[T]{
		shape: shape.clone()
	})
}

pub fn (layer &FlattenLayer[T]) output_shape() []int {
	product := arrays.fold(layer.shape, 1, fn (a int, b int) int {
		return a * b
	})
	return [product]
}

pub fn (_ &FlattenLayer[T]) variables() []&autograd.Variable[T] {
	return []&autograd.Variable[T]{}
}

pub fn (layer &FlattenLayer[T]) forward(mut input autograd.Variable[T]) ?&autograd.Variable[T] {
	output := input.value.reshape([input.value.shape[0], -1])?
	mut result := input.context.variable(output)

	if input.requires_grad {
		gate := layers.flatten_gate[T](result, layer.shape)
		gate.cache(mut result, input)?
	}
	return result
}
