module layers

import vtl.autograd
import vtl.nn.types

pub fn layer_output_shape[T](layer types.Layer) []int {
	match layer {
		DropoutLayer[T] {
			return layer.output_shape()
		}
		EluLayer[T] {
			return layer.output_shape()
		}
		FlattenLayer[T] {
			return layer.output_shape()
		}
		InputLayer[T] {
			return layer.output_shape()
		}
		LeakyReluLayer[T] {
			return layer.output_shape()
		}
		LinearLayer[T] {
			return layer.output_shape()
		}
		MaxPool2DLayer[T] {
			return layer.output_shape()
		}
		ReLULayer[T] {
			return layer.output_shape()
		}
		SigmoidLayer[T] {
			return layer.output_shape()
		}
		else {
			panic(@FN + ': Unknown layer type ${typeof(layer).name}')
		}
	}
}

pub fn layer_forward[T](layer types.Layer, mut input autograd.Variable[T]) !&autograd.Variable[T] {
	match layer {
		DropoutLayer[T] {
			return layer.forward[T](mut input)
		}
		EluLayer[T] {
			return layer.forward[T](mut input)
		}
		FlattenLayer[T] {
			return layer.forward[T](mut input)
		}
		InputLayer[T] {
			return layer.forward[T](mut input)
		}
		LeakyReluLayer[T] {
			return layer.forward[T](mut input)
		}
		LinearLayer[T] {
			return layer.forward[T](mut input)
		}
		MaxPool2DLayer[T] {
			return layer.forward[T](mut input)
		}
		ReLULayer[T] {
			return layer.forward[T](mut input)
		}
		SigmoidLayer[T] {
			return layer.forward[T](mut input)
		}
		else {
			return error(@FN + ': Unknown layer type ${typeof(layer).name}')
		}
	}
}
