module types

import vtl.autograd

// Layer is a generic interface for a neural network layer.
pub interface Layer[T] {
	output_shape() []int
	variables() []&autograd.Variable[T]
	forward(mut input autograd.Variable[T]) !&autograd.Variable[T]
}
