module types

// import vtl.nn.layers

// Layer is a generic interface for a neural network layer.
pub interface Layer {
	output_shape() []int
	// variables() []&autograd.Variable<T>
	// forward(input &autograd.Variable<T>)
}
