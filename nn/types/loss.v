module types

import vtl
import vtl.autograd

// Loss is a generic interface for loss functions.
pub interface Loss {
	// loss(input &autograd.Variable<T>, target &vtl.Tensor<T>) &autograd.Variable<T>
}
