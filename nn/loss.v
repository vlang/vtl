module nn

import vtl
import vtl.autograd

// Loss is a generic interface for loss functions.
pub interface Loss<T> {
	loss(input &autograd.Variable<T>, target &vtl.Tensor<T>) &autograd.Variable<T>
}
