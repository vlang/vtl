module types

import vtl
import vtl.autograd

// Loss defines a public behavior contract for this module.
pub interface Loss[T] {
	loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T]
}
