module types

import vtl
import vtl.autograd

pub interface Loss[T] {
	loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T]
}
