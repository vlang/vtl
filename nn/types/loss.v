module types

import vtl
import vtl.autograd

// LossFn computes a loss value using opaque wrapper pointers.
pub type LossFn = fn (loss voidptr, input voidptr, target voidptr) !voidptr

// Loss wraps a concrete loss implementation without storing a generic interface.
pub struct Loss[T] {
	ptr     voidptr
	loss_fn LossFn = unsafe { nil }
}

// loss creates a typed loss wrapper.
pub fn loss[T](ptr voidptr, loss_fn LossFn) Loss[T] {
	return Loss[T]{
		ptr:     ptr
		loss_fn: loss_fn
	}
}

// loss computes a scalar loss variable for model output and target tensors.
pub fn (loss Loss[T]) loss(input &autograd.Variable[T], target &vtl.Tensor[T]) !&autograd.Variable[T] {
	result := loss.loss_fn(loss.ptr, voidptr(input), voidptr(target))!
	return unsafe { &autograd.Variable[T](result) }
}
