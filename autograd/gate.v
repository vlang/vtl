module autograd

import vtl

// CacheParam defines a public behavior contract for this module.
pub interface CacheParam {}

// Gate is a generic interface for autograd operations.
// Any struct that implements backward for a given T satisfies Gate[T].
// This allows both core math gates (AddGate, MatMulGate, …) and
// higher-level nn gates (LinearGate, ReLUGate, …) to participate in
// backpropagation without circular imports.
pub interface Gate[T] {
	backward(payload &Payload[T]) ![]&vtl.Tensor[T]
}

// BackwardFn dispatches a stored gate without relying on a generic interface
// inside Node[T]. Both parameters and return values stay opaque at the graph
// storage boundary to avoid V generic interface specialization drift between
// Payload[f32] and Payload[f64].
pub type BackwardFn = fn (gate voidptr, payload voidptr) ![]voidptr

// tensor_ptrs_to_voidptrs converts typed gradient tensor pointers at dispatch
// boundaries. The typed gate implementations remain the source of truth.
pub fn tensor_ptrs_to_voidptrs[T](tensors []&vtl.Tensor[T]) []voidptr {
	mut result := []voidptr{cap: tensors.len}
	for tensor in tensors {
		result << voidptr(tensor)
	}
	return result
}

// gate_backward is kept for backwards compatibility but now delegates
// directly through the Gate[T] interface instead of a manual match.
pub fn gate_backward[T](gate Gate[T], payload &Payload[T]) ![]&vtl.Tensor[T] {
	return gate.backward(payload)
}
