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

// gate_backward is kept for backwards compatibility but now delegates
// directly through the Gate[T] interface instead of a manual match.
pub fn gate_backward[T](gate Gate[T], payload &Payload[T]) ![]&vtl.Tensor[T] {
	return gate.backward(payload)
}
