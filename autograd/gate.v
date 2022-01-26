module autograd

import vtl

pub interface CacheParam {}

// Gate is an object that can cache the result of an operation,
// as well as backpropogate a payload backwards along the
// computational graph
//
// Structs that implement from this interface can add instance
// variables if additional caching is needed, and these need
// to be populated when writing the cached operation
// @todo: Make this generic once it works as expected
pub interface Gate {
	// backward(payload &Payload<T>) []&vtl.Tensor<T>
	// cache(mut result Variable<T>, args ...CacheParam)
}

// @todo: Implement this somehow :D
pub fn gate_backward<T>(gate Gate, payload &Payload<T>) []&vtl.Tensor<T> {
	return []&vtl.Tensor<T>{}
}

// @todo: Implement this somehow :D
pub fn gate_cache<T>(gate Gate, mut result Variable<T>, args ...CacheParam) {
}
