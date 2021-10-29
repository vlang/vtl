module loss

import vtl
import vtl.autograd
import vtl.nn.internal

pub struct MseGate<T> {
pub:
	cache  &autograd.Variable<T>
	target &vtl.Tensor<T>
}

pub fn new_mse_gate<T>(cache &autograd.Variable<T>, target &vtl.Tensor<T>) &MseGate<T> {
	return &MseGate<T>{
		cache: cache
		target: target
	}
}

pub fn (g &MseGate<T>) backward<T>(payload &autograd.Payload<T>) []&vtl.Tensor<T> {
	gradient := payload.variable.grad

	grad := gradient.value
	norm := vtl.divide_scalar(vtl.multiply_scalar(grad, T(2)), T(gradient.size))

	mut ret := vtl.new_tensor_like<T>(g.cache)
	// mut iter := vtl.iterators<T>([g.cache, g.target])
	// for {
	// 	vals, pos := vtl.iterators_next<T>(mut iter) or { break }
	// 	val := norm
	// 	ret.data.set<T>(pos, val)
	// }

	return [res]
}

pub fn (g &MseGate<T>) cache<T>(mut result autograd.Variable<T>, args ...autograd.CacheParam) {
	result.grad = vtl.zeros_like<T>(result.value)
	result.requires_grad = true

	register<T>('MSE', g, result, ...args)
}
