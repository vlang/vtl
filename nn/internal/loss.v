module internal

import math
import vtl

pub fn mse_backward<T>(gradient &vtl.Tensor<T>, cache &vtl.Tensor<T>, target &vtl.Tensor<T>) []&vtl.Tensor<T> {
	dup := gradient.add<T>(gradient)
	norm := dup.divide_scalar(vtl.new_t<T>(gradient.size))
	subs := cache.substract(target)
	return [norm.multiply(subs)]
}

pub fn sigmoid_cross_entropy_backward<T>(gradient &vtl.Tensor<T>, cache &vtl.Tensor<T>, target &vtl.Tensor<T>) []&vtl.Tensor<T> {
	batch_size := cache.shape[0]
	mut iter, shape := gradient.iterators<T>([cache, target])
	mut ret := vtl.new_tensor_like_with_shape<T>(cache, shape)
	for {
		vals, i := vtl.iterators_next<T>(mut iter) or { break }
		val := vals[0] * (vtl.new_t<T>(1) / vtl.new_t<T>(1) + vtl.new_t<T>(math.exp(-f64(vals[1]))) - vals[1]) / vtl.new_t<T>(batch_size)
		ret.set(i, val)
	}
	return [ret]
}

pub fn softmax_cross_entropy_backward<T>(gradient &vtl.Tensor<T>, cache &vtl.Tensor<T>, target &vtl.Tensor<T>) []&vtl.Tensor<T> {
	// batch_size := cache.shape[0]
	mut ret := vtl.new_tensor_like<T>(cache)
	// @todo: implement softmax_cross_entropy_backward
	return [ret]
}
