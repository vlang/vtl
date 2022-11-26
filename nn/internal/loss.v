module internal

import math
import vtl

pub fn mse_backward[T](gradient &vtl.Tensor[T], cache &vtl.Tensor[T], target &vtl.Tensor[T]) ?[]&vtl.Tensor[T] {
	dup := gradient.add[T](gradient)?
	norm := dup.divide_scalar(vtl.cast[T](gradient.size))?
	subs := cache.subtract(target)?
	return [norm.multiply(subs)?]
}

pub fn sigmoid_cross_entropy_backward[T](gradient &vtl.Tensor[T], cache &vtl.Tensor[T], target &vtl.Tensor[T]) ?[]&vtl.Tensor[T] {
	batch_size := cache.shape[0]
	mut iters, shape := gradient.iterators[T]([cache, target])?
	mut ret := vtl.tensor_like_with_shape[T](cache, shape)
	for {
		vals, i := iters.next() or { break }
		val := vals[0] * (vtl.cast[T](1) / vtl.cast[T](1) + vtl.cast[T](math.exp(-f64(vals[1]))) - vals[1]) / vtl.cast[T](batch_size)
		ret.set(i, val)
	}
	return [ret]
}

pub fn softmax_cross_entropy_backward[T](gradient &vtl.Tensor[T], cache &vtl.Tensor[T], target &vtl.Tensor[T]) ?[]&vtl.Tensor[T] {
	// batch_size := cache.shape[0]
	mut ret := vtl.tensor_like[T](cache)
	// @todo: implement softmax_cross_entropy_backward
	return [ret]
}
