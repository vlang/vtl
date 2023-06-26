module internal

import math
import vtl

pub fn mse_backward[T](gradient &vtl.Tensor[T], cache &vtl.Tensor[T], target &vtl.Tensor[T]) ![]&vtl.Tensor[T] {
	ret := gradient.nmap([cache, target], fn [gradient] [T](vals []T, i []int) T {
		return vals[0] * (vals[1] - vals[2]) / vtl.cast[T](gradient.size())
	})!
	return [ret]
}

pub fn sigmoid_cross_entropy_backward[T](gradient &vtl.Tensor[T], cache &vtl.Tensor[T], target &vtl.Tensor[T]) ![]&vtl.Tensor[T] {
	batch_size := cache.shape[0]
	ret := gradient.nmap([cache, target], fn [batch_size] [T](vals []T, i []int) T {
		return vals[0] * (vtl.cast[T](1) / vtl.cast[T](1) + vtl.cast[T](math.exp(-f64(vals[1]))) - vals[1]) / vtl.cast[T](batch_size)
	})!
	return [ret]
}

pub fn softmax_cross_entropy_backward[T](gradient &vtl.Tensor[T], cache &vtl.Tensor[T], target &vtl.Tensor[T]) ![]&vtl.Tensor[T] {
	// batch_size := cache.shape[0]
	mut ret := vtl.tensor_like[T](cache)
	// TODO: implement softmax_cross_entropy_backward
	return [ret]
}
