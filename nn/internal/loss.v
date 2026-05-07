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
	// Reshape target from [batch_size] → [batch_size, 1] so shapes align with cache [batch_size, 1].
	// Without this, broadcasting [4,1] × [4] produces [4,4] instead of [4,1].
	target_2d := target.reshape([batch_size, 1])!
	// Iterate over cache (the logits, shape [batch_size, 1]).
	// vals[0] = logit, vals[1] = target label, vals[2] = upstream gradient (scalar, broadcast from [1]).
	// Gradient of sigmoid-BCE w.r.t. logit: upstream * (sigmoid(logit) - target) / batch_size
	ret := cache.nmap([target_2d, gradient], fn [batch_size] [T](vals []T, i []int) T {
		logit := vals[0]
		tgt := vals[1]
		upstream := vals[2]
		sigma := vtl.cast[T](1) / (vtl.cast[T](1) + vtl.cast[T](math.exp(-f64(logit))))
		return upstream * (sigma - tgt) / vtl.cast[T](batch_size)
	})!
	return [ret]
}

pub fn softmax_cross_entropy_backward[T](gradient &vtl.Tensor[T], cache &vtl.Tensor[T], target &vtl.Tensor[T]) ![]&vtl.Tensor[T] {
	// batch_size := cache.shape[0]
	mut ret := vtl.tensor_like[T](cache)
	// TODO: implement softmax_cross_entropy_backward
	return [ret]
}
