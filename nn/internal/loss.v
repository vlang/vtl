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

// softmax_cross_entropy_backward computes the gradient of the SCE loss w.r.t.
// the logits. The gradient for each logit is:
//   upstream * (softmax(logit_i) - target_i) / batch_size
//
// This is the standard gradient derived from the log-softmax formulation
// (equivalent to Arraymancer's implementation).
pub fn softmax_cross_entropy_backward[T](gradient &vtl.Tensor[T], cache &vtl.Tensor[T], target &vtl.Tensor[T]) ![]&vtl.Tensor[T] {
	batch_size := cache.shape[0]
	n_classes := cache.shape[1]
	upstream := f64(gradient.get([0]))
	mut ret_data := []f64{len: batch_size * n_classes}
	for i in 0 .. batch_size {
		// Compute softmax for sample i
		mut max_logit := f64(cache.get([i, 0]))
		for c in 1 .. n_classes {
			v := f64(cache.get([i, c]))
			if v > max_logit {
				max_logit = v
			}
		}
		mut sum_exp := f64(0)
		for c in 0 .. n_classes {
			sum_exp += math.exp(f64(cache.get([i, c])) - max_logit)
		}
		for c in 0 .. n_classes {
			sm := math.exp(f64(cache.get([i, c])) - max_logit) / sum_exp
			tgt := f64(target.get([i, c]))
			ret_data[i * n_classes + c] = upstream * (sm - tgt) / f64(batch_size)
		}
	}
	ret := vtl.from_array(ret_data.map(vtl.cast[T](it)), [batch_size, n_classes])!
	return [ret]
}
