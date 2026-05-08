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

// bce computes binary cross entropy between input and target.
// input values should be in (0,1) — caller is responsible for clamping/sigmoid.
@[inline]
pub fn bce[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	one := vtl.cast[T](1)
	epsilon := vtl.cast[T](1e-7)
	clamped := input.map(fn [one, epsilon] [T](val T, i []int) T {
		if val <= epsilon { return epsilon }
		if val >= one - epsilon { return one - epsilon }
		return val
	})
	sum := clamped.nreduce([target], vtl.cast[T](0), fn [one] [T](acc T, vals []T, i []int) T {
		x := vals[0]
		y := vals[1]
		return acc + (-y * vtl.cast[T](math.log(vtl.cast[f64](vals[0]))) -
			(one - y) * vtl.cast[T](math.log(vtl.cast[f64](one - vals[0]))))
	})!
	size := input.size()
	return vtl.from_1d([sum / vtl.cast[T](size)])
}

// bce_backward computes the gradient of BCE w.r.t. the raw logits (before sigmoid).
// If from_logits=true, the upstream gradient is multiplied by the sigmoid derivative.
pub fn bce_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], target &vtl.Tensor[T], from_logits bool) !&vtl.Tensor[T] {
	batch_size := input.shape[0]
	upstream := f64(gradient.get([0]))

	// Compute sigmoid of input
	mut sigmoid_vals := []f64{len: input.size()}
	for i in 0 .. input.size() {
		x := f64(input.get_nth(i))
		sigmoid_vals[i] = 1.0 / (1.0 + math.exp(-x))
	}

	mut ret_data := []f64{len: input.size()}
	for i in 0 .. input.size() {
		p := sigmoid_vals[i]
		t := f64(target.get_nth(i))
		mut grad := (p - t) / vtl.cast[T](batch_size)
		if from_logits {
			grad = grad * vtl.cast[T](p * (1.0 - p))
		}
		ret_data[i] = upstream * vtl.cast[f64](grad)
	}

	ret_shape := input.shape.clone()
	ret := vtl.from_array(ret_data.map(vtl.cast[T](it)), ret_shape)!
	return ret
}

// huber computes the Huber loss (smooth L1)
pub fn huber[T](input &vtl.Tensor[T], target &vtl.Tensor[T], delta T) !&vtl.Tensor[T] {
	one := vtl.cast[T](1)
	half := vtl.cast[T](0.5)
	mut total := vtl.cast[T](0)
	for i in 0 .. input.size() {
		diff := math.abs(vtl.cast[f64](input.get_nth(i) - target.get_nth(i)))
		if diff > vtl.cast[f64](delta) {
			total += delta * vtl.cast[T](diff - 0.5 * vtl.cast[f64](delta))
		} else {
			diff_f := vtl.cast[f64](input.get_nth(i) - target.get_nth(i))
			total += half * vtl.cast[T](diff_f * diff_f)
		}
	}
	return vtl.from_1d([total / vtl.cast[T](input.size())])
}

// huber_backward computes the gradient of Huber loss w.r.t. input
pub fn huber_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], target &vtl.Tensor[T], delta T) !&vtl.Tensor[T] {
	batch_size := input.size()
	upstream := f64(gradient.get([0]))
	mut ret_data := []f64{len: batch_size}
	for i in 0 .. batch_size {
		diff := f64(input.get_nth(i) - target.get_nth(i))
		abs_diff := math.abs(diff)
		mut grad := f64(0)
		if abs_diff > vtl.cast[f64](delta) {
			grad = if diff > 0 { 1.0 } else { -1.0 } * vtl.cast[f64](delta)
		} else {
			grad = diff
		}
		ret_data[i] = upstream * grad / f64(batch_size)
	}
	ret := vtl.from_array(ret_data.map(vtl.cast[T](it)), input.shape.clone())!
	return ret
}

// nll computes Negative Log Likelihood loss (assumes input is log-probs)
// Target is one-hot or class probabilities; we compute -sum(target * log(input))
pub fn nll[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	one := vtl.cast[T](1)
	mut total := vtl.cast[T](0)
	for i in 0 .. input.size() {
		log_prob := f64(input.get_nth(i))
		tgt := f64(target.get_nth(i))
		total += vtl.cast[T](-tgt * log_prob)
	}
	return vtl.from_1d([total / vtl.cast[T](input.size())])
}

// nll_backward computes gradient of NLL w.r.t. input (log-probs)
pub fn nll_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	batch_size := input.size()
	upstream := f64(gradient.get([0]))
	mut ret_data := []f64{len: batch_size}
	for i in 0 .. batch_size {
		tgt := f64(target.get_nth(i))
		ret_data[i] = upstream * (-tgt / f64(vtl.cast[f64](input.get_nth(i)) + 1e-9))
	}
	ret := vtl.from_array(ret_data.map(vtl.cast[T](it)), input.shape.clone())!
	return ret
}

// kl_div computes KL Divergence loss D_KL(P || Q) = sum(P * log(P/Q))
// input: log-probs Q, target: probabilities P
pub fn kl_div[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	mut total := vtl.cast[T](0)
	for i in 0 .. input.size() {
		p := f64(target.get_nth(i))
		q := f64(input.get_nth(i))
		if p > 0 {
			total += vtl.cast[T](p * math.log(p / math.exp(q)))
		}
	}
	return vtl.from_1d([total / vtl.cast[T](input.size())])
}

// kl_div_backward computes gradient of KL Divergence w.r.t. input (log-probs Q)
pub fn kl_div_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	batch_size := input.size()
	upstream := f64(gradient.get([0]))
	mut ret_data := []f64{len: batch_size}
	for i in 0 .. batch_size {
		p := f64(target.get_nth(i))
		ret_data[i] = upstream * (-p / f64(vtl.cast[f64](input.get_nth(i)) + 1e-9)) / f64(batch_size)
	}
	ret := vtl.from_array(ret_data.map(vtl.cast[T](it)), input.shape.clone())!
	return ret
}

// cross_entropy computes CrossEntropyLoss (LogSoftmax + NLL combined).
// This is the standard cross-entropy for multi-class classification.
// input: [batch, n_classes] raw logits
// target: [batch, n_classes] one-hot targets
pub fn cross_entropy[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	// Forward: log_softmax(x) = x - log(sum(exp(x - max)))
	//          loss = -mean(sum(target * log_softmax(x)))
	batch_size := input.shape[0]
	n_classes := input.shape[1]
	mut total_loss := f64(0)
	for i in 0 .. batch_size {
		mut max_logit := f64(input.get([i, 0]))
		for c in 1 .. n_classes {
			v := f64(input.get([i, c]))
			if v > max_logit { max_logit = v }
		}
		mut sum_exp := f64(0)
		for c in 0 .. n_classes {
			sum_exp += math.exp(f64(input.get([i, c])) - max_logit)
		}
		log_sum_exp := math.log(sum_exp) + max_logit
		mut dot := f64(0)
		for c in 0 .. n_classes {
			dot += f64(target.get([i, c])) * (f64(input.get([i, c])) - log_sum_exp)
		}
		total_loss -= dot
	}
	return vtl.from_1d([vtl.cast[T](total_loss / f64(batch_size))])
}

// cross_entropy_backward computes gradient of CrossEntropyLoss w.r.t. input logits.
// dL/dx_i = (softmax(x)_i - target_i) / batch_size
pub fn cross_entropy_backward[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	batch_size := input.shape[0]
	n_classes := input.shape[1]
	upstream := f64(gradient.get([0]))
	mut ret_data := []f64{len: batch_size * n_classes}
	for i in 0 .. batch_size {
		mut max_logit := f64(input.get([i, 0]))
		for c in 1 .. n_classes {
			v := f64(input.get([i, c]))
			if v > max_logit { max_logit = v }
		}
		mut sum_exp := f64(0)
		for c in 0 .. n_classes {
			sum_exp += math.exp(f64(input.get([i, c])) - max_logit)
		}
		for c in 0 .. n_classes {
			sm := math.exp(f64(input.get([i, c])) - max_logit) / sum_exp
			tgt := f64(target.get([i, c]))
			ret_data[i * n_classes + c] = upstream * (sm - tgt) / f64(batch_size)
		}
	}
	return vtl.from_array(ret_data.map(vtl.cast[T](it)), [batch_size, n_classes])
}
