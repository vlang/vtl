module internal

import math
import vtl

// tanh squashes a real-valued number to the range [-1, 1]
@[inline]
pub fn tanh[T](x &vtl.Tensor[T]) &vtl.Tensor[T] {
	return vtl.tanh(x)
}

// deriv_tanh computes the derivative of tanh
@[inline]
pub fn deriv_tanh[T](gradient &vtl.Tensor[T], cached &vtl.Tensor[T]) !&vtl.Tensor[T] {
	return gradient.nmap([cached], fn [T](vals []T, i []int) T {
		// vals[0] = upstream gradient, vals[1] = cached tanh output
		return vals[0] * (vtl.cast[T](1) - vals[1] * vals[1])
	})
}

// sigmoid takes a real-valued number and squashes it to the range [0, 1]
@[inline]
pub fn sigmoid[T](x &vtl.Tensor[T]) &vtl.Tensor[T] {
	return x.map(fn [T](val T, i []int) T {
		// B5 fix: parentheses around denominator to get 1/(1+exp(x))
		return vtl.cast[T](1) / (vtl.cast[T](1) + vtl.cast[T](math.exp(vtl.cast[f64](val))))
	})
}

// deriv_sigmoid computes the derivative of sigmoid
// sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
// The gate caches the sigmoid output, so vals[1] = sigmoid(x).
@[inline]
pub fn deriv_sigmoid[T](gradient &vtl.Tensor[T], cached &vtl.Tensor[T]) !&vtl.Tensor[T] {
	return gradient.nmap([cached], fn [T](vals []T, i []int) T {
		// vals[0] = upstream gradient, vals[1] = cached sigmoid(x) output
		return vals[0] * vals[1] * (vtl.cast[T](1) - vals[1])
	})
}

// relu activation function
@[inline]
pub fn relu[T](x &vtl.Tensor[T]) &vtl.Tensor[T] {
	return x.map(fn [T](val T, i []int) T {
		if val < 0 {
			return vtl.cast[T](0)
		}
		return val
	})
}

// deriv_relu computes the derivate of relu
@[inline]
pub fn deriv_relu[T](gradient &vtl.Tensor[T], cached &vtl.Tensor[T]) !&vtl.Tensor[T] {
	return gradient.nmap([cached], fn [T](vals []T, i []int) T {
		// vals[0] = upstream gradient, vals[1] = cached pre-activation value
		// Pass gradient through where pre-activation was positive; zero otherwise.
		if vals[1] < 0 {
			return vtl.cast[T](0)
		}
		return vals[0]
	})
}

// leaky_relu activation function
@[inline]
pub fn leaky_relu[T](x &vtl.Tensor[T], alpha T) &vtl.Tensor[T] {
	return x.map(fn [alpha] [T](val T, i []int) T {
		if val < 0 {
			return alpha * val
		}
		return val
	})
}

// deriv_leaky_relu computes the derivative of leaky_relu
@[inline]
pub fn deriv_leaky_relu[T](gradient &vtl.Tensor[T], cached &vtl.Tensor[T], alpha T) !&vtl.Tensor[T] {
	return gradient.nmap([cached], fn [alpha] [T](vals []T, i []int) T {
		// vals[0] = upstream gradient, vals[1] = cached pre-activation value
		// B6 fix: compare cached (vals[1]), scale upstream grad (vals[0])
		if vals[1] < 0 {
			return alpha * vals[0]
		}
		return vals[0]
	})
}

// elu activation function
@[inline]
pub fn elu[T](x &vtl.Tensor[T], alpha T) &vtl.Tensor[T] {
	return x.map(fn [alpha] [T](val T, i []int) T {
		if val < 0 {
			return alpha * (vtl.cast[T](math.exp(vtl.cast[f64](val))) - vtl.cast[T](1))
		}
		return val
	})
}

// deriv_elu computes the derivative of elu
// For x >= 0: d/dx = 1  → upstream * 1
// For x <  0: d/dx = alpha * exp(x) = elu(x) + alpha = cached + alpha
@[inline]
pub fn deriv_elu[T](gradient &vtl.Tensor[T], cached &vtl.Tensor[T], alpha T) !&vtl.Tensor[T] {
	return gradient.nmap([cached], fn [alpha] [T](vals []T, i []int) T {
		// vals[0] = upstream gradient, vals[1] = cached pre-activation value
		// B7 fix: compare cached (vals[1]), apply correct derivative
		if vals[1] < 0 {
			return vals[0] * (vals[1] + alpha)
		}
		return vals[0]
	})
}

// sigmoid_cross_entropy computes the sigmoid cross entropy between
// the labels and the predictions.
// Uses the numerically stable logsumexp formulation (Arraymancer source of truth):
//   loss = mean( -y*x + max(x,0) + log1p(exp(-|x|)) )
@[inline]
pub fn sigmoid_cross_entropy[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	batch_size := input.shape[0]
	// B3 fix: reshape target from [batch] to [batch, 1] to match input shape [batch, 1]
	target_2d := target.reshape([batch_size, 1])!
	// B4 fix: numerically stable logsumexp formula aligned with Arraymancer
	sum := input.nreduce([target_2d], vtl.cast[T](0), fn [T](acc T, vals []T, i []int) T {
		x := f64(vals[0])
		y := f64(vals[1])
		next := -y * x + math.max(x, f64(0)) + math.log1p(math.exp(-math.abs(x)))
		return acc + vtl.cast[T](next)
	})!
	result := sum / vtl.cast[T](batch_size)
	return vtl.from_1d([result])
}

// mse squared error between the labels and the predictions
@[inline]
pub fn mse[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	sum := input.nreduce([target], vtl.cast[T](0), fn [T](acc T, vals []T, i []int) T {
		next := vtl.cast[T](math.pow(f64(vals[0] - vals[1]), 2.0))
		return acc + next
	})!
	size := input.size()
	result := sum / vtl.cast[T](size)
	return vtl.from_1d([result])
}

// softmax_cross_entropy computes the mean cross-entropy loss for a batch of
// logit vectors using the numerically stable log-sum-exp trick.
//
// input:  [batch_size, n_classes] — raw logits (unnormalised scores)
// target: [batch_size, n_classes] — one-hot or soft label targets
// returns: scalar loss tensor of shape [1]
//
// Formula (Arraymancer-style, numerically stable):
//   For each sample i:
//     lse_i = log( sum_j exp(logit_ij - max_i) ) + max_i
//     loss_i = lse_i - sum_j(target_ij * logit_ij)
//   mean_loss = mean over batch
@[inline]
pub fn softmax_cross_entropy[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	batch_size := input.shape[0]
	n_classes := input.shape[1]
	mut total_loss := f64(0)
	for i in 0 .. batch_size {
		// Compute per-sample max for numerical stability
		mut max_logit := f64(input.get([i, 0]))
		for c in 1 .. n_classes {
			v := f64(input.get([i, c]))
			if v > max_logit {
				max_logit = v
			}
		}
		// log-sum-exp
		mut sum_exp := f64(0)
		for c in 0 .. n_classes {
			sum_exp += math.exp(f64(input.get([i, c])) - max_logit)
		}
		lse := math.log(sum_exp) + max_logit
		// dot(target_i, logit_i)
		mut dot := f64(0)
		for c in 0 .. n_classes {
			dot += f64(target.get([i, c])) * f64(input.get([i, c]))
		}
		total_loss += lse - dot
	}
	mean_loss := total_loss / f64(batch_size)
	return vtl.from_1d([vtl.cast[T](mean_loss)])
}
