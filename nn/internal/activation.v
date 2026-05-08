module internal

import math
import vtl

// tanh squashes a real-valued number to the range [-1, 1]
@[inline]
pub fn tanh[T](x &vtl.Tensor[T]) &vtl.Tensor[T] {
	return x.tanh()
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
		// correct sigmoid: 1 / (1 + exp(-x))
		return vtl.cast[T](1) / (vtl.cast[T](1) + vtl.cast[T](math.exp(-vtl.cast[f64](val))))
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

// gelu applies the Gaussian Error Linear Unit activation.
// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
@[inline]
pub fn gelu[T](x &vtl.Tensor[T]) &vtl.Tensor[T] {
	return x.map(fn [T](val T, i []int) T {
		one := vtl.cast[T](1)
		sqrt_2_over_pi := vtl.cast[T](0.7978845608028654)
		coef := vtl.cast[T](0.044715)
		tanh_arg := sqrt_2_over_pi * (val + coef * val * val * val)
		// tanh via epsilon-approximation: (exp(z)-exp(-z))/(exp(z)+exp(-z))
		exp_2z := math.exp(vtl.cast[f64](2.0 * tanh_arg))
		tanh_z := (exp_2z - vtl.cast[T](1)) / (exp_2z + vtl.cast[T](1))
		return vtl.cast[T](0.5) * val * (one + tanh_z)
	})
}

// deriv_gelu computes the derivative of GELU.
// d/dx GELU(x) = 0.5 * (1 + tanh(z)) + 0.5 * x * sech^2(z) * dz/dx
// where z = sqrt(2/pi) * (x + 0.044715 * x^3)
// and dz/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
@[inline]
pub fn deriv_gelu[T](gradient &vtl.Tensor[T], cached &vtl.Tensor[T]) !&vtl.Tensor[T] {
	return gradient.nmap([cached], fn [T](vals []T, i []int) T {
		x := vals[0]
		one := vtl.cast[T](1)
		sqrt_2_over_pi := vtl.cast[T](0.7978845608028654)
		coef := vtl.cast[T](0.044715)
		// compute tanh(z) and z for cached x
		z := sqrt_2_over_pi * (x + coef * x * x * x)
		exp_2z := math.exp(vtl.cast[f64](2.0 * z))
		tanh_z := (exp_2z - one) / (exp_2z + one)
		dz_dx := sqrt_2_over_pi * (one + vtl.cast[T](3.0 * 0.044715) * x * x)
		sech2_z := one - tanh_z * tanh_z
		return vals[0] * vtl.cast[T](0.5) * (one + tanh_z + x * sech2_z * dz_dx)
	})
}

// swish applies the Swish activation: x * sigmoid(beta * x), beta=1.
@[inline]
pub fn swish[T](x &vtl.Tensor[T]) &vtl.Tensor[T] {
	return x.map(fn [T](val T, i []int) T {
		one := vtl.cast[T](1)
		exp_neg := vtl.cast[T](math.exp(-vtl.cast[f64](val)))
		sig := one / (one + exp_neg)
		return val * sig
	})
}

// deriv_swish computes the derivative of Swish(x) = x * sigmoid(x).
// d/dx Swish = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
//            = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
@[inline]
pub fn deriv_swish[T](gradient &vtl.Tensor[T], cached &vtl.Tensor[T]) !&vtl.Tensor[T] {
	return gradient.nmap([cached], fn [T](vals []T, i []int) T {
		x := vals[0]
		one := vtl.cast[T](1)
		exp_neg := vtl.cast[T](math.exp(-vtl.cast[f64](x)))
		sig := one / (one + exp_neg)
		// derivative: gradient * [sigmoid + x * sigmoid * (1-sigmoid)]
		return vals[0] * sig * (one + x * (one - sig))
	})
}

// mish applies the Mish activation: x * tanh(softplus(x)).
// softplus(x) = log(1 + exp(x))
@[inline]
pub fn mish[T](x &vtl.Tensor[T]) &vtl.Tensor[T] {
	return x.map(fn [T](val T, i []int) T {
		one := vtl.cast[T](1)
		sp := vtl.cast[T](math.log1p(math.exp(vtl.cast[f64](val))))
		// tanh(sp) via (exp(2sp)-1)/(exp(2sp)+1)
		exp_2sp := math.exp(vtl.cast[f64](2.0 * sp))
		tanh_sp := (exp_2sp - one) / (exp_2sp + one)
		return val * tanh_sp
	})
}

// deriv_mish computes the derivative of Mish(x) = x * tanh(softplus(x)).
// d/dx Mish = tanh(softplus(x)) + x * (1 - tanh^2(softplus(x))) * sigmoid(x)
@[inline]
pub fn deriv_mish[T](gradient &vtl.Tensor[T], cached &vtl.Tensor[T]) !&vtl.Tensor[T] {
	return gradient.nmap([cached], fn [T](vals []T, i []int) T {
		x := vals[0]
		one := vtl.cast[T](1)
		exp_x := math.exp(vtl.cast[f64](x))
		sig := exp_x / (exp_x + vtl.cast[T](1))
		sp := vtl.cast[T](math.log1p(exp_x))
		exp_2sp := math.exp(vtl.cast[f64](2.0 * sp))
		tanh_sp := (exp_2sp - one) / (exp_2sp + one)
		// derivative: gradient * [tanh_sp + x * (1-tanh_sp^2) * sigmoid(x)]
		return vals[0] * (tanh_sp + x * (one - tanh_sp * tanh_sp) * sig)
	})
}

// deriv_softmax computes the Jacobian-vector product for softmax.
// For a softmax slice s_i = exp(x_i) / sum_j exp(x_j), the Jacobian is:
//   dL/dx_k = sum_i L_i * ds_i/dx_k
//   ds_i/dx_k = s_i * (delta_ik - s_k)   (i = k: s_i*(1-s_i), i ≠ k: -s_i*s_k)
//
// We implement the fast "jacobian * grad" version: grad_out * J^T gives
//   dL/dx_k = sum_i grad_out_i * ds_i/dx_k = grad_out_i * s_i * (delta_ik - s_k)
//           = s_k * (grad_out_k - sum_i grad_out_i * s_i)
@[inline]
pub fn deriv_softmax[T](gradient &vtl.Tensor[T], input &vtl.Tensor[T], dim int) !&vtl.Tensor[T] {
	shape := input.shape
	ndim := shape.len
	actual_dim := if dim < 0 { ndim + dim } else { dim }
	n := shape[actual_dim]

	if actual_dim != ndim - 1 {
		return error('deriv_softmax: only last-dimension softmax is implemented')
	}

	// Compute number of rows = product of all dims except last
	mut nrows := 1
	for d in 0 .. ndim - 1 {
		nrows *= shape[d]
	}
	mut output := vtl.zeros[T](shape)

	// Recompute softmax per row to get the jacobian contraction
	for r in 0 .. nrows {
		base := r * n
		// Recompute softmax on the input (not softmax output) for backward
		mut max_val := input.get_nth(base)
		for c in 1 .. n {
			v := input.get_nth(base + c)
			if v > max_val { max_val = v }
		}
		mut sum_exp := vtl.cast[T](0)
		mut softmax_vals := []T{len: n}
		for c in 0 .. n {
			exp_val := vtl.cast[T](math.exp(vtl.cast[f64](input.get_nth(base + c) - max_val)))
			softmax_vals[c] = exp_val
			sum_exp += exp_val
		}
		for c in 0 .. n {
			softmax_vals[c] = softmax_vals[c] / sum_exp
		}
		// weighted_sum = sum_i grad_i * softmax_i
		mut weighted_sum := vtl.cast[T](0)
		for c in 0 .. n {
			weighted_sum += gradient.get_nth(base + c) * softmax_vals[c]
		}
		// dL/dx_k = softmax_k * (grad_k - weighted_sum)
		for c in 0 .. n {
			grad_val := gradient.get_nth(base + c)
			output.set_nth(base + c, softmax_vals[c] * (grad_val - weighted_sum))
		}
	}
	return output
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

// softmax_forward computes softmax along a specified dimension.
pub fn softmax_forward[T](input &vtl.Tensor[T], dim int) !&vtl.Tensor[T] {
	shape := input.shape
	ndim := shape.len
	actual_dim := if dim < 0 { ndim + dim } else { dim }
	n := shape[actual_dim]

	if actual_dim != ndim - 1 {
		return error('softmax_forward: only last-dimension softmax is implemented')
	}

	// For each slice along the last axis: compute exp(x - max(x)) / sum(exp(x - max(x)))
	size := input.size()
	mut output_data := []f64{len: size}
	rows := size / n

	for r in 0 .. rows {
		base := r * n
		// Find max for numerical stability
		mut max_val := f64(input.get_nth(base))
		for c in 1 .. n {
			v := f64(input.get_nth(base + c))
			if v > max_val { max_val = v }
		}
		// Compute exp and sum
		mut exp_sum := f64(0)
		for c in 0 .. n {
			e := math.exp(f64(input.get_nth(base + c)) - max_val)
			output_data[base + c] = e
			exp_sum += e
		}
		// Normalize
		for c in 0 .. n {
			output_data[base + c] /= exp_sum
		}
	}
	return vtl.from_array(output_data.map(vtl.cast[T](it)), shape)!
}
