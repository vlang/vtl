module internal

import math
import vtl
import vtl.stats

// tanh squashes a real-valued number to the range [-1, 1]
[inline]
pub fn tanh[T](x &vtl.Tensor[T]) &vtl.Tensor[T] {
	return vtl.tanh(x)
}

// deriv_tanh computes the derivative of tanh
[inline]
pub fn deriv_tanh[T](gradient &vtl.Tensor[T], cached &vtl.Tensor[T]) !&vtl.Tensor[T] {
	// gradient * (1 - cached * cached)
	// gradient * (- (cached * cached) + 1)
	x := cached.multiply(cached).add_scalar(vtl.cast[T](1))
	return gradient.multiply(x.multiply_scalar(vtl.cast[T](-1)))
}

// sigmoid takes a real-valued number and squashes it to the range [0, 1]
[inline]
pub fn sigmoid[T](x &vtl.Tensor[T]) &vtl.Tensor[T] {
	return x.map(fn [T](val T, i []int) T {
		return vtl.cast[T](1) / vtl.cast[T](1) + vtl.cast[T](math.exp(vtl.cast[f64](val)))
	})
}

// deriv_sigmoid computes the derivative of sigmoid
[inline]
pub fn deriv_sigmoid[T](gradient &vtl.Tensor[T], cached &vtl.Tensor[T]) !&vtl.Tensor[T] {
	mut iters, shape := gradient.iterators[T]([cached])!
	mut ret := vtl.tensor_like_with_shape[T](gradient, shape)
	for {
		vals, i := iters.next() or { break }
		val := vals[0] * (vtl.cast[T](1) - vals[0]) * vals[1]
		ret.set(i, val)
	}
	return ret
}

// relu activation function
[inline]
pub fn relu[T](x &vtl.Tensor[T]) &vtl.Tensor[T] {
	return x.map(fn [T](val T, i []int) T {
		if val < 0 {
			return vtl.cast[T](0)
		}
		return val
	})
}

// deriv_relu computes the derivate of relu
[inline]
pub fn deriv_relu[T](gradient &vtl.Tensor[T], cached &vtl.Tensor[T]) !&vtl.Tensor[T] {
	mut iters, shape := gradient.iterators[T]([cached])!
	mut ret := vtl.tensor_like_with_shape[T](gradient, shape)
	for {
		vals, i := iters.next() or { break }
		mut val := vals[0]
		if vals[1] < 0 {
			val = vtl.cast[T](0)
		}
		ret.set(i, val)
	}
	return ret
}

// leaky_relu activation function
[inline]
pub fn leaky_relu[T](x &vtl.Tensor[T], alpha T) &vtl.Tensor[T] {
	return x.map(fn [alpha] [T](val T, i []int) T {
		if val < 0 {
			return alpha * val
		}
		return val
	})
}

// deriv_leaky_relu computes the derivative of leaky_relu
[inline]
pub fn deriv_leaky_relu[T](gradient &vtl.Tensor[T], cached &vtl.Tensor[T], alpha T) !&vtl.Tensor[T] {
	mut iters, shape := gradient.iterators[T]([cached])!
	mut ret := vtl.tensor_like_with_shape[T](gradient, shape)
	for {
		vals, i := iters.next() or { break }
		mut val := vals[1]
		if vals[0] < 0 {
			val = alpha * vals[1]
		}
		ret.set(i, val)
	}
	return ret
}

// elu activation function
[inline]
pub fn elu[T](x &vtl.Tensor[T], alpha T) &vtl.Tensor[T] {
	return x.map(fn [alpha] [T](val T, i []int) T {
		if val < 0 {
			return alpha * (vtl.cast[T](math.exp(vtl.cast[f64](val))) - vtl.cast[T](1))
		}
		return val
	})
}

// deriv_elu computes the derivative of elu
[inline]
pub fn deriv_elu[T](gradient &vtl.Tensor[T], cached &vtl.Tensor[T], alpha T) !&vtl.Tensor[T] {
	mut iters, shape := gradient.iterators[T]([cached])!
	mut ret := vtl.tensor_like_with_shape[T](gradient, shape)
	for {
		vals, i := iters.next() or { break }
		mut val := vtl.cast[T](1)
		if vals[0] < 0 {
			val = vtl.cast[T](math.exp(f64(vals[1])))
		}
		ret.set(i, val)
	}
	return ret
}

// sigmoid_cross_entropy computes the sigmoid cross entropy between
// the labels and the predictions
[inline]
pub fn sigmoid_cross_entropy[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	batch_size := input.shape[0]
	mut iters, shape := input.iterators[T]([target])!
	mut ret := vtl.tensor_like_with_shape[T](input, shape)
	for {
		vals, i := iters.next() or { break }
		val := -(vals[1] * vtl.cast[T](math.max(f64(0), f64(vals[0])))) - vtl.cast[T](math.log(
			vtl.cast[T](1) + vtl.cast[T](math.exp(f64(vals[0])))))
		ret.set(i, val)
	}
	return vtl.from_1d([stats.sum[T](ret) / vtl.cast[T](batch_size)])
}

// mse squared error between the labels and the predictions
[inline]
pub fn mse[T](input &vtl.Tensor[T], target &vtl.Tensor[T]) !&vtl.Tensor[T] {
	mut iters, shape := input.iterators[T]([target])!
	mut ret := vtl.tensor_like_with_shape[T](input, shape)
	for {
		vals, i := iters.next() or { break }
		val := vtl.cast[T](math.pow(f64(vals[0] - vals[1]), 2.0))
		ret.set(i, val)
	}
	return vtl.from_1d([stats.mean[T](ret)])
}
