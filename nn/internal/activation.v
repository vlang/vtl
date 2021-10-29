module internal

import math
import vtl

// tanh squashes a real-valued number to the range [-1, 1]
[inline]
pub fn tanh<T>(x &vtl.Tensor<T>) &vtl.Tensor<T> {
	return vtl.tanh(x)
}

// deriv_tanh computes the derivative of tanh
[inline]
pub fn deriv_tanh<T>(grandient &vtl.Tensor<T>, cached &vtl.Tensor<T>) &vtl.Tensor<T> {
	// gradient * (1 - cached * cached)
	// gradient * (- (cached * cached) + 1)
	x := vtl.add_scalar(vtl.multiply(cached, cached), T(1))
	return vtl.multiply(grandient, vtl.multiply_scalar(x, T(-1)))
}

// sigmoid takes a real-valued number and squashes it to the range [0, 1]
[inline]
pub fn sigmoid<T>(x &vtl.Tensor<T>) &vtl.Tensor<T> {
	mut ret := vtl.new_tensor_like<T>(x)
	mut iter := t.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := T(1) / T(1) + T(math.exp(f64(val)))
		ret.data.set<T>(pos, next_val)
	}
	return ret
}

// deriv_sigmoid computes the derivative of sigmoid
[inline]
pub fn deriv_sigmoid<T>(grandient &vtl.Tensor<T>, cached &vtl.Tensor<T>) &vtl.Tensor<T> {
	mut ret := vtl.new_tensor_like<T>(gradient)
	mut iters := vtl.iterators<T>([gradient, cached])
	for {
		vals, pos := vtl.iterators_next<T>(mut iters) or { break }
		val := vals[0] * (T(1) - vals[0]) * vals[1]
		ret.data.set<T>(pos, val)
	}
	return ret
}

// relu activation function
[inline]
pub fn relu<T>(x &vtl.Tensor<T>) &vtl.Tensor<T> {
	mut ret := vtl.new_tensor_like<T>(x)
	mut iter := x.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := if val < 0 { 0 } else { val }
		ret.data.set<T>(pos, val)
	}
	return ret
}

// deriv_relu computes the derivate of relu
[inline]
pub fn deriv_relu<T>(grandient &vtl.Tensor<T>, cached &vtl.Tensor<T>) &vtl.Tensor<T> {
	mut ret := vtl.new_tensor_like<T>(gradient)
	mut iters := vtl.iterators<T>([gradient, cached])
	for {
		vals, pos := vtl.iterators_next<T>(mut iters) or { break }
		val := if vals[0] < 0 { 0 } else { vals[1] }
		ret.data.set<T>(pos, val)
	}
	return ret
}

// leaky_relu activation function
[inline]
pub fn leaky_relu<T>(x &vtl.Tensor<T>, alpha T) &vtl.Tensor<T> {
	mut ret := vtl.new_tensor_like<T>(x)
	mut iter := x.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := if val < 0 { alpha * val } else { val }
		ret.data.set<T>(pos, val)
	}
	return ret
}

// deriv_leaky_relu computes the derivative of leaky_relu
[inline]
pub fn deriv_leaky_relu<T>(grandient &vtl.Tensor<T>, cached &vtl.Tensor<T>, alpha T) &vtl.Tensor<T> {
	mut ret := vtl.new_tensor_like<T>(gradient)
	mut iters := vtl.iterators<T>([gradient, cached])
	for {
		vals, pos := vtl.iterators_next<T>(mut iters) or { break }
		val := if vals[0] <= 0 { alpha * vals[1] } else { vals[1] }
		ret.data.set<T>(pos, val)
	}
	return ret
}

// elu activation function
[inline]
pub fn elu<T>(x &vtl.Tensor<T>, alpha T) &vtl.Tensor<T> {
	mut ret := vtl.new_tensor_like<T>(x)
	mut iter := x.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := if val < 0 { alpha * (T(math.exp(f64(val))) - T(1)) } else { val }
		ret.data.set<T>(pos, val)
	}
	return ret
}

// deriv_elu computes the derivative of elu
[inline]
pub fn deriv_elu<T>(grandient &vtl.Tensor<T>, cached &vtl.Tensor<T>, alpha T) &vtl.Tensor<T> {
	mut ret := vtl.new_tensor_like<T>(gradient)
	mut iters := vtl.iterators<T>([gradient, cached])
	for {
		vals, pos := vtl.iterators_next<T>(mut iters) or { break }
		val := if vals[0] <= 0 { T(math.exp(f64(vals[1]))) } else { T(1) }
		ret.data.set<T>(pos, val)
	}
	return ret
}

// sigmoid_cross_entropy_with_logits computes the sigmoid cross entropy between
// the labels and the predictions
[inline]
pub fn sigmoid_cross_entropy_with_logits<T>(input &vtl.Tensor<T>, target &vtl.Tensor<T>) T {
	batch_size := input.shape[0]
	mut ret := vtl.new_tensor_like<T>(input)
	mut iter := vtl.iterators<T>([input, target])
	for {
		vals, pos := vtl.iterators_next<T>(mut iter) or { break }
		val := -(vals[1] * T(math.max(f64(0), f64(vals[0])))) - T(math.log(T(1) +
			T(math.exp(f64(vals[0])))))
		ret.data.set<T>(pos, val)
	}
	return vtl.sum(ret) / T(batch_size)
}

// mse squared error between the labels and the predictions
[inline]
pub fn mse<T>(input &vtl.Tensor<T>, target &vtl.Tensor<T>) &vtl.Tensor<T> {
	mut ret := vtl.new_tensor_like<T>(input)
	mut iter := vtl.iterators<T>([input, target])
	for {
		vals, pos := vtl.iterators_next<T>(mut iter) or { break }
		val := T(math.pow(f64(vals[0] - vals[1]), 2.0))
		ret.data.set<T>(pos, val)
	}
	return vtl.from_1d([vtl.mean(ret)])
}
