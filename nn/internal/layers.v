module internal

import vtl

pub fn dropout<T>(input &vtl.Tensor<T>, mask &vtl.Tensor<T>, prob f64) ?&vtl.Tensor<T> {
	mut iters, shape := input.iterators<T>([mask])?
	mut ret := vtl.new_tensor_like_with_shape<T>(input, shape)
	for {
		vals, i := iters.next() or { break }
		val := vals[0] * vals[1] / vtl.new_t<T>(prob)
		ret.set(i, val)
	}
	return ret
}

pub fn dropout_backwards<T>(gradient &vtl.Tensor<T>, mask &vtl.Tensor<T>, prob f64) ?&vtl.Tensor<T> {
	mut iters, shape := gradient.iterators<T>([mask])
	mut ret := vtl.new_tensor_like_with_shape<T>(gradient, shape)
	for {
		vals, i := iters.next() or { break }
		val := vals[0] * vals[1] / vtl.new_t<T>(prob)
		ret.set(i, val)
	}
	return ret
}

pub fn maxpool2d<T>(input &vtl.Tensor<T>, kernel []int, padding []int, stride []int) (&vtl.Tensor<int>, &vtl.Tensor<T>) {
	nn := input.shape[0]
	cc := input.shape[1]
	hh := input.shape[2]
	ww := input.shape[3]

	kk := kernel[0]
	kw := kernel[1]

	outh := (hh + 2 * padding[0] - kk) / stride[0] + 1
	outw := (ww + 2 * padding[1] - kw) / stride[1] + 1

	max_indices := vtl.zeros<int>([nn, cc, outh, outw])
	output := vtl.zeros<T>([nn, cc, outh, outw])

	// @todo: Implement maxpool here

	return max_indices, output
}

pub fn maxpool2d_backward<T>(shape []int, max_indices &vtl.Tensor<int>, grad_output &vtl.Tensor<T>) &vtl.Tensor<T> {
	if grad_output.size != max_indices.size {
		panic('maxpool2d_backward: grad_output and max_indices must have the same size')
	}

	// @todo: @ulises-jeremias to override this on other backends
	mut ret := vtl.zeros<T>(shape)
	for i in 0 .. grad_output.size {
		idx := max_indices.data.get<int>(i)
		ret.set_nth(idx, grad_output.data.get(i))
	}
	return ret
}
