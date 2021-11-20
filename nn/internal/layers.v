module internal

import vtl

pub fn maxpool_backward<T>(shape []int, max_indices &vtl.Tensor<int>, grad_output &vtl.Tensor<T>) {
	if grad_output.size != max_indices.size {
		panic('maxpool_backward: grad_output and max_indices must have the same size')
	}

	mut ret := vtl.zeros<T>(shape)
	for i in 0 .. grad_output.size {
		idx := max_indices.data[i]
		ret.data.set<T>(idx, grad_output.data[i])
	}
	return ret
}
