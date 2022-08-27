module internal

import vtl

pub fn sgd_optimize<T>(mut value vtl.Tensor<T>, gradient &vtl.Tensor<T>, learning_rate f64) ? {
	mut iters, _ := value.iterators([gradient])?
	for {
		vals, i := vtl.iterators_next<T>(mut iters) or { break }
		val := vals[0] - vtl.new_t<T>(learning_rate) * vals[1]
		value.set(i, val)
	}
}
