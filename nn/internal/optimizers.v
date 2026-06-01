module internal

import vtl

// sgd_optimize exposes this operation as part of the public API.
pub fn sgd_optimize[T](mut value vtl.Tensor[T], gradient &vtl.Tensor[T], learning_rate f64) ! {
	mut iters, _ := value.iterators([gradient])!
	for {
		vals, i := iters.next() or { break }
		val := vals[0] - vtl.cast[T](learning_rate) * vals[1]
		value.set(i, val)
	}
}
