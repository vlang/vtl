module vtl

// add adds two tensors elementwise
[inline]
pub fn add<T>(a &Tensor<T>, b &Tensor<T>) ?&Tensor<T> {
	// @todo: Implement using nmap
	// return a.nmap<T>(handle_add, b)
	mut iters, shape := a.iterators<T>([b])?
	mut ret := new_tensor_like_with_shape<T>(a, shape)
	for {
		vals, i := iterators_next<T>(mut iters) or { break }
		val := vals[0] + vals[1]
		ret.set(i, val)
	}
	return ret
}

// add adds a scalar to a tensor elementwise
[inline]
pub fn add_scalar<T>(a &Tensor<T>, scalar T) &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(a)
	mut iter := a.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := val + scalar
		ret.set(i, next_val)
	}
	return ret
}

// substract substracts two tensors elementwise
[inline]
pub fn substract<T>(a &Tensor<T>, b &Tensor<T>) ?&Tensor<T> {
	// @todo: Implement using nmap
	mut iters, shape := a.iterators<T>([b])?
	mut ret := new_tensor_like_with_shape<T>(a, shape)
	for {
		vals, i := iterators_next<T>(mut iters) or { break }
		val := vals[0] - vals[1]
		ret.set(i, val)
	}
	return ret
}

// substract substracts a scalar to a tensor elementwise
[inline]
pub fn substract_scalar<T>(a &Tensor<T>, scalar T) &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(a)
	mut iter := a.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := val - scalar
		ret.set(i, next_val)
	}
	return ret
}

// divide divides two tensors elementwise
[inline]
pub fn divide<T>(a &Tensor<T>, b &Tensor<T>) ?&Tensor<T> {
	// @todo: Implement using nmap
	mut iters, shape := a.iterators<T>([b])?
	mut ret := new_tensor_like_with_shape<T>(a, shape)
	for {
		vals, i := iterators_next<T>(mut iters) or { break }
		val := vals[0] / vals[1]
		ret.set(i, val)
	}
	return ret
}

// divide divides a scalar to a tensor elementwise
[inline]
pub fn divide_scalar<T>(a &Tensor<T>, scalar T) &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(a)
	mut iter := a.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := val / scalar
		ret.set(i, next_val)
	}
	return ret
}

// multiply multiplies two tensors elementwise
[inline]
pub fn multiply<T>(a &Tensor<T>, b &Tensor<T>) ?&Tensor<T> {
	// @todo: Implement using nmap
	// return a.nmap<T>(handle_multiply, b)
	mut iters, shape := a.iterators<T>([b])?
	mut ret := new_tensor_like_with_shape<T>(a, shape)
	for {
		vals, i := iterators_next<T>(mut iters) or { break }
		val := vals[0] * vals[1]
		ret.set(i, val)
	}
	return ret
}

// multiply multiplies a scalar to a tensor elementwise
[inline]
pub fn multiply_scalar<T>(a &Tensor<T>, scalar T) &Tensor<T> {
	// @todo: Implement using map
	mut ret := new_tensor_like<T>(a)
	mut iter := a.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := val * scalar
		ret.set(i, next_val)
	}
	return ret
}
