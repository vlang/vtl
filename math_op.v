module vtl

fn handle_add<T>(xs []T, _ int) T {
	return xs[0] + xs[1]
}

// add adds two tensors elementwise
[inline]
pub fn add<T>(a &Tensor<T>, b &Tensor<T>) &Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_add, b)
	mut ret := new_tensor_like<T>(a)
	mut iters := iterators<T>([a, b])
	for {
		vals, pos := iterators_next<T>(mut iters) or { break }
		val := handle_add<T>(vals, pos)
		ret.data.set<T>(pos, val)
	}
	return ret
}

// add adds a scalar to a tensor elementwise
[inline]
pub fn add_scalar<T>(a &Tensor<T>, scalar T) &Tensor<T> {
	// @todo: Implement using a.map
	mut ret := new_tensor_like<T>(a)
	mut iter := a.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_add<T>([val, scalar], pos)
		ret.data.set<T>(pos, next_val)
	}
	return ret
}

fn handle_substract<T>(xs []T, _ int) T {
	return xs[0] - xs[1]
}

// substract substracts two tensors elementwise
[inline]
pub fn substract<T>(a &Tensor<T>, b &Tensor<T>) &Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_substract, b)
	mut ret := new_tensor_like<T>(a)
	mut iters := iterators<T>([a, b])
	for {
		vals, pos := iterators_next<T>(mut iters) or { break }
		val := handle_substract<T>(vals, pos)
		ret.data.set<T>(pos, val)
	}
	return ret
}

// substract substracts a scalar to a tensor elementwise
[inline]
pub fn substract_scalar<T>(a &Tensor<T>, scalar T) &Tensor<T> {
	// @todo: Implement using a.map
	mut ret := new_tensor_like<T>(a)
	mut iter := a.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_substract<T>([val, scalar], pos)
		ret.data.set<T>(pos, next_val)
	}
	return ret
}

fn handle_divide<T>(xs []T, _ int) T {
	return xs[0] / xs[1]
}

// divide divides two tensors elementwise
[inline]
pub fn divide<T>(a &Tensor<T>, b &Tensor<T>) &Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_divide, b)
	mut ret := new_tensor_like<T>(a)
	mut iters := iterators<T>([a, b])
	for {
		vals, pos := iterators_next<T>(mut iters) or { break }
		val := handle_divide<T>(vals, pos)
		ret.data.set<T>(pos, val)
	}
	return ret
}

// divide divides a scalar to a tensor elementwise
[inline]
pub fn divide_scalar<T>(a &Tensor<T>, scalar T) &Tensor<T> {
	// @todo: Implement using a.map
	mut ret := new_tensor_like<T>(a)
	mut iter := a.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_divide<T>([val, scalar], pos)
		ret.data.set<T>(pos, next_val)
	}
	return ret
}

fn handle_multiply<T>(xs []T, _ int) T {
	return xs[0] * xs[1]
}

// multiply multiplies two tensors elementwise
[inline]
pub fn multiply<T>(a &Tensor<T>, b &Tensor<T>) &Tensor<T> {
	// @todo: Implement using a.nmap
	// return a.nmap<T>(handle_multiply, b)
	mut ret := new_tensor_like<T>(a)
	mut iters := iterators<T>([a, b])
	for {
		vals, pos := iterators_next<T>(mut iters) or { break }
		val := handle_multiply<T>(vals, pos)
		ret.data.set<T>(pos, val)
	}
	return ret
}

// multiply multiplies a scalar to a tensor elementwise
[inline]
pub fn multiply_scalar<T>(a &Tensor<T>, scalar T) &Tensor<T> {
	// @todo: Implement using a.map
	mut ret := new_tensor_like<T>(a)
	mut iter := a.iterator()
	for {
		val, pos := iter.next() or { break }
		next_val := handle_multiply<T>([val, scalar], pos)
		ret.data.set<T>(pos, next_val)
	}
	return ret
}
