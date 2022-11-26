module vtl

fn handle_add[T](a T, b T) ?T {
	$if T is bool {
		return td[T](a).bool() || td[T](b).bool()
	} $else $if T is string {
		return '${a.str()}${b.str()}'
	} $else {
		return a + b
	}
	panic('this part of the code is never reached. just a workaround')
}

// add adds two tensors elementwise
[inline]
pub fn (a &Tensor[T]) add[T](b &Tensor[T]) ?&Tensor[T] {
	// @todo: Implement using nmap
	mut iters, shape := a.iterators[T]([b])?
	mut ret := tensor_like_with_shape[T](a, shape)
	for {
		vals, i := iters.next() or { break }
		val := handle_add[T](vals[0], vals[1])?
		ret.set(i, val)
	}
	return ret
}

// add adds a scalar to a tensor elementwise
[inline]
pub fn (a &Tensor[T]) add_scalar[T](scalar T) ?&Tensor[T] {
	// @todo: Implement using map
	mut ret := tensor_like[T](a)
	mut iter := a.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_add[T](val, scalar)?
		ret.set(i, next_val)
	}
	return ret
}

fn handle_subtract[T](a T, b T) ?T {
	$if T is bool {
		return td[T](a).bool() && td[T](b).bool()
	} $else $if T is string {
		return error(@FN + ' is not supported for type ${typeof(a).name}')
	} $else {
		return a - b
	}
	panic('this part of the code is never reached. just a workaround')
}

// subtract subtracts two tensors elementwise
[inline]
pub fn (a &Tensor[T]) subtract[T](b &Tensor[T]) ?&Tensor[T] {
	// @todo: Implement using nmap
	mut iters, shape := a.iterators[T]([b])?
	mut ret := tensor_like_with_shape[T](a, shape)
	for {
		vals, i := iters.next() or { break }
		val := handle_subtract[T](vals[0], vals[1])?
		ret.set(i, val)
	}
	return ret
}

// subtract subtracts a scalar to a tensor elementwise
[inline]
pub fn (a &Tensor[T]) subtract_scalar[T](scalar T) ?&Tensor[T] {
	// @todo: Implement using map
	mut ret := tensor_like[T](a)
	mut iter := a.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_subtract[T](val, scalar)?
		ret.set(i, next_val)
	}
	return ret
}

fn handle_divide[T](a T, b T) ?T {
	$if T is bool || T is string {
		return error(@FN + ' is not supported for type ${typeof(a).name}')
	} $else {
		return a / b
	}
	panic('this part of the code is never reached. just a workaround')
}

// divide divides two tensors elementwise
[inline]
pub fn (a &Tensor[T]) divide[T](b &Tensor[T]) ?&Tensor[T] {
	// @todo: Implement using nmap
	mut iters, shape := a.iterators[T]([b])?
	mut ret := tensor_like_with_shape[T](a, shape)
	for {
		vals, i := iters.next() or { break }
		val := handle_divide[T](vals[0], vals[1])?
		ret.set(i, val)
	}
	return ret
}

// divide divides a scalar to a tensor elementwise
[inline]
pub fn (a &Tensor[T]) divide_scalar[T](scalar T) ?&Tensor[T] {
	// @todo: Implement using map
	mut ret := tensor_like[T](a)
	mut iter := a.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_divide[T](val, scalar)?
		ret.set(i, next_val)
	}
	return ret
}

fn handle_multiply[T](a T, b T) ?T {
	$if T is bool || T is string {
		panic(@FN + ' is not supported for type ${typeof(a).name}')
	} $else {
		return a * b
	}
	panic('this part of the code is never reached. just a workaround')
}

// multiply multiplies two tensors elementwise
[inline]
pub fn (a &Tensor[T]) multiply[T](b &Tensor[T]) ?&Tensor[T] {
	// @todo: Implement using nmap
	mut iters, shape := a.iterators[T]([b])?
	mut ret := tensor_like_with_shape[T](a, shape)
	for {
		vals, i := iters.next() or { break }
		val := handle_multiply[T](vals[0], vals[1])?
		ret.set(i, val)
	}
	return ret
}

// multiply multiplies a scalar to a tensor elementwise
[inline]
pub fn (a &Tensor[T]) multiply_scalar[T](scalar T) ?&Tensor[T] {
	// @todo: Implement using map
	mut ret := tensor_like[T](a)
	mut iter := a.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := handle_multiply[T](val, scalar)?
		ret.set(i, next_val)
	}
	return ret
}
