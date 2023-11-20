module vtl

// apply applies a function to each element of a given Tensor
pub fn (mut t Tensor[T]) apply[T](f fn (x T, i []int) T) {
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := f(val, i)
		t.set(i, next_val)
	}
}

// map maps a function to a given Tensor retuning a new Tensor with same shape
pub fn (t &Tensor[T]) map[T](f fn (x T, i []int) T) &Tensor[T] {
	mut ret := tensor_like[T](t)
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		next_val := f(val, i)
		ret.set(i, next_val)
	}
	return ret
}

// reduce reduces a function to a given Tensor retuning a new agregatted value
pub fn (t &Tensor[T]) reduce[T](init T, f fn (acc T, x T, i []int) T) T {
	mut ret := init
	mut iter := t.iterator()
	for {
		val, i := iter.next() or { break }
		ret = f(ret, val, i)
	}
	return ret
}

// napply applies a function to each element of a given Tensor with params
pub fn (mut t Tensor[T]) napply[T](ts []&Tensor[T], f fn (xs []T, i []int) T) ! {
	mut iters, _ := t.iterators[T](ts)!
	for {
		vals, i := iters.next() or { break }
		val := f(vals, i)
		t.set(i, val)
	}
}

// nmap maps a function to a given list of Tensor retuning a new Tensor with same shape
pub fn (t &Tensor[T]) nmap[T](ts []&Tensor[T], f fn (xs []T, i []int) T) !&Tensor[T] {
	mut iters, shape := t.iterators[T](ts)!
	mut ret := tensor_like_with_shape[T](t, shape)
	for {
		vals, i := iters.next() or { break }
		val := f(vals, i)
		ret.set(i, val)
	}
	return ret
}

// nreduce reduces a function to a given list of Tensor retuning a new agregatted value
pub fn (t &Tensor[T]) nreduce[T](ts []&Tensor[T], init T, f fn (acc T, xs []T, i []int) T) !T {
	mut ret := init
	mut iters, _ := t.iterators[T](ts)!
	for {
		vals, i := iters.next() or { break }
		ret = f(ret, vals, i)
	}
	return ret
}

// with_dims returns a new Tensor adding dimensions so that it has
// at least `n` dimensions
pub fn (t &Tensor[T]) with_dims[T](n int) !&Tensor[T] {
	if t.rank() >= n {
		return t.view()
	}
	d := n - t.rank()
	mut newshape := []int{len: d, init: 1}
	newshape << t.shape
	return t.reshape(newshape)
}

// with_broadcast expands a `Tensor`s dimensions n times by broadcasting
// the shape and strides
pub fn (t &Tensor[T]) with_broadcast[T](n int) !&Tensor[T] {
	mut newshape := []int{}
	newshape << t.shape
	newshape << []int{len: n, init: 1}
	mut newstrides := []int{}
	newstrides << t.strides
	newstrides << []int{len: n}
	return t.as_strided(newshape, newstrides)
}

// diagonal returns a view of the diagonal entries
// of a two dimensional tensor
pub fn (t &Tensor[T]) diagonal[T]() &Tensor[T] {
	nel := iarray_min(t.shape)
	newshape := [nel]
	newstrides := [iarray_sum(t.strides)]
	mut ret := &Tensor[T]{
		data: t.data
		shape: newshape
		strides: newstrides
		size: nel
		memory: t.memory
	}
	ret.ensure_memory()
	return ret
}

// ravel returns a flattened view of an Tensor if possible,
// otherwise a flattened copy
@[inline]
pub fn (t &Tensor[T]) ravel[T]() !&Tensor[T] {
	return t.reshape([-1])
}

// reshape returns an Tensor with a new shape
pub fn (t &Tensor[T]) reshape[T](shape []int) !&Tensor[T] {
	size := size_from_shape(t.shape)
	newshape, _ := shape_with_autosize(shape, size)!
	mut ret := tensor_like_with_shape[T](t, newshape)
	ret.data = t.data
	ret.ensure_memory()
	return ret
}

// as_strided returns a view of the Tensor with new shape and strides
pub fn (t &Tensor[T]) as_strided[T](shape []int, strides []int) !&Tensor[T] {
	newshape, _ := shape_with_autosize(shape, t.size)!
	mut ret := tensor_like_with_shape_and_strides[T](t, newshape, strides)
	ret.data = t.data
	ret.ensure_memory()
	return ret
}

// transpose permutes the axes of an tensor in a specified
// order and returns a view of the data
pub fn (t &Tensor[T]) transpose[T](order []int) !&Tensor[T] {
	mut ret := t.view()
	n := order.len
	t.assert_rank(n)!
	mut permutation := []int{len: 32}
	mut reverse_permutation := []int{len: 32, init: -1}
	mut i := 0
	for i < n {
		mut axis := order[i]
		if axis < 0 {
			axis = t.rank() + axis
		}
		if axis < 0 || axis >= t.rank() {
			return error('Bad permutation')
		}
		if reverse_permutation[axis] != -1 {
			return error('Bad permutation')
		}
		reverse_permutation[axis] = i
		permutation[i] = axis
		i++
	}
	mut ii := 0
	for ii < n {
		ret.shape[ii] = t.shape[permutation[ii]]
		ret.strides[ii] = t.strides[permutation[ii]]
		ii++
	}
	ret.ensure_memory()
	return ret
}

// t returns a ful transpose of an tensor, with the axes
// reversed
pub fn (t &Tensor[T]) t[T]() !&Tensor[T] {
	order := irange(0, t.rank())
	return t.transpose(order.reverse())
}

// swapaxes returns a view of an tensor with two axes
// swapped.
pub fn (t &Tensor[T]) swapaxes[T](a1 int, a2 int) !&Tensor[T] {
	mut order := irange(0, t.rank())
	tmp := order[a1]
	order[a1] = order[a2]
	order[a2] = tmp
	return t.transpose(order)
}

fn fabs(x f64) f64 {
	return if x > 0.0 { x } else { -x }
}

// slice returns a tensor from a variadic list of indexing operations
pub fn (t &Tensor[T]) slice[T](idx ...[]int) !&Tensor[T] {
	mut newshape := t.shape.clone()
	mut newstrides := t.strides.clone()
	mut indexer := []int{}
	for i, dex in idx {
		mut fi := 0
		mut li := 0
		// dimension is entirely included in output
		if dex.len == 0 {
			assert newshape[i] == t.shape[i]
			assert newstrides[i] == t.strides[i]
			indexer << 0
		}
		// dimension sliced from array
		else if dex.len == 1 {
			newshape[i] = 0
			newstrides[i] = 0
			fi = dex[0]
			if fi < 0 {
				fi += t.shape[i]
			}
			indexer << fi
		}
		// dimension specified by start and stop value
		else if dex.len == 2 {
			fi = dex[0]
			li = dex[1]
			if fi < 0 {
				fi += t.shape[i]
			}
			if li < 0 {
				li += t.shape[i]
			}
			if fi == li {
				newshape[i] = 0
				newstrides[i] = 0
				indexer << fi
			} else {
				newshape[i] = li - fi
				indexer << fi
			}
		}
		// dimension specified by start, stop, and step
		else if dex.len == 3 {
			fi = dex[0]
			li = dex[1]
			step := dex[2]
			abstep := int(fabs(step))
			if fi < 0 {
				fi += t.shape[i]
			}
			if li < 0 {
				li += t.shape[i]
			}
			offset := li - fi
			newshape[i] = offset / abstep + offset % abstep
			newstrides[i] = step * newstrides[i]
			indexer << fi
		}
	}
	// remove 0 shaped dimensions
	newshape_, newstrides_ := filter_shape_not_strides(newshape, newstrides)!
	mut offset := 0
	for i in 0 .. indexer.len {
		offset += t.strides[i] * indexer[i]
	}
	mut ret := &Tensor[T]{
		shape: newshape_.clone()
		strides: newstrides_.clone()
		size: size_from_shape(newshape_)
		data: t.data.offset[T](offset)
		memory: .row_major
	}
	ret.ensure_memory()
	return ret
}

// slice_hilo returns a view of an array from a list of starting
// indices and a list of closing indices.
pub fn (t &Tensor[T]) slice_hilo[T](idx1 []int, idx2 []int) !&Tensor[T] {
	mut newshape := t.shape.clone()
	mut newstrides := t.strides.clone()
	idx_start := pad_with_zeros(idx1, t.rank())
	idx_end := pad_with_max(idx2, t.shape, t.rank())
	mut idx := []int{cap: t.rank()}
	for ii in 0 .. t.rank() {
		mut fi := idx_start[ii]
		if fi < 0 {
			fi += t.shape[ii]
		}
		mut li := idx_end[ii]
		if li < 0 {
			li += t.shape[ii]
		}
		if fi == li {
			newshape[ii] = 0
			newstrides[ii] = 0
			idx << fi
		} else {
			offset := li - fi
			newshape[ii] = offset
			idx << fi
		}
	}
	// remove 0 shaped dimensions
	newshape_, newstrides_ := filter_shape_not_strides(newshape, newstrides)!
	mut offset := 0
	for i in 0 .. t.rank() {
		offset += t.strides[i] * idx[i]
	}
	mut ret := &Tensor[T]{
		shape: newshape_.clone()
		strides: newstrides_.clone()
		size: size_from_shape(newshape_)
		data: t.data.offset[T](offset)
		memory: .row_major
	}
	ret.ensure_memory()
	return ret
}
