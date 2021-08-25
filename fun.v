module vtl

import math
import vtl.storage

pub type MapFn = fn (x T, i int) U

pub type ApplyFn = fn (x T, i int) T

pub type NMapFn = fn (x []T, i int) U

pub type NApplyFn = fn (x []T, i int)

// map maps a function to a given Tensor retuning a new Tensor with same shape
pub fn (t &Tensor<T>) map<T, U>(f MapFn<T, U>) &Tensor<U> {
	mut ret := new_tensor_like<T>(t).as_type<T, U>()
	mut iter := t.iterator()
	mut pos := iter.pos
	for {
		val := iter.next() or { break }
		next_val := f(val, pos)
		storage.storage_set<U>(ret.data, pos, next_val)
		pos = iter.pos
	}
	return ret
}

// map maps a function to a given list of Tensor retuning a new Tensor with same shape
pub fn (t &Tensor<T>) nmap<T, U>(f NMapFn<T, U>, ts ...Tensor<T>) &Tensor<U> {
	mut ret := new_tensor_like<T>(t).as_type<T, U>()
	mut iters := t.iterators<T>(ts)
	mut i := 0
	for {
		vals := iterators_next<T>(mut iters) or { break }
		val := f(vals, i)
		storage.storage_set<U>(ret.data, i, val)
		i++
	}
	return ret
}

// napply applies a function to each element of a given Tensor
pub fn (t &Tensor<T>) napply<T>(f NApplyFn<T>, ts ...Tensor<T>) {
	mut iters := t.iterators<T>(ts)
	mut i := 0
	for {
		vals := iterators_next<T>(mut iters) or { break }
		val := f(vals, i)
		storage.storage_set<T>(t.data, i, val)
		i++
	}
}

// equal checks if two Tensors are equal
fn equal<T>(t &Tensor<T>, other &Tensor<T>) bool {
	if t.shape != other.shape {
		return false
	}
	mut iters := iterators<T>([t, other])
	for {
		vals := iterators_next<T>(mut iters) or { break }
		if vals[0] != vals[1] {
			return false
		}
	}
	return true
}

// equal checks if two Tensors are equal
pub fn (t &Tensor<T>) equal<T>(other &Tensor<T>) bool {
	return equal<T>(t, other)
}

// diagonal returns a view of the diagonal entries
// of a two dimensional tensor
pub fn (t &Tensor<T>) diagonal<T>() &Tensor<T> {
	nel := iarray_min(t.shape)
	newshape := [nel]
	newstrides := [iarray_sum(t.strides)]
	return &Tensor<T>{
		data: t.data
		shape: newshape
		strides: newstrides
		size: nel
		memory: t.memory
	}
}

// ravel returns a flattened view of an Tensor if possible,
// otherwise a flattened copy
[inline]
pub fn (t &Tensor<T>) ravel<T>() &Tensor<T> {
	return t.reshape([-1])
}

// reshape returns an Tensor with a new shape
pub fn (t &Tensor<T>) reshape<T>(shape []int) &Tensor<T> {
	size := size_from_shape(shape)
	newshape, newsize := shape_with_autosize(shape, size)
	if newsize != size {
		panic('${@METHOD}: cannot reshape')
	}
	mut ret := new_tensor_like_with_shape<T>(t, newshape)
	ret.data = t.data
	return ret
}

// transpose permutes the axes of an tensor in a specified
// order and returns a view of the data
pub fn (t &Tensor<T>) transpose<T>(order []int) &Tensor<T> {
	mut ret := t.view()
	n := order.len
	assert_rank<T>(t, n)
	mut permutation := []int{len: 32}
	mut reverse_permutation := []int{len: 32, init: -1}
	mut i := 0
	for i < n {
		mut axis := order[i]
		if axis < 0 {
			axis = t.rank() + axis
		}
		if axis < 0 || axis >= t.rank() {
			panic('Bad permutation')
		}
		if reverse_permutation[axis] != -1 {
			panic('Bad permutation')
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
	return ret
}

// t returns a ful transpose of an tensor, with the axes
// reversed
pub fn (t &Tensor<T>) t<T>() &Tensor<T> {
	order := irange(0, t.rank())
	return t.transpose(order.reverse())
}

// swapaxes returns a view of an tensor with two axes
// swapped.
pub fn (t &Tensor<T>) swapaxes<T>(a1 int, a2 int) &Tensor<T> {
	mut order := irange(0, t.rank())
	tmp := order[a1]
	order[a1] = order[a2]
	order[a2] = tmp
	return t.transpose(order)
}

// slice returns a tensor from a variadic list of indexing operations
pub fn (t &Tensor<T>) slice<T>(idx ...[]int) &Tensor<T> {
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
			abstep := int(math.abs(step))
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
	newshape_, newstrides_ := filter_shape_not_strides(newshape, newstrides)
	mut offset := 0
	for i in 0 .. indexer.len {
		offset += t.strides[i] * indexer[i]
	}
	mut ret := &Tensor<T>{
		shape: newshape_.clone()
		strides: newstrides_.clone()
		size: size_from_shape(newshape_)
		data: storage.storage_offset<T>(t.data, offset)
		memory: .colmajor
	}
	ensure_memory<T>(mut ret)
	return ret
}

// slice_hilo returns a view of an array from a list of starting
// indices and a list of closing indices.
pub fn (t &Tensor<T>) slice_hilo<T>(idx1 []int, idx2 []int) &Tensor<T> {
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
	newshape_, newstrides_ := filter_shape_not_strides(newshape, newstrides)
	mut offset := 0
	for i in 0 .. t.rank() {
		offset += t.strides[i] * idx[i]
	}
	mut ret := &Tensor<T>{
		shape: newshape_.clone()
		strides: newstrides_.clone()
		size: size_from_shape(newshape_)
		data: storage.storage_offset<T>(t.data, offset)
		memory: .colmajor
	}
	ensure_memory<T>(mut ret)
	return ret
}
