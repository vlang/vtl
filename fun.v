module vtl

import math
import vtl.etype
import vtl.storage

pub type MapFn = fn (x etype.Num, i int) etype.Num

pub type ApplyFn = fn (x etype.Num, i int) etype.Num

pub type NMapFn = fn (x []etype.Num, i int) etype.Num

pub type NApplyFn = fn (x []etype.Num, i int) etype.Num

// map maps a function to a given Tensor retuning a new Tensor with same shape
pub fn (t Tensor) map(f MapFn) Tensor {
	mut ret := new_tensor_like(t)
	mut iter := t.iterator()
	mut pos := iter.pos
	for _ in 0 .. ret.size() {
		if val := iter.next() {
			next_val := f(val, pos)
			storage.storage_set(ret.data, pos, next_val)
			pos = iter.pos
		}
	}
	return ret
}

// map_as maps a function to a given Tensor retuning a new Tensor with same shape
pub fn (t Tensor) map_as<T>(f MapFn) Tensor {
	mut ret := new_tensor_like_with_etype(t, T.name)
	mut iter := t.iterator()
	mut pos := iter.pos
	for _ in 0 .. ret.size() {
		if val := iter.next() {
			next_val := f(val, pos)
			storage.storage_set(ret.data, pos, next_val)
			pos = iter.pos
		}
	}
	return ret
}

// apply applies a function to each element of a given Tensor
pub fn (t Tensor) apply(f ApplyFn) {
	mut iter := t.iterator()
	mut pos := iter.pos
	for _ in 0 .. t.size() {
		if val := iter.next() {
			next_val := f(val, pos)
			storage.storage_set(t.data, pos, next_val)
			pos = iter.pos
		}
	}
}

// nmap maps a function to a given list of Tensor retuning a new Tensor with same shape
pub fn (t Tensor) nmap(f NMapFn, ts ...Tensor) Tensor {
	mut ret := new_tensor_like(t)
	mut iters := t.iterators(...ts)
	for i in 0 .. t.size {
		val := f(iters.next(), i)
		storage.storage_set(ret.data, i, val)
	}
	return ret
}

// nmap_as maps a function to a given list of Tensor retuning a new Tensor with same shape
pub fn (t Tensor) nmap_as<T>(f NMapFn, ts ...Tensor) Tensor {
	mut ret := new_tensor_like_with_etype(t, T.name)
	mut iters := t.iterators(...ts)
	for i in 0 .. t.size {
		val := f(iters.next(), i)
		storage.storage_set(ret.data, i, val)
	}
	return ret
}

// apply applies a function to each element of a given Tensor
pub fn (t Tensor) napply(f NApplyFn, ts ...Tensor) {
	mut iters := t.iterators(...ts)
	for i in 0 .. t.size {
		val := f(iters.next(), i)
		storage.storage_set(t.data, i, val)
	}
}

// checks if two Tensors are equal
pub fn (t Tensor) equal(other Tensor) bool {
	if t.shape != other.shape {
		return false
	}
	mut iters := t.iterators(...[other])
	for _ in 0 .. t.size {
		vals := iters.next()
		if vals[0] != vals[1] {
			return false
		}
	}
	return true
}

// diagonal returns a view of the diagonal entries
// of a two dimensional tensor
pub fn (t Tensor) diagonal() Tensor {
	nel := iarray_min(t.shape)
	newshape := [nel]
	newstrides := [iarray_sum(t.strides)]
	return Tensor{
		data: t.data
		shape: newshape
		strides: newstrides
		size: nel
	}
}

// ravel returns a flattened view of an Tensor if possible,
// otherwise a flattened copy
[inline]
pub fn (t Tensor) ravel() Tensor {
	return t.reshape([-1])
}

// reshape returns an Tensor with a new shape
pub fn (t Tensor) reshape(shape []int) Tensor {
	size := size_from_shape(shape)
	newshape, newsize := shape_with_autosize(shape, size)
	if newsize != size {
		panic('reshape: Cannot reshape')
	}
	mut ret := new_tensor_like_with_shape(t, newshape)
	ret.data = t.data
	return ret
}

// transpose permutes the axes of an tensor in a specified
// order and returns a view of the data
pub fn (t Tensor) transpose(order []int) Tensor {
	mut ret := t.view()
	n := order.len
	assert_rank(t, n)
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
pub fn (t Tensor) t() Tensor {
	order := irange(0, t.rank())
	return t.transpose(order.reverse())
}

// swapaxes returns a view of an tensor with two axes
// swapped.
pub fn (t Tensor) swapaxes(a1 int, a2 int) Tensor {
	mut order := irange(0, t.rank())
	tmp := order[a1]
	order[a1] = order[a2]
	order[a2] = tmp
	return t.transpose(order)
}

// slice returns a tensor from a variadic list of indexing operations
pub fn (t Tensor) slice(idx ...[]int) Tensor {
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
	mut ret := Tensor{
		shape: newshape_.clone()
		strides: newstrides_.clone()
		size: size_from_shape(newshape_)
		data: t.data.offset(offset)
		memory: .colmajor
	}
	ensure_memory(mut ret)
	return ret
}
