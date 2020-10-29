module num

import vsl.vmath as math

// NdArray stored the data necessary to operate on a contiguous
// data buffer in n dimensions
pub struct NdArray {
mut:
	storage CpuStorage
pub:
	size    int
pub mut:
	flags   ArrayFlags
	shape   []int
	strides []int
	ndims   int
}

// allocate_cpu allocates an ndarray onto CPU storage with a given shape
// and memory layout
pub fn allocate_cpu(shape []int, order string) NdArray {
	if shape.len == 0 {
		return NdArray{
			storage: cpu(1)
			shape: []
			strides: [1]
			ndims: 0
			flags: default_flags(order, 1)
			size: 1
		}
	}
	size := shape_size(shape)
	storage := cpu(size)
	return NdArray{
		storage: storage
		shape: shape
		strides: strides(shape, order)
		ndims: shape.len
		flags: default_flags(order, shape.len)
		size: size
	}
}

// str returns the string representation of an ndarray.
pub fn (t NdArray) str() string {
	return array2string(t, ', ', '')
}

pub fn (t NdArray) buffer() &f64 {
	return t.storage.buffer
}

pub fn (t NdArray) f64_array() []f64 {
	mut a := []f64{cap: t.size}
	a.push_many(t.buffer(), t.size)
	return a
}

// get returns a scalar value at a provided index
pub fn (n NdArray) get(index []int) f64 {
	return n.storage.get(index, n.shape, n.strides)
}

// set sets a scalar value at a provided index
pub fn (mut n NdArray) set(index []int, value f64) {
	n.storage.set(index, n.shape, n.strides, value)
}

// fill sets all the values in an ndarray to the provided
// value
pub fn (n NdArray) fill(value f64) {
	for iter := n.iter(); !iter.done; iter.next() {
		unsafe {
			*iter.ptr = value
		}
	}
}

// assign sets the values of an ndarray equal to the values of another
// NdArray of the same shape
pub fn (n NdArray) assign(other NdArray) {
	otherb := broadcast_if(other, n.shape)
	for iter := n.iter2(otherb); !iter.done; iter.next() {
		unsafe {
			*iter.ptr_a = *iter.ptr_b
		}
	}
}

// slice returns a view of an ndarray from a variadic list
// of indexing operations.  The returned view does not
// own its new data, but shares data with another ndarray
pub fn (t NdArray) slice(idx [][]int) NdArray {
	mut newshape := t.shape.clone()
	mut newstrides := t.strides.clone()
	mut newflags := default_flags('C', t.ndims)
	newflags.owndata = false
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
		if dex.len == 1 {
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
	mut ptr := 0
	mut i := 0
	for i < indexer.len {
		ptr += t.strides[i] * indexer[i]
		i++
	}
	mut ret := NdArray{
		shape: newshape_
		strides: newstrides_
		ndims: newshape_.len
		size: shape_size(newshape_)
		storage: t.storage.offset(ptr)
		flags: newflags
	}
	ret.update_flags(all_flags())
	return ret
}

// slice_hilo returns a view of an array from a list of starting
// indices and a list of closing indices.  This is slightly less
// general than slice, but is used for internal methods since
// it doesn't use variadic arguments.  This method will be used
// until V has better support for 2D arrays.
pub fn (t NdArray) slice_hilo(idx1 []int, idx2 []int) NdArray {
	mut newshape := t.shape.clone()
	mut newstrides := t.strides.clone()
	mut newflags := default_flags('C', t.ndims)
	newflags.owndata = false
	mut ii := 0
	idx_start := pad_with_zeros(idx1, t.ndims)
	idx_end := pad_with_max(idx2, t.shape, t.ndims)
	mut idx := []int{}
	for ii < t.ndims {
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
		ii++
	}
	newshape_, newstrides_ := filter_shape_not_strides(newshape, newstrides)
	mut ptr := 0
	mut i := 0
	for i < t.ndims {
		ptr += t.strides[i] * idx[i]
		i++
	}
	mut ret := NdArray{
		shape: newshape_
		strides: newstrides_
		ndims: newshape_.len
		size: shape_size(newshape_)
		storage: t.storage.offset(ptr)
		flags: newflags
	}
	ret.update_flags(all_flags())
	return ret
}

// reshape returns an ndarray with a new shape, as a
// view if possible.  If a view is not possible, copies
// data and returns a c-contiguous array
pub fn (t NdArray) reshape(shape []int) NdArray {
	mut ret := t.view()
	mut newshape := shape.clone()
	mut newsize := 1
	cur_size := t.size
	mut autosize := -1
	for i, val in newshape {
		if val < 0 {
			if autosize >= 0 {
				panic('Only one dimension can be autosized')
			}
			autosize = i
		} else {
			newsize *= val
		}
	}
	if autosize >= 0 {
		newshape = newshape.clone()
		newshape[autosize] = cur_size / newsize
		newsize *= newshape[autosize]
	}
	if newsize != cur_size {
		panic('Cannot reshape')
	}
	mut newstrides := [0].repeat(newshape.len)
	if t.flags.fortran && !t.flags.contiguous {
		newstrides = fstrides(newshape)
	} else {
		newstrides = cstrides(newshape)
	}
	if t.flags.contiguous || t.flags.fortran {
		ret.shape = newshape
		ret.strides = newstrides
		ret.ndims = newshape.len
	} else {
		ret = t.copy('C')
		ret.shape = newshape
		ret.strides = newstrides
		ret.ndims = newshape.len
	}
	ret.update_flags(all_flags())
	return ret
}

// transpose permutes the axes of an ndarray in a specified
// order and returns a view of the data
pub fn (t NdArray) transpose(order []int) NdArray {
	mut ret := t.view()
	n := order.len
	if n != t.ndims {
		panic('Bad number of dimensions')
	}
	mut permutation := [0].repeat(32)
	mut reverse_permutation := [-1].repeat(32)
	mut i := 0
	for i < n {
		mut axis := order[i]
		if axis < 0 {
			axis = t.ndims + axis
		}
		if axis < 0 || axis >= t.ndims {
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
	ret.update_flags(all_flags())
	return ret
}

// t returns a ful transpose of an ndarray, with the axes
// reversed
pub fn (t NdArray) t() NdArray {
	order := irange(0, t.ndims)
	return t.transpose(order.reverse())
}

// swapaxes returns a view of an ndarray with two axes
// swapped.
pub fn (t NdArray) swapaxes(a1 int, a2 int) NdArray {
	mut order := irange(0, t.ndims)
	tmp := order[a1]
	order[a1] = order[a2]
	order[a2] = tmp
	return t.transpose(order)
}

// ravel returns a flattened view of an ndarray if possible,
// otherwise a flattened copy
pub fn (t NdArray) ravel() NdArray {
	return t.reshape([-1])
}

// returns a copy of an array with a particular memory
// layout, either c-contiguous or fortran contiguous
pub fn (t NdArray) copy(order string) NdArray {
	mut ret := allocate_cpu(t.shape, order)
	for iter := ret.iter2(t); !iter.done; iter.next() {
		unsafe {
			*iter.ptr_a = *iter.ptr_b
		}
	}
	ret.update_flags(all_flags())
	return ret
}

// returns a view of an ndarray, identical to the
// parent but not owning its own data.
pub fn (t NdArray) view() NdArray {
	mut newflags := dup_flags(t.flags)
	newflags.owndata = false
	return NdArray{
		shape: t.shape.clone()
		strides: t.strides.clone()
		storage: t.storage
		flags: newflags
		size: t.size
		ndims: t.ndims
	}
}

// diagonal returns a view of the diagonal entries
// of a two dimensional ndarray
pub fn (t NdArray) diagonal() NdArray {
	nel := shape_min(t.shape)
	newshape := [nel]
	newstrides := [shape_sum(t.strides)]
	mut newflags := dup_flags(t.flags)
	newflags.owndata = false
	mut ret := NdArray{
		storage: t.storage
		ndims: 1
		shape: newshape
		strides: newstrides
		flags: newflags
		size: nel
	}
	ret.update_flags(all_flags())
	return ret
}

// add adds two ndarrays elementwise
pub fn (t NdArray) add(other NdArray) NdArray {
	return add(t, other)
}

// subtract subtracts two ndarrays elementwise
pub fn (t NdArray) subtract(other NdArray) NdArray {
	return subtract(t, other)
}

// divide dives two ndarrays elementwise
pub fn (t NdArray) divide(other NdArray) NdArray {
	return divide(t, other)
}

// multiply multiplies two ndarrays elementwise
pub fn (t NdArray) multiply(other NdArray) NdArray {
	return multiply(t, other)
}

// min returns the minimum value in an ndarray
pub fn (t NdArray) min() f64 {
	return min(t)
}

// max returns the maximum value in an ndarray
pub fn (t NdArray) max() f64 {
	return max(t)
}

// min_axis returns the minimum value of an ndarray along an axis
pub fn (t NdArray) min_axis(axis int) NdArray {
	return min_axis(t, axis)
}

// max_axis returns the maximum value of an ndarray along an axis
pub fn (t NdArray) max_axis(axis int) NdArray {
	return max_axis(t, axis)
}

// sum_axis returns the sum of an ndarray along an axis
pub fn (t NdArray) sum_axis(axis int) NdArray {
	return sum_axis(t, axis)
}

// sum returns the sum of an ndarray
pub fn (t NdArray) sum() f64 {
	return sum(t)
}

// prod returns the product of an ndarray
pub fn (t NdArray) prod() f64 {
	return prod(t)
}

// ptp returns max - min of an ndarray
pub fn (t NdArray) ptp() f64 {
	return ptp(t)
}
