module vtl

import arrays

// assert_square_matrix panics if the given tensor is not a square matrix
[inline]
fn (t &Tensor[T]) assert_square_matrix[T]() ? {
	if t.is_square_matrix() {
		return error('Matrix is not square')
	}
}

// assert_square_matrix panics if the given tensor is not a matrix
[inline]
fn (t &Tensor[T]) assert_matrix[T]() ? {
	if t.is_matrix() {
		return error('Tensor is not two-dimensional')
	}
}

// irange returns an array between start and stop, incremented by 1
fn irange(start int, stop int) []int {
	mut ret := []int{cap: stop - start}
	for i in start .. stop {
		ret << i
	}
	return ret
}

// assert_rank ensures that a Tensor has a given rank
[inline]
fn (t &Tensor[T]) assert_rank[T](n int) ? {
	if n != t.rank() {
		return error('Bad number of dimensions')
	}
}

// assert_min_rank ensures that a Tensor has at least a given rank
[inline]
fn (t &Tensor[T]) assert_min_rank[T](n int) ? {
	if n > t.rank() {
		return error('Bad number of dimensions')
	}
}

// ensure_memory sets a correct memory layout to a given tensor
[inline]
pub fn (mut t Tensor[T]) ensure_memory[T]() {
	if t.is_col_major() {
		if !t.is_col_major_contiguous() {
			t.memory = .row_major
		}
	}
	if t.is_contiguous() {
		if t.rank() > 1 {
			t.memory = .row_major
		}
	}
}

// assert_shape_off_axis ensures that the shapes of Tensors match
// for concatenation, except along the axis being joined
fn assert_shape_off_axis[T](ts []&Tensor[T], axis int, shape []int) ?[]int {
	mut retshape := shape.clone()
	for t in ts {
		if t.shape.len != retshape.len {
			return error('All inputs must share the same number of axes')
		}
		mut i := 0
		for i < shape.len {
			if i != axis && t.shape[i] != shape[i] {
				return error('All inputs must share a shape off axis')
			}
			i++
		}
		retshape[axis] += t.shape[axis]
	}
	return retshape
}

// assert_shape ensures that the shapes of Tensors match
// for each tensor given list of tensors
[inline]
fn assert_shape[T](shape []int, ts []&Tensor[T]) ? {
	for t in ts {
		if shape != t.shape {
			return error('All shapes must be equal')
		}
	}
}

// is_col_major_contiguous checks if an array is contiguous with a col-major
// memory layout
fn is_col_major_contiguous(shape []int, strides []int, ndims int) bool {
	if ndims == 0 {
		return true
	}
	if ndims == 1 {
		return shape[0] == 1 || strides[0] == 1
	}
	mut sd := 1
	for i in 0 .. ndims {
		dim := shape[i]
		if dim == 0 {
			return true
		}
		if strides[i] != sd {
			return false
		}
		sd *= dim
	}
	return true
}

// is_row_major_contiguous checks if an array is contiguous with a row-major
// memory layout
fn is_row_major_contiguous(shape []int, strides []int, ndims int) bool {
	if ndims == 0 {
		return true
	}
	if ndims == 1 {
		return shape[0] == 1 || strides[0] == 1
	}
	mut sd := 1
	mut i := ndims - 1
	for i > 0 {
		dim := shape[i]
		if dim == 0 {
			return true
		}
		if strides[i] != sd {
			return false
		}
		sd *= dim
		i--
	}
	return true
}

// clip_axis is just a check for negative axes, so that negative axes can be inferred
fn clip_axis(axis int, size int) ?int {
	mut next_axis := axis
	if next_axis < 0 {
		next_axis += size
	}
	if next_axis < 0 || next_axis > size {
		return error('axis out of range')
	}
	return next_axis
}

// strides_from_shape returns the strides from a shape and memory format
fn strides_from_shape(shape []int, memory MemoryFormat) []int {
	mut accum := 1
	mut result := []int{len: shape.len}
	if memory == .row_major {
		for i := shape.len - 1; i >= 0; i-- {
			result[i] = accum
			accum *= shape[i]
		}
		return result
	}
	for i in 0 .. shape.len {
		result[i] = accum
		accum *= shape[i]
	}
	return result
}

// size_from_shape returns the allocated size for a given shape
fn size_from_shape(shape []int) int {
	mut accum := 1
	for i in shape {
		accum *= i
	}
	return accum
}

// shape_with_autosize returns a new shape and size with autosize
// applied if needed
fn shape_with_autosize(shape []int, size int) ?([]int, int) {
	mut newshape := shape.clone()
	mut newsize := 1
	mut autosize := -1
	for i, val in newshape {
		if val < 0 {
			if autosize >= 0 {
				return error('Only one dimension can be autosized')
			}
			autosize = i
		} else {
			newsize *= val
		}
	}
	if autosize >= 0 {
		newshape = newshape.clone()
		newshape[autosize] = size / newsize
		newsize *= newshape[autosize]
	}

	if size != newsize {
		return error("Shape and size don't match")
	}

	return newshape, newsize
}

// filter_shape_not_strides removes 0 size dimensions from the shape
// and strides of an array
fn filter_shape_not_strides(shape []int, strides []int) ?([]int, []int) {
	mut newshape := []int{}
	mut newstrides := []int{}
	for i := 0; i < shape.len; i++ {
		if shape[i] != 0 {
			newshape << shape[i]
			newstrides << strides[i]
		}
	}
	return newshape, newstrides
}

// pad_with_zeros pads a shape with zeros to support an indexing
// operation
fn pad_with_zeros(pad []int, ndims int) []int {
	diff := ndims - pad.len
	mut newpad := pad.clone()
	mut i := 0
	for i < diff {
		newpad << 0
		i++
	}
	return newpad
}

// pad_with_max pads a shape with the maximum axis value to support
// an indexing operation
fn pad_with_max(pad []int, shape []int, ndims int) []int {
	mut newpad := pad.clone()
	diff := ndims - pad.len
	if diff > 0 {
		newpad << shape[pad.len..]
	}
	return newpad
}

// iarray_min returns the minimum value of a given array of int values
// the use of arrays.min give us an optimizad version of this function
[inline]
fn iarray_min(arr []int) int {
	return arrays.min[int](arr) or { 0 }
}

// iarray_sum returns the sum value of a given array of int values
fn iarray_sum(arr []int) int {
	mut ret := 0
	for i in arr {
		ret += i
	}
	return ret
}

// iarray_prod returns the prod value of a given array of int values
fn iarray_prod(arr []int) int {
	mut ret := 0
	for i in arr {
		ret *= i
	}
	return ret
}
