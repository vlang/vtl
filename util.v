module vtl

import arrays

// strides_from_shape returns the strides from a shape and memory format
fn strides_from_shape(shape []int, memory MemoryFormat) []int {
	mut accum := 1
	mut result := []int{len: shape.len}
	if memory == .rowmajor {
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
fn shape_with_autosize(shape []int, size int) ([]int, int) {
	mut newshape := shape
	mut newsize := size
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
		newshape[autosize] = size / newsize
		newsize *= newshape[autosize]
	}
	return newshape, newsize
}

// filter_shape_not_strides removes 0 size dimensions from the shape
// and strides of an array
fn filter_shape_not_strides(shape []int, strides []int) ([]int, []int) {
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
	mut newpad := pad
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
	mut newpad := pad
	diff := ndims - pad.len
	if diff > 0 {
		newpad << shape[pad.len..]
	}
	return newpad
}

// assert_rank ensures that a Tensor has a given rank
[inline]
fn assert_rank(t Tensor, n int) {
	if n != t.rank() {
		panic('Bad number of dimensions')
	}
}

// assert_shape_off_axis ensures that the shapes of Tensors match
// for concatenation, except along the axis being joined
fn assert_shape_off_axis(ts []Tensor, axis int, shape []int) []int {
	mut retshape := shape.clone()
	for t in ts {
		if t.shape.len != retshape.len {
			panic('All inputs must share the same number of axes')
		}
		mut i := 0
		for i < shape.len {
			if (i != axis) && (t.shape[i] != shape[i]) {
				panic('All inputs must share a shape off axis')
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
fn assert_shape(shape []int, ts []Tensor) {
	for t in ts {
		if shape != t.shape {
			panic('All shapes must be equal')
		}
	}
}

// ensure_memory sets a correct memory layout to a given tensor
[inline]
fn ensure_memory(mut t Tensor) {
	if t.is_colmajor() {
		if !t.is_colmajor_contiguous() {
			t.memory = .rowmajor
		}
	}
	if t.is_contiguous() {
		if t.rank() > 1 {
			t.memory = .rowmajor
		}
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

// iarray_min returns the minimum value of a given array of int values
// the use of arrays.min give us an optimizad version of this function
[inline]
fn iarray_min(arr []int) int {
	return arrays.min<int>(arr)
}

// iarray_sum returns the sum value of a given array of int values
fn iarray_sum(arr []int) int {
	mut ret := 0
	for i in arr {
		ret += i
	}
	return ret
}
