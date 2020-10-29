module num

import vsl.vmath as math

// shape_size returns the number of elements represented
// by a provided shape
fn shape_size(shape []int) int {
	mut sz := 1
	for s in shape {
		sz *= s
	}
	return sz
}

// strides returns contiguous strides for a given shape
// and memory layout
fn strides(shape []int, order string) []int {
	if order == 'C' {
		return cstrides(shape)
	} else {
		return fstrides(shape)
	}
}

// cstrides returns row-major strides for a given shape
fn cstrides(shape []int) []int {
	mut sz := 1
	mut strides := shape.clone()
	for i := 0; i < shape.len; i++ {
		strides[shape.len - i - 1] = sz
		sz *= shape[shape.len - i - 1]
	}
	return strides
}

// fstrides returns col-major strides for a given shape
fn fstrides(shape []int) []int {
	mut sz := 1
	mut strides := shape.clone()
	for i := 0; i < shape.len; i++ {
		strides[i] = sz
		sz *= shape[i]
	}
	return strides
}

// shape_compare asserts that two shapes are equal
pub fn shape_compare(s1 []int, s2 []int) bool {
	if s1.len != s2.len {
		return false
	}
	for i := 0; i < s1.len; i++ {
		if s1[i] != s2[i] {
			return false
		}
	}
	return true
}

// is_fortran_contiguous checks if an array is contiguous with a col-major
// memory layout
fn is_fortran_contiguous(shape []int, strides []int, ndims int) bool {
	if ndims == 0 {
		return true
	}
	if ndims == 1 {
		return shape[0] == 1 || strides[0] == 1
	}
	mut sd := 1
	mut i := 0
	for i < ndims {
		dim := shape[i]
		if dim == 0 {
			return true
		}
		if strides[i] != sd {
			return false
		}
		sd *= dim
		i++
	}
	return true
}

// is_contiguous checks if an array is contiguous with a row-major
// memory layout
fn is_contiguous(shape []int, strides []int, ndims int) bool {
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

// assert_shape_off_axis ensures that the shapes of ndarrays match
// for concatenation, except along the axis being joined.
fn assert_shape_off_axis(ts []NdArray, axis int, shape []int) []int {
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

// irange returns an array between start and stop, incremented by
// 1
pub fn irange(start int, stop int) []int {
	mut ret := []int{}
	for i in start .. stop {
		ret << i
	}
	return ret
}

// shape_min returns the minimum value from the shape of an ndarray
fn shape_min(shape []int) int {
	mut mn := 0
	for i, dim in shape {
		if i == 0 {
			mn = dim
		}
		if dim < mn {
			mn = dim
		}
	}
	return mn
}

// shape_sum sums a shape to get the total size of dimensions in
// an ndarray
fn shape_sum(shape []int) int {
	mut ret := 0
	for i in shape {
		ret += i
	}
	return ret
}

// applies a reduction operation to an axis
fn axis_op(n NdArray, axis int, op TwoParamsFn) NdArray {
	mut iter := n.axis(axis)
	ret := iter.next().copy('C')
	for i := 1; i < iter.size; i++ {
		apply2(ret, iter.next(), op)
	}
	return ret
}

fn axis_op_dims(n NdArray, axis int, op TwoParamsFn) NdArray {
	mut iter := n.axis_with_dims(axis)
	ret := iter.next().copy('C')
	for i := 1; i < iter.size; i++ {
		apply2(ret, iter.next(), op)
	}
	return ret
}

// checks if two floating point ndarrays are close within a tolerance
pub fn allclose(a NdArray, b NdArray) bool {
	rtol := 1e-5
	atol := 1e-8
	if !shape_compare(a.shape, b.shape) {
		return false
	} else {
		for iter := a.iter2(b); !iter.done; iter.next() {
			i := *iter.ptr_a
			j := *iter.ptr_b
			if math.abs(i - j) > (atol + rtol * math.abs(j)) {
				return false
			}
		}
	}
	return true
}

fn assert_shape(shape []int, arrs []NdArray) {
	for arr in arrs {
		if !shape_compare(shape, arr.shape) {
			panic('All shapes must be equal')
		}
	}
}
