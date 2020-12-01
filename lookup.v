module vtl

import math

// get returns a scalar value from a Tensor at the provided index
[inline]
pub fn (t Tensor) get(index []int) Num {
	offset := t.offset(index)
	return storage_get(t.data, offset, t.etype)
}

// offset returns the index to a Tensor's data at
// a given index
[inline]
pub fn (t Tensor) offset(index []int) int {
	mut offset := 0
	for i in 0 .. t.rank() {
		mut j := index[i]
		if j < 0 {
			j += t.shape[i]
		}
		offset += j * t.strides[i]
	}
	return offset
}

// stride offset returns the index of the starting offset
// for arrays that may be negatively strided
pub fn (t Tensor) strided_offset(shape []int, strides []int) int {
	mut offset := 0
	for i in 0 .. t.rank() {
		if strides[i] < 0 {
			unsafe {
				offset += (shape[i] - 1) * int(math.abs(strides[i]))
			}
		}
	}
	return offset
}
