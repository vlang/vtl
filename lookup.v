module vtl

import vtl.storage
import vsl.vmath as math

// get returns a scalar value from a Tensor at the provided index
[inline]
pub fn (t &Tensor<T>) get<T>(index []int) T {
	offset := t.offset_index(index)
	return storage.storage_get<T>(t.data, offset)
}

// offset_index returns the index to a Tensor's data at
// a given index
[inline]
pub fn (t &Tensor<T>) offset_index<T>(index []int) int {
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

// strided_offset_index returns the index of the starting offset
// for arrays that may be negatively strided
pub fn (t &Tensor<T>) strided_offset_index<T>() int {
	mut offset := 0
	for i in 0 .. t.rank() {
		if t.strides[i] < 0 {
			offset += (t.shape[i] - 1) * int(math.abs(t.strides[i]))
		}
	}
	return offset
}
