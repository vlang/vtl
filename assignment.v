module vtl

import vtl.storage

// set copies a scalar value into a Tensor at the provided index
[inline]
pub fn (t Tensor<T>) set<T>(index []int, val T) {
	offset := t.offset_index(index)
	storage.storage_set<T>(t.data, offset, val)
}

// fill fills an entire Tensor with a given value
[inline]
pub fn (t Tensor<T>) fill<T>(val T) {
	storage.storage_fill<T>(t.data, val)
}
