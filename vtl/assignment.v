module vtl

import vtl.etype
import vtl.storage

// set copies a scalar value into a Tensor at the provided index
[inline]
pub fn (mut t Tensor) set(index []int, val etype.Num) {
	offset := t.offset(index)
	storage.storage_set(t.data, offset, val)
}

// fill fills an entire Tensor with a given value
[inline]
pub fn (mut t Tensor) fill(val etype.Num) {
	storage.storage_fill(t.data, val)
}

// assign sets the values of an Tensor equal to the values of another
// Tensor of the same shape
pub fn (mut t Tensor) assign(other Tensor) {
	otherb := other.broadcast_to(t.shape)
	mut iter := otherb.iterator()
	for val in iter {
		storage.storage_set(t.data, iter.pos, val)
	}
}
