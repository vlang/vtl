module vtl

import storage

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

// assign sets the values of an Tensor equal to the values of another
// Tensor of the same shape
pub fn (mut t Tensor<T>) assign<T>(other &Tensor<T>) {
	mut iters := iterators<T>([t, other])
	mut t_iter := iters[0]
	mut pos := t_iter.pos
	for {
		vals := iterators_next<T>(mut iters) or { break }
		storage.storage_set<T>(t.data, pos, vals[1])
		pos = t_iter.pos
	}
}
