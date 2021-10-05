module vtl

// set copies a scalar value into a Tensor at the provided index
[inline]
pub fn (mut t Tensor<T>) set<T>(index []int, val T) {
	offset := t.offset_index(index)
	t.data.set<T>(offset, val)
}

// fill fills an entire Tensor with a given value
[inline]
pub fn (mut t Tensor<T>) fill<T>(val T) {
	t.data.fill<T>(val)
}

// assign sets the values of an Tensor equal to the values of another
// Tensor of the same shape
pub fn (mut t Tensor<T>) assign<T>(other &Tensor<T>) {
	mut iters := t.iterators<T>([other])
	for {
		vals, pos := iterators_next<T>(mut iters) or { break }
		t.data.set<T>(pos, vals[1])
	}
}
