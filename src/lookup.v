module vtl

// get returns a scalar value from a Tensor at the provided index
@[inline]
pub fn (t &Tensor[T]) get[T](index []int) T {
	offset := t.offset_index(index)
	return t.data.get[T](offset)
}

// get_nth returns a scalar value from a Tensor at the provided index
@[inline]
pub fn (t &Tensor[T]) get_nth[T](n int) T {
	index := t.nth_index(n)
	return t.get[T](index)
}

// offset_index returns the index to a Tensor's data at
// a given index
@[inline]
pub fn (t &Tensor[T]) offset_index[T](index []int) int {
	mut offset := 0
	for i in 0 .. t.rank() {
		mut j := index[i]
		if j < 0 {
			j += t.shape[i]
		}
		offset += j * t.strides[i]
		if t.strides[i] < 0 {
			offset += t.shape[i] - 1
		}
	}

	if offset < 0 {
		offset = t.size - 1 + offset
	}

	return offset
}

// nth_index returns the nth index of a Tensor's shape
// for `n == 2` and a `shape` of `[2, 2]` the _nth index_ is `[1, 0]`
// and for a `shape` of `[2, 3]` and `n == 3` the _nth index_ is `[0, 1, 1]`
// in sorted order.
pub fn (t &Tensor[T]) nth_index[T](n int) []int {
	rank := t.rank()
	mut index := []int{len: rank}
	for i in 0 .. rank {
		index[i] = 0
	}
	mut i := 0
	for {
		if i == n {
			return index
		}
		i += 1
		for j := rank - 1; j >= 0; j -= 1 {
			if index[j] < t.shape[j] - 1 {
				index[j] += 1
				break
			}
			index[j] = 0
		}
	}

	return index
}
