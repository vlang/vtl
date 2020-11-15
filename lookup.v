module vtl

// get returns a scalar value from a Tensor at the provided index
[inline]
pub fn (t Tensor) get(index []int) voidptr {
	mut offset := t.offset(index)
	unsafe {
		return t.data.get(offset)
	}
}

// offset returns a pointer to a Tensor's data at
// a given index
[inline]
pub fn (t Tensor) offset(index []int) int {
	mut offset := 0
	for i := 0; i < t.rank(); i++ {
		mut j := index[i]
		if j < 0 {
			j += t.shape[i]
		}
		offset += j * t.strides[i]
	}
	return offset
}
