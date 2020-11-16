module vtl

// get returns a scalar value from a Tensor at the provided index
[inline]
pub fn (t Tensor) get(index []int) voidptr {
	offset := t.offset(index)
	unsafe {
		return storage_get(t.data, offset)
	}
}

// offset returns a pointer to a Tensor's data at
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
