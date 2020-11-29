module vtl

// rank returns the number of dimensions of a given Tensor
pub fn (t Tensor) rank() int {
	return t.shape.len
}

// size returns the number of allocated elements for a given tensor
pub fn (t Tensor) size() int {
	return t.size
}

// is_matrix returns if a Tensor is a nxm matrix or not
[inline]
pub fn (t Tensor) is_matrix() bool {
	return t.rank() == 2
}

// is_matrix returns if a Tensor is a square matrix or not
[inline]
pub fn (t Tensor) is_square_matrix() bool {
	return t.rank() == 2 && t.shape[0] == t.shape[1]
}

// is_matrix returns if a Tensor is a square 1D vector or not
[inline]
pub fn (t Tensor) is_vector() bool {
	return t.rank() == 1
}

// is_rowmajor returns if a Tensor is supposed to store its data in Row-Major
// order
[inline]
pub fn (t Tensor) is_rowmajor() bool {
	// @todo: we need to ensure that t.memory is the source of truth
	return t.memory == .rowmajor
}

// is_colmajor returns if a Tensor is supposed to store its data in Col-Major
// order
[inline]
pub fn (t Tensor) is_colmajor() bool {
	// @todo: we need to ensure that t.memory is the source of truth
	return t.memory == .colmajor
}

// is_rowmajor verifies if a Tensor stores its data in Row-Major
// order
[inline]
pub fn (t Tensor) is_rowmajor_contiguous() bool {
	if t.rank() == 1 && t.strides[0] != 1 {
		return false
	}
	mut z := 1
	for i := t.shape.len - 1; i > 0; i-- {
		if t.shape[i] != 1 && t.strides[i] != z {
			return false
		}
		z *= t.shape[i]
	}
	return true
}

// is_colmajor verifies if a Tensor stores its data in Col-Major
// order
[inline]
pub fn (t Tensor) is_colmajor_contigous() bool {
	if t.rank() == 1 && t.strides[0] != 1 {
		return false
	}
	mut z := 1
	for i := 0; i < t.shape.len; i++ {
		if t.shape[i] != 1 && t.strides[i] != z {
			return false
		}
		z *= t.shape[i]
	}
	return true
}

// is_contiguous verifies that a Tensor is contiguous independent of
// memory layout
[inline]
pub fn (t Tensor) is_contiguous() bool {
	return t.is_rowmajor_contiguous() || t.is_colmajor_contigous()
}
