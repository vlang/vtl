module vtl

import vtl.storage

// VclTensor is the main structure defined by VTL to manage N Dimensional data
[heap]
pub struct VclTensor[T] {
pub mut:
	data    &storage.VclStorage[T]
	memory  MemoryFormat
	size    int
	shape   []int
	strides []int
}

// vcl returns a VclTensor from a Tensor
pub fn (t &Tensor[T]) vcl(params storage.VclStorageParams) ?&VclTensor[T] {
	row_tensor := t.copy(.row_major)
	cldata := row_tensor.data.vcl(params)?
	return &VclTensor[T]{
		data: cldata
		memory: row_tensor.memory
		size: row_tensor.size
		shape: row_tensor.shape
		strides: row_tensor.strides
	}
}

// cpu returns a Tensor from a VclTensor
pub fn (t &VclTensor[T]) cpu() ?&Tensor[T] {
	data := t.data.cpu()?
	return &Tensor[T]{
		data: data
		memory: t.memory
		size: t.size
		shape: t.shape
		strides: t.strides
	}
}

// vcl returns a VclTensor from a VclTensor
[inline]
pub fn (t &VclTensor[T]) vcl() ?&VclTensor[T] {
	return t
}

// release releases the VclTensor's data
pub fn (t &VclTensor[T]) release() ? {
	return t.data.release()
}

// // str returns the string representation of a VclTensor
// [inline]
// pub fn (t &VclTensor<T>) str() string {
// 	return t.data.str()
// }

// rank returns the number of dimensions of a given VclTensor
pub fn (t &VclTensor[T]) rank() int {
	return t.shape.len
}

// size returns the number of allocated elements for a given tensor
pub fn (t &VclTensor[T]) size() int {
	return t.size
}

// is_matrix returns if a VclTensor is a nxm matrix or not
[inline]
pub fn (t &VclTensor[T]) is_matrix() bool {
	return t.rank() == 2
}

// is_matrix returns if a VclTensor is a square matrix or not
[inline]
pub fn (t &VclTensor[T]) is_square_matrix() bool {
	return t.rank() == 2 && t.shape[0] == t.shape[1]
}

// is_matrix returns if a VclTensor is a square 1D vector or not
[inline]
pub fn (t &VclTensor[T]) is_vector() bool {
	return t.rank() == 1
}

// is_row_major returns if a VclTensor is supposed to store its data in Row-Major
// order
[inline]
pub fn (t &VclTensor[T]) is_row_major() bool {
	// @todo: we need to ensure that t.memory is the source of truth
	return t.memory == .row_major
}

// is_col_major returns if a VclTensor is supposed to store its data in Col-Major
// order
[inline]
pub fn (t &VclTensor[T]) is_col_major() bool {
	// @todo: we need to ensure that t.memory is the source of truth
	return t.memory == .col_major
}

// is_row_major verifies if a VclTensor stores its data in Row-Major
// order
[inline]
pub fn (t &VclTensor[T]) is_row_major_contiguous() bool {
	return is_row_major_contiguous(t.shape, t.strides, t.rank())
}

// is_col_major verifies if a VclTensor stores its data in Col-Major
// order
[inline]
pub fn (t &VclTensor[T]) is_col_major_contiguous() bool {
	return is_col_major_contiguous(t.shape, t.strides, t.rank())
}

// is_contiguous verifies that a VclTensor is contiguous independent of
// memory layout
[inline]
pub fn (t &VclTensor[T]) is_contiguous() bool {
	return t.is_row_major_contiguous() || t.is_col_major_contiguous()
}
