module vtl

import storage

// MemoryFormat is a sum type that lists the possible memory layouts
pub enum MemoryFormat {
	row_major
	col_major
}

// Tensor is the main structure defined by VTL to manage N Dimensional data
[heap]
pub struct Tensor<T> {
pub mut:
	data    &storage.Storage<T>
	memory  MemoryFormat
	size    int
	shape   []int
	strides []int
}

// str returns the string representation of a Tensor
[inline]
pub fn (t &Tensor<T>) str() string {
	return tensor_str<T>(t, ', ', '')
}

// rank returns the number of dimensions of a given Tensor
pub fn (t &Tensor<T>) rank() int {
	return t.shape.len
}

// size returns the number of allocated elements for a given tensor
pub fn (t &Tensor<T>) size() int {
	return t.size
}

// is_matrix returns if a Tensor is a nxm matrix or not
[inline]
pub fn (t &Tensor<T>) is_matrix() bool {
	return t.rank() == 2
}

// is_matrix returns if a Tensor is a square matrix or not
[inline]
pub fn (t &Tensor<T>) is_square_matrix() bool {
	return t.rank() == 2 && t.shape[0] == t.shape[1]
}

// is_matrix returns if a Tensor is a square 1D vector or not
[inline]
pub fn (t &Tensor<T>) is_vector() bool {
	return t.rank() == 1
}

// is_row_major returns if a Tensor is supposed to store its data in Row-Major
// order
[inline]
pub fn (t &Tensor<T>) is_row_major() bool {
	// @todo: we need to ensure that t.memory is the source of truth
	return t.memory == .row_major
}

// is_col_major returns if a Tensor is supposed to store its data in Col-Major
// order
[inline]
pub fn (t &Tensor<T>) is_col_major() bool {
	// @todo: we need to ensure that t.memory is the source of truth
	return t.memory == .col_major
}

// is_row_major verifies if a Tensor stores its data in Row-Major
// order
[inline]
pub fn (t &Tensor<T>) is_row_major_contiguous() bool {
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

// is_col_major verifies if a Tensor stores its data in Col-Major
// order
[inline]
pub fn (t &Tensor<T>) is_col_major_contiguous() bool {
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
pub fn (t &Tensor<T>) is_contiguous() bool {
	return t.is_row_major_contiguous() || t.is_col_major_contiguous()
}

// to_array returns the flatten representation of a tensor in a v array storing
// elements of type T
pub fn (t &Tensor<T>) to_array() []T {
	mut arr := []T{cap: t.size}
	mut iter := t.iterator()
	for {
		val, _ := iter.next() or { break }
		arr << T(val)
	}
	return arr
}

// as_type returns a new Tensor with a cast to a given type
// pub fn (t &Tensor<T>) as_type<T, U>() &Tensor<U> {
// 	t_arr := t.to_array<T>()
// 	arr := t_arr.map(U(it))

// 	return from_array<U>(arr, t.shape, t.memory)
// }

// copy returns a copy of a Tensor with a particular memory
// layout, either row_major-contiguous or col_major-contiguous
[inline]
pub fn (t &Tensor<T>) copy(memory MemoryFormat) &Tensor<T> {
	strides := strides_from_shape(t.shape, memory)
	size := size_from_shape(t.shape)
	return &Tensor<T>{
		data: t.data.clone()
		memory: memory
		shape: t.shape
		strides: strides
		size: size
	}
}

// view returns a view of a Tensor, identical to the
// parent but not owning its own data
[inline]
pub fn (t &Tensor<T>) view() &Tensor<T> {
	return &Tensor<T>{
		data: t.data
		memory: t.memory
		shape: t.shape.clone()
		strides: t.strides.clone()
		size: t.size
	}
}
