module vtl

import vtl.storage

pub enum MemoryFormat {
	row_major
	col_major
}

pub struct Tensor {
pub:
	memory  MemoryFormat
pub mut:
	shape   []int
	strides []int
	data    &storage.CpuStorage // @todo: improve using strategy
}

// get returns a scalar value from a Tensor at the provided index
[inline]
pub fn (t Tensor) get(index []int) voidptr {
	mut offset := t.offset(index)
	unsafe {
		return t.data.get(offset)
	}
}

// set copies a scalar value into a Tensor at the provided index
[inline]
pub fn (mut t Tensor) set(index []int, val voidptr) {
	mut offset := t.offset(index)
	unsafe {t.data.set(offset, val)}
}

[inline]
pub fn (t Tensor) rank() int {
	return t.shape.len
}

// offset returns a pointer to a Tensor's data at
// a given index
[inline]
fn (t Tensor) offset(index []int) int {
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
