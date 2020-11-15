module vtl

import vtl.storage

pub enum MemoryFormat {
	rowmajor
	colmajor
}

pub struct Tensor {
mut:
	data    storage.CpuStorage // @todo: improve using strategy
pub:
	memory  MemoryFormat
pub mut:
	shape   []int
	strides []int
}

pub fn tensor_to_varray<T>(t Tensor) []T {
        return storage_to_varray<T>(t.data)
}
