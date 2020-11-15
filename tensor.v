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
	data    storage.CpuStorage // @todo: improve using strategy
}

pub struct TensorData {
pub:
	shape   []int
	init    voidptr = voidptr(0)
	memory  MemoryFormat = .row_major
	storage StorageStrategy = .cpu
}
