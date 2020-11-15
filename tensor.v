module vtl

import vtl.storage

pub enum MemoryFormat {
	rowmajor
	colmajor
}

pub struct Tensor {
pub:
	memory  MemoryFormat
pub mut:
	shape   []int
	strides []int
	data    storage.CpuStorage // @todo: improve using strategy
}
