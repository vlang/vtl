module vtl

pub enum MemoryFormat {
	rowmajor
	colmajor
}

pub struct Tensor {
mut:
	data    Storage
pub:
	memory  MemoryFormat
pub mut:
	size    int
	shape   []int
	strides []int
}
