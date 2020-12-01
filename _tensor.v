module vtl

// `MemoryFormat` is a sum type that lists the possible memory layouts
pub enum MemoryFormat {
	rowmajor
	colmajor
}

pub struct Tensor {
	etype   string = default_type
mut:
	data    Storage
	memory  MemoryFormat
pub mut:
	size    int
	shape   []int
	strides []int
}
