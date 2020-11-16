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

pub fn tensor_to_varray<T>(t Tensor) []T {
	mut arr := []T{}
	mut iter := t.init_strided_iteration()
	for _ in 0 .. t.size {
		arr.push(storage_get(t.data, iter.pos))
		iter.next()
	}
	return arr
}
