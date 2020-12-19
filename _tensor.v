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
pub mut:
	memory  MemoryFormat
	size    int
	shape   []int
	strides []int
}

// str returns the string representation of a Tensor
pub fn (t Tensor) str() string {
	return tensor_str(t, ', ', '')
}

// tensor_to_varray<T> returns the flatten representation of a tensor in a v array storing
// elements of type T
pub fn tensor_to_varray<T>(t Tensor) []T {
	mut arr := []T{}
	mut iter := t.iterator()
	for _ in 0 .. t.size {
		arr << num_as_type<T>(iter.next())
	}
	return arr
}

// tensor_as_type<T> returns a new Tensor with a cast to a given type
pub fn tensor_as_type<T>(t Tensor) Tensor {
	arr := tensor_to_varray<T>(t)
	return new_tensor_from_varray<T>(arr, shape: t.shape, memory: t.memory)
}
