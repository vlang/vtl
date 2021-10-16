module vtl

import storage

[params]
pub struct TensorData {
pub:
	memory MemoryFormat = .row_major
}

// from_varray takes a one dimensional array of T values
// and coerces it into an arbitrary shaped Tensor if possible.
// Panics if the shape provided does not hold the provided array
pub fn from_array<T>(arr []T, shape []int, params TensorData) &Tensor<T> {
	size := size_from_shape(shape)
	if size != arr.len {
		panic('Bad shape for array, shape [$arr.len] cannot fit into shape $shape')
	}
	data_storage := storage.from_array<T>(arr)
	if shape.len == 0 {
		return &Tensor<T>{
			memory: params.memory
			strides: [1]
			shape: []
			size: size
			data: data_storage
		}
	}
	strides := strides_from_shape(shape, params.memory)
	return &Tensor<T>{
		shape: shape
		strides: strides
		memory: params.memory
		size: size
		data: data_storage
	}
}

// new_tensor allocates a Tensor onto specified device with a given data
fn new_tensor<T>(init T, shape []int, params TensorData) &Tensor<T> {
	if shape.len == 0 {
		data_storage := storage.new_storage<T>(1, 0, init)
		return &Tensor<T>{
			memory: params.memory
			strides: [1]
			shape: []
			size: 1
			data: data_storage
		}
	}
	strides := strides_from_shape(shape, params.memory)
	size := size_from_shape(shape)
	data_storage := storage.new_storage<T>(size, 0, init)
	return &Tensor<T>{
		shape: shape.clone()
		memory: params.memory
		strides: strides
		size: size
		data: data_storage
	}
}

// new_tensor_like returns a new tensor created with similar storage properties
// as the Tensor t
fn new_tensor_like<T>(t &Tensor<T>) &Tensor<T> {
	storage := t.data.like<T>()
	return &Tensor<T>{
		shape: t.shape.clone()
		strides: t.strides.clone()
		memory: t.memory
		size: t.size
		data: storage
	}
}

// new_tensor_like_with_shape returns a new tensor created with similar storage properties
// as the Tensor `t` with a given shape
fn new_tensor_like_with_shape<T>(t &Tensor<T>, shape []int) &Tensor<T> {
	strides := strides_from_shape(shape, t.memory)
	size := size_from_shape(shape)
	storage := t.data.like_with_len<T>(size)
	return &Tensor<T>{
		shape: shape.clone()
		strides: strides
		memory: t.memory
		size: size
		data: storage
	}
}
