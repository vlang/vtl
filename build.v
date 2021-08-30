module vtl

import storage

pub struct TensorBuildData {
	memory  MemoryFormat = .rowmajor
	storage storage.StorageStrategy = .cpu
}

[inline]
fn (d TensorBuildData) with_shape(shape []int) TensorBuildDataWithShape {
	return TensorBuildDataWithShape{
		shape: shape
		memory: d.memory
		storage: d.storage
	}
}

pub struct TensorBuildDataWithShape {
	shape   []int
	memory  MemoryFormat = .rowmajor
	storage storage.StorageStrategy = .cpu
}

[inline]
fn (d TensorBuildDataWithShape) without_shape() TensorBuildData {
	return TensorBuildData{
		memory: d.memory
		storage: d.storage
	}
}

// from_varray takes a one dimensional array of T values
// and coerces it into an arbitrary shaped Tensor if possible.
// Panics if the shape provided does not hold the provided array
pub fn from_array<T>(arr []T, shape []int, data TensorBuildData) &Tensor<T> {
	size := size_from_shape(shape)
	if size != arr.len {
		panic('Bad shape for array, shape [$arr.len] cannot fit into shape $shape')
	}
	data_storage := storage.from_array<T>(arr, data.storage)
	if shape.len == 0 {
		return &Tensor<T>{
			memory: data.memory
			strides: [1]
			shape: []
			size: size
			data: data_storage
		}
	}
	strides := strides_from_shape(shape, data.memory)
	return &Tensor<T>{
		shape: shape
		strides: strides
		memory: data.memory
		size: size
		data: data_storage
	}
}

// as_type returns a new Tensor with a cast to a given type
// pub fn (t &Tensor<T>) as_type<T, U>() &Tensor<U> {
// 	t_arr := t.to_array<T>()
// 	arr := t_arr.map(U(it))

// 	return from_array<U>(arr, t.shape, memory: t.memory)
// }

// new_tensor allocates a Tensor onto specified device with a given data
fn new_tensor<T>(init T, data TensorBuildDataWithShape) &Tensor<T> {
	if data.shape.len == 0 {
		data_storage := storage.new_storage<T>(1, 0, init, data.storage)
		return &Tensor<T>{
			memory: data.memory
			strides: [1]
			shape: []
			size: 1
			data: data_storage
		}
	}
	strides := strides_from_shape(data.shape, data.memory)
	size := size_from_shape(data.shape)
	data_storage := storage.new_storage<T>(size, 0, init, data.storage)
	return &Tensor<T>{
		shape: data.shape.clone()
		memory: data.memory
		strides: strides
		size: size
		data: data_storage
	}
}

// new_tensor_like returns a new tensor created with similar storage properties
// as the Tensor t
fn new_tensor_like<T>(t &Tensor<T>) &Tensor<T> {
	storage := storage.storage_like<T>(t.data)
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
	storage := storage.storage_like_with_len<T>(t.data, size)
	return &Tensor<T>{
		shape: shape.clone()
		strides: strides
		memory: t.memory
		size: size
		data: storage
	}
}
