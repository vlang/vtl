module vtl

import vtl.storage

pub struct TensorBuildData {
pub:
	shape   []int
	memory  MemoryFormat = .row_major
	init    voidptr
	storage storage.StorageStrategy
}

pub fn new_tensor<T>(data TensorBuildData) Tensor {
	if data.shape.len == 0 {
		return Tensor{
			memory: data.memory
			strides: [1]
			shape: []
			data: &storage.new_storage<T>({
				cap: 0
				strategy: data.storage
			})
		}
	}
	strides := strides_from_shape(data.shape, data.memory)
	size := size_from_shape(data.shape)
	return Tensor{
		memory: data.memory
		strides: strides
		data: &storage.new_storage<T>({
			cap: size
			strategy: data.storage
			init: data.init
		})
	}
}

fn strides_from_shape(shape []int, memory MemoryFormat) []int {
	mut accum := 1
	mut result := []int{len: shape.len}
	if memory == .row_major {
		for i := shape.len - 1; i >= 0; i-- {
			result[i] = accum
			accum *= shape[i]
		}
		return result
	}
	for i in 0 .. shape.len {
		result[i] = accum
		accum *= shape[i]
	}
	return result
}

pub fn size_from_shape(shape []int) int {
	mut accum := 1
	for i in shape {
		accum *= i
	}
	return accum
}
