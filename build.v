module vtl

pub struct TensorData {
pub:
	shape   []int
	init    voidptr = voidptr(0)
	memory  MemoryFormat = .row_major
	storage StorageStrategy = .cpu
}

pub fn new_tensor<T>(data TensorData) Tensor {
	if data.shape.len == 0 {
		data_storage := new_storage<T>({
			strategy: data.storage
		})
		return Tensor{
			memory: data.memory
			strides: [1]
			shape: []
			data: data_storage
		}
	}
	strides := strides_from_shape(data.shape, data.memory)
	size := size_from_shape(data.shape)
	data_storage := new_storage<T>({
		len: size
		init: data.init
		strategy: data.storage
	})
	return Tensor{
		memory: data.memory
		strides: strides
		data: data_storage
	}
}

pub fn tensor_to_varray<T>(t Tensor) []T {
        return storage_to_varray<T>(t.data)
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

fn size_from_shape(shape []int) int {
	mut accum := 1
	for i in shape {
		accum *= i
	}
	return accum
}
