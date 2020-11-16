module vtl

pub struct TensorData {
pub:
	shape   []int
	init    voidptr = voidptr(0)
	memory  MemoryFormat = .rowmajor
	storage StorageStrategy = .cpu
}

// Return a new Tensor of given shape and type, without initializing entries
pub fn empty<T>(shape []int) Tensor {
	return new_tensor<T>({ shape: shape })
}

// Return a new Tensor with the same shape and type as a given Tensor.
pub fn empty_like(t Tensor) Tensor {
	return new_tensor_like(t)
}

// Return a new Tensor of given shape and type, filled with zeros
pub fn zeros<T>(shape []int) Tensor {
	return new_tensor<T>({ shape: shape })
}

// Return an Tensor of zeros with the same shape and type as a given Tensor
pub fn zeros_like(t Tensor) Tensor {
	return new_tensor_like(t)
}

// Return a new Tensor of given shape and type, filled with ones
pub fn ones<T>(shape []int) Tensor {
	return full<T>(shape, T(1))
}

// Return an Tensor of ones with the same shape and type as a given Tensor
pub fn ones_like(t Tensor) Tensor {
	return full_like<int>(t, 1)
}

// Return a new Tensor of given shape and type, filled with val
pub fn full<T>(shape []int, val T) Tensor {
	return new_tensor<T>({ shape: shape, init: &val })
}

// Return a full Tensor with the same shape and type as a given Tensor
pub fn full_like<T>(t Tensor, val T) Tensor {
	mut new_tensor := new_tensor_like(t)
        new_tensor.fill(&val)
        return new_tensor
}

// from_1d takes a one dimensional array of floating point values
// and returns a one dimensional Tensor if possible
pub fn from_1d<T>(arr []T) Tensor {
	return from_varray<T>(arr, [arr.len])
}

// from_2d takes a two dimensional array of floating point values
// and returns a two-dimensional Tensor if possible
pub fn from_2d<T>(a [][]T) Tensor {
	mut ret := new_tensor<T>({ shape: [a.len, a[0].len] })
	for i in 0 .. a.len {
		for j in 0 .. a[0].len {
                        val := a[i][j]
                        ret.set([i, j], &val)
		}
	}
	return ret
}

// from_varray takes a one dimensional array of T values
// and coerces it into an arbitrary shaped Tensor if possible.
// Panics if the shape provided does not hold the provided array
pub fn from_varray<T>(arr []T, shape []int) Tensor {
	return new_tensor_from_varray<T>(arr, { shape: shape })
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

pub fn new_tensor_like(t Tensor) Tensor {
        return Tensor{
                shape: t.shape
                strides: t.strides
                memory: t.memory
                data: new_storage_like(t.data)
        }
} 

pub fn new_tensor_from_varray<T>(arr []T, data TensorData) Tensor {
        size := size_from_shape(data.shape)
        if size != arr.len {
		panic('Bad shape for array, shape [$arr.len] cannot fit into shape $data.shape')
	}
	data_storage := new_storage_from_varray<T>(arr, data.storage)
        if data.shape.len == 0 {
		return Tensor{
			memory: data.memory
			strides: [1]
			shape: []
			data: data_storage
		}
	}
	strides := strides_from_shape(data.shape, data.memory)
        return Tensor{
                shape: data.shape
                strides: strides
                memory: data.memory
                data: data_storage
        }
}
