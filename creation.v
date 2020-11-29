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
	return new_tensor<T>({
		shape: shape
	})
}

// Return a new Tensor with the same shape and type as a given Tensor.
pub fn empty_like(t Tensor) Tensor {
	return new_tensor_like(t)
}

// The identity array is a square array with ones on the main diagonal.
pub fn identity<T>(n int) Tensor {
	return eye<T>(n, n, 0)
}

// Return a 2-D array with ones on the diagonal and zeros elsewhere.
pub fn eye<T>(m int, n int, k int) Tensor {
	mut ret := zeros<T>([m, n])
	for i in 0 .. m {
		for j in 0 .. n {
			if i == j - k {
				val := T(1)
				ret.set([i, j], &val)
			}
		}
	}
	return ret
}

// Return a new Tensor of given shape and type, filled with zeros
pub fn zeros<T>(shape []int) Tensor {
	return new_tensor<T>({
		shape: shape
	})
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
	return full_like<f64>(t, 1.0)
}

// Return a new Tensor of given shape and type, filled with val
pub fn full<T>(shape []int, val T) Tensor {
	return new_tensor<T>({
		shape: shape
		init: &val
	})
}

// Return a full Tensor with the same shape and type as a given Tensor
pub fn full_like<T>(t Tensor, val T) Tensor {
	mut new_tensor := new_tensor_like(t)
	new_tensor.fill(&val)
	return new_tensor
}

pub struct BuildRangeData {
	from int
	to   int
}

// range returns a Tensor containing values ranging from [start, stop)
pub fn range<T>(data BuildRangeData) Tensor {
	mut res := empty<T>([data.stop - data.start])
	for i := data.start; i < data.stop; i++ {
		v := T(i)
		res.set([i], &v)
	}
	return res
}

// from_1d takes a one dimensional array of floating point values
// and returns a one dimensional Tensor if possible
pub fn from_1d<T>(arr []T) Tensor {
	return from_varray<T>(arr, [arr.len])
}

// from_2d takes a two dimensional array of floating point values
// and returns a two-dimensional Tensor if possible
pub fn from_2d<T>(a [][]T) Tensor {
	mut ret := new_tensor<T>({
		shape: [a.len, a[0].len]
	})
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
	return new_tensor_from_varray<T>(arr, {
		shape: shape
	})
}

// returns a copy of an array with a particular memory
// layout, either rowmajor-contiguous or colmajor-contiguous
[inline]
pub fn (t Tensor) copy(memory MemoryFormat) Tensor {
	return new_tensor_like_with_memory(t, memory)
}

pub fn new_tensor<T>(data TensorData) Tensor {
	if data.shape.len == 0 {
		data_storage := new_storage<T>({
			strategy: data.storage
                        len: 1
		})
		return Tensor{
			memory: data.memory
			strides: [1]
			shape: []
			size: 1
			data: &data_storage
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
		shape: data.shape
		memory: data.memory
		strides: strides
		size: size
		data: &data_storage
	}
}

pub fn new_tensor_like(t Tensor) Tensor {
	storage := new_storage_like(t.data)
	return Tensor{
		shape: t.shape
		strides: t.strides
		memory: t.memory
		size: t.size
		data: &storage
	}
}

pub fn new_tensor_like_with_memory(t Tensor, memory MemoryFormat) Tensor {
	strides := strides_from_shape(t.shape, memory)
	size := size_from_shape(t.shape)
	storage := new_storage_like_with_len(t.data, size)
	return Tensor{
		shape: t.shape
		strides: strides
		memory: t.memory
		size: size
		data: &storage
	}
}

pub fn new_tensor_like_with_shape(t Tensor, shape []int) Tensor {
	strides := strides_from_shape(shape, t.memory)
	size := size_from_shape(shape)
	storage := new_storage_like_with_len(t.data, size)
	return Tensor{
		shape: shape
		strides: strides
		memory: t.memory
		size: size
		data: &storage
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
			size: size
			data: &data_storage
		}
	}
	strides := strides_from_shape(data.shape, data.memory)
	return Tensor{
		shape: data.shape
		strides: strides
		memory: data.memory
		size: size
		data: &data_storage
	}
}
