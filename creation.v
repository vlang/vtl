module vtl

import vtl.etype
import vtl.storage

pub struct TensorData {
pub:
	shape   []int
	init    etype.Num    = etype.Num(f64(0.0))
	memory  MemoryFormat = .rowmajor
	storage storage.StorageStrategy = .cpu
}

// Return a new Tensor of given shape and type, without initializing entries
pub fn empty(shape []int) Tensor {
	return new_tensor(shape: shape)
}

// Return a new Tensor with the same shape and type as a given Tensor.
pub fn empty_like(t Tensor) Tensor {
	return new_tensor_like(t)
}

// The identity array is a square array with ones on the main diagonal.
pub fn identity(n int) Tensor {
	return eye(n, n, 0)
}

// Return a 2-D array with ones on the diagonal and zeros elsewhere.
pub fn eye(m int, n int, k int) Tensor {
	mut ret := zeros([m, n])
	for i in 0 .. m {
		for j in 0 .. n {
			if i == j - k {
				ret.set([i, j], 1.0)
			}
		}
	}
	return ret
}

// Return a new Tensor of given shape and type, filled with zeros
pub fn zeros(shape []int) Tensor {
	return new_tensor(shape: shape)
}

// Return an Tensor of zeros with the same shape and type as a given Tensor
pub fn zeros_like(t Tensor) Tensor {
	return new_tensor_like(t)
}

// Return a new Tensor of given shape and type, filled with ones
pub fn ones(shape []int) Tensor {
	return full(shape, 1.0)
}

// Return an Tensor of ones with the same shape and type as a given Tensor
pub fn ones_like(t Tensor) Tensor {
	return full_like(t, 1.0)
}

// Return a new Tensor of given shape and type, filled with val
pub fn full(shape []int, val etype.Num) Tensor {
	return new_tensor(shape: shape, init: val)
}

// Return a full Tensor with the same shape and type as a given Tensor
pub fn full_like(t Tensor, val etype.Num) Tensor {
	mut new_tensor := new_tensor_like(t)
	new_tensor.fill(val)
	return new_tensor
}

pub struct BuildRangeData {
	from int
	to   int
}

// range returns a Tensor containing values ranging from [from, to)
pub fn range(data BuildRangeData) Tensor {
	mut res := empty([data.to - data.from])
	for i := data.from; i < data.to; i++ {
		res.set([i], f64(i))
	}
	return res
}

// seq returns a Tensor containing values ranging from [0, to)
[inline]
pub fn seq(n int) Tensor {
	return range(to: n)
}

// from_1d takes a one dimensional array of floating point values
// and returns a one dimensional Tensor if possible
pub fn from_1d<T>(arr []T) Tensor {
	return from_varray<T>(arr, [arr.len])
}

// from_2d takes a two dimensional array of floating point values
// and returns a two-dimensional Tensor if possible
pub fn from_2d<T>(a [][]T) Tensor {
	mut arr := []T{}
	for i in 0 .. a.len {
		for j in 0 .. a[0].len {
			arr << a[i][j]
		}
	}
	return from_varray<T>(arr, [a.len, a[0].len])
}

// from_varray takes a one dimensional array of T values
// and coerces it into an arbitrary shaped Tensor if possible.
// Panics if the shape provided does not hold the provided array
pub fn from_varray<T>(arr []T, shape []int) Tensor {
	return new_tensor_from_varray<T>(arr, shape: shape)
}

// returns a copy of an array with a particular memory
// layout, either rowmajor-contiguous or colmajor-contiguous
[inline]
pub fn (t Tensor) copy(memory MemoryFormat) Tensor {
	return new_tensor_like_with_memory(t, memory)
}

pub fn new_tensor(data TensorData) Tensor {
	etype := data.init.etype()
	if data.shape.len == 0 {
		data_storage := storage.new_storage(strategy: data.storage, etype: etype, len: 1)
		return Tensor{
			memory: data.memory
			strides: [1]
			shape: []
			size: 1
			etype: etype
			data: data_storage
		}
	}
	strides := strides_from_shape(data.shape, data.memory)
	size := size_from_shape(data.shape)
	data_storage := storage.new_storage(
		len: size
		init: data.init
		etype: etype
		strategy: data.storage
	)
	return Tensor{
		shape: data.shape
		memory: data.memory
		strides: strides
		size: size
		etype: etype
		data: data_storage
	}
}

pub fn new_tensor_like(t Tensor) Tensor {
	storage := storage.new_storage_like(t.data)
	return Tensor{
		shape: t.shape
		strides: t.strides
		memory: t.memory
		size: t.size
		etype: t.etype
		data: storage
	}
}

pub fn new_tensor_like_with_memory(t Tensor, memory MemoryFormat) Tensor {
	strides := strides_from_shape(t.shape, memory)
	size := size_from_shape(t.shape)
	storage := storage.new_storage_like_with_len(t.data, size)
	return Tensor{
		shape: t.shape
		strides: strides
		memory: t.memory
		size: size
		etype: t.etype
		data: storage
	}
}

pub fn new_tensor_like_with_shape(t Tensor, shape []int) Tensor {
	strides := strides_from_shape(shape, t.memory)
	size := size_from_shape(shape)
	storage := storage.new_storage_like_with_len(t.data, size)
	return Tensor{
		shape: shape
		strides: strides
		memory: t.memory
		size: size
		etype: t.etype
		data: storage
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
			etype: T.name
			data: data_storage
		}
	}
	strides := strides_from_shape(data.shape, data.memory)
	return Tensor{
		shape: data.shape
		strides: strides
		memory: data.memory
		size: size
		etype: T.name
		data: data_storage
	}
}
