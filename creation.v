module vtl

// empty returns a new Tensor of given shape and type, without initializing entries
[inline]
pub fn empty<T>(shape []int, memory MemoryFormat) &Tensor<T> {
	return new_tensor<T>(T(0), shape, memory)
}

// empty_like returns a new Tensor of given shape and type as a given Tensor
[inline]
pub fn empty_like<T>(t &Tensor<T>) &Tensor<T> {
	return new_tensor_like<T>(t)
}

// identity returns an array is a square array with ones on the main diagonal
[inline]
pub fn identity<T>(n int, memory MemoryFormat) &Tensor<T> {
	return eye<T>(n, n, 0, memory)
}

// eye returns a 2D array with ones on the diagonal and zeros elsewhere
pub fn eye<T>(m int, n int, k int, memory MemoryFormat) &Tensor<T> {
	mut ret := zeros<T>([m, n], memory)
	for i in 0 .. m {
		for j in 0 .. n {
			if i == j - k {
				ret.set([i, j], T(1))
			}
		}
	}
	return ret
}

// zeros returns a new tensor of a given shape and type, filled with zeros
[inline]
pub fn zeros<T>(shape []int, memory MemoryFormat) &Tensor<T> {
	return new_tensor<T>(T(0), shape, memory)
}

// zeros_like returns a new Tensor of given shape and type as a given Tensor, filled with zeros
[inline]
pub fn zeros_like<T>(t &Tensor<T>) &Tensor<T> {
	return new_tensor_like<T>(t)
}

// ones returns a new tensor of a given shape and type, filled with ones
[inline]
pub fn ones<T>(shape []int, memory MemoryFormat) &Tensor<T> {
	return full<T>(shape, T(1), memory)
}

// ones_like returns a new tensor of a given shape and type, filled with ones
[inline]
pub fn ones_like<T>(t &Tensor<T>) &Tensor<T> {
	return full_like<T>(t, T(1))
}

// full returns a new tensor of a given shape and type, filled with the given value
[inline]
pub fn full<T>(shape []int, val T, memory MemoryFormat) &Tensor<T> {
	return new_tensor<T>(val, shape, memory)
}

// full_like returns a new tensor of the same shape and type as a given Tensor filled with a given val
pub fn full_like<T>(t &Tensor<T>, val T) &Tensor<T> {
	mut new_tensor := new_tensor_like<T>(t)
	new_tensor.fill(val)
	return new_tensor
}

// range returns a Tensor containing values ranging from [from, to)
pub fn range<T>(from int, to int, memory MemoryFormat) &Tensor<T> {
	mut res := empty<T>([to - from], memory)
	for i := from; i < to; i++ {
		res.set([i], T(i))
	}
	return res
}

// seq returns a Tensor containing values ranging from [0, to)
[inline]
pub fn seq<T>(n int, memory MemoryFormat) &Tensor<T> {
	return range<T>(0, n, memory)
}

// from_1d takes a one dimensional array of floating point values
// and returns a one dimensional Tensor if possible
pub fn from_1d<T>(arr []T, memory MemoryFormat) &Tensor<T> {
	return from_array<T>(arr, [arr.len], memory)
}

// from_2d takes a two dimensional array of floating point values
// and returns a two-dimensional Tensor if possible
pub fn from_2d<T>(a [][]T, memory MemoryFormat) &Tensor<T> {
	mut arr := []T{cap: a.len * a[0].len}
	for i in 0 .. a.len {
		for j in 0 .. a[0].len {
			arr << a[i][j]
		}
	}
	shape := [a.len, a[0].len]
	return from_array<T>(arr, shape, memory)
}
