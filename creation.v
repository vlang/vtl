module vtl

// meshgrid builds dense two-dimensional XY coordinate grids from two vectors.
// The output shape is [len(y), len(x)], matching NumPy's default indexing="xy".
pub fn meshgrid[T](x &Tensor[T], y &Tensor[T]) !(&Tensor[T], &Tensor[T]) {
	if x.rank() != 1 || y.rank() != 1 {
		return error('meshgrid expects two one-dimensional tensors')
	}
	shape := [y.size(), x.size()]
	mut x_grid := empty[T](shape, memory: .row_major)
	mut y_grid := empty[T](shape, memory: .row_major)
	for row in 0 .. y.size() {
		for col in 0 .. x.size() {
			x_grid.set([row, col], x.get_nth(col))
			y_grid.set([row, col], y.get_nth(row))
		}
	}
	return x_grid, y_grid
}

// empty returns a new Tensor of given shape and type, without initializing entries

// empty exposes this operation as part of the public API.

// empty exposes this operation as part of the public API.
@[inline]
pub fn empty[T](shape []int, params TensorData) &Tensor[T] {
	return tensor[T](cast[T](0), shape, params)
}

// empty_like returns a new Tensor of given shape and type as a given Tensor

// empty_like exposes this operation as part of the public API.

// empty_like exposes this operation as part of the public API.
@[inline]
pub fn empty_like[T](t &Tensor[T]) &Tensor[T] {
	return tensor_like[T](t)
}

// identity returns an array is a square array with ones on the main diagonal

// identity exposes this operation as part of the public API.

// identity exposes this operation as part of the public API.
@[inline]
pub fn identity[T](n int, params TensorData) &Tensor[T] {
	return eye[T](n, n, 0, params)
}

// eye returns a 2D array with ones on the diagonal and zeros elsewhere
pub fn eye[T](m int, n int, k int, params TensorData) &Tensor[T] {
	mut ret := zeros[T]([m, n], params)
	for i in 0 .. m {
		for j in 0 .. n {
			if i == j - k {
				ret.set([i, j], cast[T](1))
			}
		}
	}
	return ret
}

// zeros returns a new tensor of a given shape and type, filled with zeros

// zeros exposes this operation as part of the public API.

// zeros exposes this operation as part of the public API.
@[inline]
pub fn zeros[T](shape []int, params TensorData) &Tensor[T] {
	return tensor[T](cast[T](0), shape, params)
}

// zeros_like returns a new Tensor of given shape and type as a given Tensor, filled with zeros

// zeros_like exposes this operation as part of the public API.

// zeros_like exposes this operation as part of the public API.
@[inline]
pub fn zeros_like[T](t &Tensor[T]) &Tensor[T] {
	return tensor_like[T](t)
}

// ones returns a new tensor of a given shape and type, filled with ones

// ones exposes this operation as part of the public API.

// ones exposes this operation as part of the public API.
@[inline]
pub fn ones[T](shape []int, params TensorData) &Tensor[T] {
	return full[T](shape, cast[T](1), params)
}

// ones_like returns a new tensor of a given shape and type, filled with ones

// ones_like exposes this operation as part of the public API.

// ones_like exposes this operation as part of the public API.
@[inline]
pub fn ones_like[T](t &Tensor[T]) &Tensor[T] {
	return full_like[T](t, cast[T](1))
}

// full returns a new tensor of a given shape and type, filled with the given value

// full exposes this operation as part of the public API.

// full exposes this operation as part of the public API.
@[inline]
pub fn full[T](shape []int, val T, params TensorData) &Tensor[T] {
	return tensor[T](val, shape, params)
}

// full_like returns a new tensor of the same shape and type as a given Tensor filled with a given val
pub fn full_like[T](t &Tensor[T], val T) &Tensor[T] {
	mut tensor := tensor_like[T](t)
	tensor.fill(val)
	return tensor
}

// range returns a Tensor containing values ranging from [from, to)
pub fn range[T](from int, to int, params TensorData) &Tensor[T] {
	mut res := empty[T]([to - from], params)
	mut index := 0
	for val in from .. to {
		res.set([index], cast[T](val))
		index++
	}
	return res
}

// seq returns a Tensor containing values ranging from [0, to)

// seq exposes this operation as part of the public API.

// seq exposes this operation as part of the public API.
@[inline]
pub fn seq[T](n int, params TensorData) &Tensor[T] {
	return range[T](0, n, params)
}

// from_1d takes a one dimensional array of floating point values
// and returns a one dimensional Tensor if possible
pub fn from_1d[T](arr []T, params TensorData) !&Tensor[T] {
	return from_array[T](arr, [arr.len], params)
}

// from_2d takes a two dimensional array of floating point values
// and returns a two-dimensional Tensor if possible

// from_2d exposes this operation as part of the public API.

// from_2d exposes this operation as part of the public API.
@[direct_array_access]
pub fn from_2d[T](a [][]T, params TensorData) !&Tensor[T] {
	mut arr := []T{cap: a.len * a[0].len}
	for i in 0 .. a.len {
		for j in 0 .. a[0].len {
			arr << a[i][j]
		}
	}
	shape := [a.len, a[0].len]
	return from_array[T](arr, shape, params)
}
