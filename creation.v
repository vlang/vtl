module vtl

import math

// MeshgridIndexing controls the dimension order used by meshgrid_n.
pub enum MeshgridIndexing {
	xy
	ij
}

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

// meshgrid_n builds one dense coordinate tensor for each one-dimensional
// input. With `xy`, the first two dimensions are swapped, matching NumPy's
// default indexing convention. `ij` preserves the input axis order.
pub fn meshgrid_n[T](vectors []&Tensor[T], indexing MeshgridIndexing) ![]&Tensor[T] {
	if vectors.len == 0 {
		return error('meshgrid_n expects at least one vector')
	}
	mut shape := []int{cap: vectors.len}
	for vector in vectors {
		if vector.rank() != 1 {
			return error('meshgrid_n expects one-dimensional tensors')
		}
		shape << vector.size()
	}
	if indexing == .xy && shape.len > 1 {
		first_dimension := shape[0]
		shape[0] = shape[1]
		shape[1] = first_dimension
	}
	mut grids := []&Tensor[T]{cap: vectors.len}
	for _ in vectors {
		grids << empty[T](shape, memory: .row_major)
	}
	mut total_size := 1
	for dimension in shape {
		total_size *= dimension
	}
	for flat_index in 0 .. total_size {
		mut output_index := []int{len: shape.len}
		mut remainder := flat_index
		for axis := shape.len - 1; axis >= 0; axis-- {
			output_index[axis] = remainder % shape[axis]
			remainder /= shape[axis]
		}
		for axis, vector in vectors {
			input_axis := if indexing == .xy && vectors.len > 1 {
				if axis == 0 {
					1
				} else if axis == 1 {
					0
				} else {
					axis
				}
			} else {
				axis
			}
			grids[axis].set(output_index, vector.get_nth(output_index[input_axis]))
		}
	}
	return grids
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

// arange returns evenly spaced values in the half-open interval [start, stop).
// The direction of step determines whether the result is ascending or descending.
pub fn arange[T](start f64, stop f64, step f64, params TensorData) !&Tensor[T] {
	if !math.is_finite(start) || !math.is_finite(stop) || !math.is_finite(step) || step == 0.0 {
		return error('arange start, stop, and non-zero step must be finite')
	}
	mut count := 0
	if step > 0.0 && start < stop {
		count_f64 := math.ceil((stop - start) / step)
		if count_f64 >= f64(max_int) {
			return error('arange result is too large')
		}
		count = int(count_f64)
	} else if step < 0.0 && start > stop {
		count_f64 := math.ceil((stop - start) / step)
		if count_f64 >= f64(max_int) {
			return error('arange result is too large')
		}
		count = int(count_f64)
	}
	mut result := empty[T]([count], params)
	for i in 0 .. count {
		result.set([i], cast[T](start + f64(i) * step))
	}
	return result
}

// LinspaceData configures linspace's endpoint and output memory layout.
@[params]
pub struct LinspaceData {
pub:
	endpoint bool         = true
	memory   MemoryFormat = .row_major
}

// linspace returns `num` evenly spaced values between start and stop.
// The endpoint is included by default; set endpoint: false to exclude stop.
pub fn linspace[T](start f64, stop f64, num int, params LinspaceData) !&Tensor[T] {
	if num < 0 {
		return error('linspace num must be non-negative')
	}
	mut result := empty[T]([num], memory: params.memory)
	if num == 0 {
		return result
	}
	result.set([0], cast[T](start))
	if num == 1 {
		return result
	}
	denominator := if params.endpoint { num - 1 } else { num }
	step := (stop - start) / f64(denominator)
	for i in 1 .. num {
		result.set([i], cast[T](start + f64(i) * step))
	}
	if params.endpoint {
		result.set([num - 1], cast[T](stop))
	}
	return result
}

// LogspaceData configures logspace's endpoint, base, and output memory layout.
@[params]
pub struct LogspaceData {
pub:
	endpoint bool         = true
	base     f64          = 10.0
	memory   MemoryFormat = .row_major
}

// logspace returns `num` values spaced evenly on a logarithmic scale between
// base^start and base^stop. The endpoint is included by default.
pub fn logspace[T](start f64, stop f64, num int, params LogspaceData) !&Tensor[T] {
	if !math.is_finite(start) || !math.is_finite(stop) || !math.is_finite(params.base)
		|| params.base <= 0.0 {
		return error('logspace exponents and positive base must be finite')
	}
	if num < 0 {
		return error('logspace num must be non-negative')
	}
	mut result := empty[T]([num], memory: params.memory)
	if num == 0 {
		return result
	}
	result.set([0], cast[T](math.pow(params.base, start)))
	if num == 1 {
		return result
	}
	denominator := if params.endpoint { num - 1 } else { num }
	step := (stop - start) / f64(denominator)
	for i in 1 .. num {
		result.set([i], cast[T](math.pow(params.base, start + f64(i) * step)))
	}
	if params.endpoint {
		result.set([num - 1], cast[T](math.pow(params.base, stop)))
	}
	return result
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
	if a.len == 0 {
		return error('from_2d requires at least one row')
	}
	columns := a[0].len
	mut arr := []T{cap: a.len * columns}
	for i in 0 .. a.len {
		if a[i].len != columns {
			return error('from_2d row ${i} has ${a[i].len} columns; expected ${columns}')
		}
		for j in 0 .. columns {
			arr << a[i][j]
		}
	}
	shape := [a.len, columns]
	row_major := from_array[T](arr, shape)!
	if params.memory == .col_major {
		return row_major.copy(.col_major)
	}
	return row_major
}
