module vtl

// indices returns a dense coordinate tensor with shape
// [dimensions.len, ...dimensions], matching numpy.indices for integer arrays.
// Each leading-axis plane contains the coordinates for one input dimension.
pub fn indices(dimensions []int) !&Tensor[int] {
	mut total_size := 1
	for dimension in dimensions {
		if dimension < 0 {
			return error('indices dimensions must be non-negative')
		}
		if dimension != 0 && total_size > max_int / dimension {
			return error('indices shape is too large')
		}
		total_size *= dimension
	}
	if dimensions.len > 0 && total_size > max_int / dimensions.len {
		return error('indices output shape is too large')
	}
	mut output_shape := [dimensions.len]
	output_shape << dimensions
	mut result := empty[int](output_shape, memory: .row_major)
	if total_size == 0 {
		return result
	}
	for axis, dimension in dimensions {
		mut axis_stride := 1
		for trailing_dimension in dimensions[axis + 1..] {
			axis_stride *= trailing_dimension
		}
		for flat_index in 0 .. total_size {
			coordinate := (flat_index / axis_stride) % dimension
			result.set_nth(axis * total_size + flat_index, coordinate)
		}
	}
	return result
}

// indices_sparse returns one broadcastable coordinate tensor per dimension.
// Each result has rank dimensions.len and only its own axis has non-unit
// length, matching numpy.indices(..., sparse: true) without allocating the
// dense coordinate stack.
pub fn indices_sparse(dimensions []int) ![]&Tensor[int] {
	for dimension in dimensions {
		if dimension < 0 {
			return error('indices dimensions must be non-negative')
		}
	}
	mut coordinates := []&Tensor[int]{cap: dimensions.len}
	for axis, dimension in dimensions {
		mut shape := []int{len: dimensions.len, init: 1}
		shape[axis] = dimension
		mut coordinate := empty[int](shape, memory: .row_major)
		for value in 0 .. dimension {
			coordinate.set_nth(value, value)
		}
		coordinates << coordinate
	}
	return coordinates
}
