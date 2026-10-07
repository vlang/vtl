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
