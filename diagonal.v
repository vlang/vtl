module vtl

// DiagonalData configures the offset and axes used by diagonal.
@[params]
pub struct DiagonalData {
pub:
	offset int
	axis1  int
	axis2  int = 1
}

// diagonal returns a view along the diagonal of two axes of an N-D tensor.
// The selected axes are removed and the diagonal axis is appended to the
// result, following NumPy's shape convention. The view shares input storage.
// VTL tensors remain mutable, so writes through the result update the input.
pub fn diagonal[T](input &Tensor[T], params DiagonalData) !&Tensor[T] {
	rank := input.rank()
	if rank < 2 {
		return error('diagonal requires a tensor with at least two dimensions')
	}
	axis1 := if params.axis1 < 0 { params.axis1 + rank } else { params.axis1 }
	axis2 := if params.axis2 < 0 { params.axis2 + rank } else { params.axis2 }
	if axis1 < 0 || axis1 >= rank || axis2 < 0 || axis2 >= rank {
		return error('diagonal axes ${params.axis1} and ${params.axis2} are out of range for rank ${rank}')
	}
	if axis1 == axis2 {
		return error('diagonal axes must be different')
	}
	offset := params.offset
	mut diagonal_length := 0
	mut offset_elements := 0
	if offset < input.shape[axis2] && offset > -input.shape[axis1] {
		row_start := if offset < 0 { -offset } else { 0 }
		column_start := if offset > 0 { offset } else { 0 }
		rows_remaining := input.shape[axis1] - row_start
		columns_remaining := input.shape[axis2] - column_start
		diagonal_length = if rows_remaining < columns_remaining {
			rows_remaining
		} else {
			columns_remaining
		}
		offset_elements = row_start * input.strides[axis1] + column_start * input.strides[axis2]
	}
	mut shape := []int{cap: rank - 1}
	mut strides := []int{cap: rank - 1}
	for axis in 0 .. rank {
		if axis != axis1 && axis != axis2 {
			shape << input.shape[axis]
			strides << input.strides[axis]
		}
	}
	shape << diagonal_length
	strides << input.strides[axis1] + input.strides[axis2]
	mut result := &Tensor[T]{
		shape:   shape
		strides: strides
		size:    size_from_shape(shape)
		data:    input.data.offset[T](offset_elements)
		memory:  input.memory
	}
	result.ensure_memory()
	return result
}
