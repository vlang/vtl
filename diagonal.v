module vtl

// DiagonalData configures the offset and axes used by diagonal.
@[params]
pub struct DiagonalData {
pub:
	offset int
	axis1  int
	axis2  int = 1
}

// ReadOnlyDiagonalView exposes diagonal values without tensor mutators.
pub struct ReadOnlyDiagonalView[T] {
	backing &Tensor[T] = unsafe { nil }
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

// diagonal_readonly returns a storage-sharing diagonal view with a read-only API.
// The legacy diagonal function remains writable for compatibility.
pub fn diagonal_readonly[T](input &Tensor[T], params DiagonalData) !&ReadOnlyDiagonalView[T] {
	return &ReadOnlyDiagonalView[T]{
		backing: diagonal[T](input, params)!
	}
}

// shape returns a copy of the view shape.
pub fn (view &ReadOnlyDiagonalView[T]) shape() []int {
	return view.backing.shape.clone()
}

// rank returns the number of dimensions in the view.
pub fn (view &ReadOnlyDiagonalView[T]) rank() int {
	return view.backing.rank()
}

// size returns the number of elements in the view.
pub fn (view &ReadOnlyDiagonalView[T]) size() int {
	return view.backing.size()
}

// get reads the element at the given multidimensional index.
pub fn (view &ReadOnlyDiagonalView[T]) get(index []int) T {
	return view.backing.get(index)
}

// get_nth reads an element in logical row-major order.
pub fn (view &ReadOnlyDiagonalView[T]) get_nth(index int) T {
	return view.backing.get_nth(index)
}

// to_array returns a copy of the view's logical values.
pub fn (view &ReadOnlyDiagonalView[T]) to_array() []T {
	return view.backing.to_array()
}
