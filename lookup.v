module vtl

// argwhere returns the coordinates of non-zero elements as a row-major tensor
// with shape [number of matches, input rank].
pub fn argwhere[T](t &Tensor[T]) !&Tensor[int] {
	mut coordinates := []int{}
	mut matches := 0
	for flat_index in 0 .. t.size {
		if td[T](t.get_nth(flat_index)).bool() {
			for coordinate in t.nth_index(flat_index) {
				coordinates << coordinate
			}
			matches++
		}
	}
	return from_array[int](coordinates, [matches, t.rank()])
}

// take returns a copy of the tensor with the values at `indices` selected
// along `axis`. Negative axes and negative indices count from the end.
// The indices are a one-dimensional list; use slice for range-based views.
pub fn (t &Tensor[T]) take[T](indices []int, axis int) !&Tensor[T] {
	rank := t.rank()
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('take axis ${axis} is out of range for rank ${rank}')
	}
	mut normalized_indices := []int{cap: indices.len}
	for index in indices {
		normalized := if index < 0 { index + t.shape[axis_index] } else { index }
		if normalized < 0 || normalized >= t.shape[axis_index] {
			return error('take index ${index} is out of range for axis size ${t.shape[axis_index]}')
		}
		normalized_indices << normalized
	}

	mut output_shape := t.shape.clone()
	output_shape[axis_index] = indices.len
	mut result := empty[T](output_shape, memory: t.memory)
	for flat_index in 0 .. result.size {
		mut output_index := result.nth_index(flat_index)
		mut input_index := output_index.clone()
		input_index[axis_index] = normalized_indices[output_index[axis_index]]
		result.set(output_index, t.get(input_index))
	}
	return result
}

// take_along_axis gathers values using an index tensor with the same rank as
// the receiver. Dimensions must match except along `axis`.
pub fn (t &Tensor[T]) take_along_axis[T](indices &Tensor[int], axis int) !&Tensor[T] {
	rank := t.rank()
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('take_along_axis axis ${axis} is out of range for rank ${rank}')
	}
	if indices.rank() != rank {
		return error('take_along_axis index tensor must have rank ${rank}')
	}
	for dimension in 0 .. rank {
		if dimension != axis_index && indices.shape[dimension] != t.shape[dimension] {
			return error('take_along_axis index shape must match tensor shape outside the selected axis')
		}
	}

	mut result := empty[T](indices.shape, memory: t.memory)
	for flat_index in 0 .. result.size {
		result_index := result.nth_index(flat_index)
		mut input_index := result_index.clone()
		selected := indices.get(result_index)
		selected_index := if selected < 0 { selected + t.shape[axis_index] } else { selected }
		if selected_index < 0 || selected_index >= t.shape[axis_index] {
			return error('take_along_axis index ${selected} is out of range for axis size ${t.shape[axis_index]}')
		}
		input_index[axis_index] = selected_index
		result.set(result_index, t.get(input_index))
	}
	return result
}

// get returns a scalar value from a Tensor at the provided index

// get exposes this operation as part of the public API.

// get exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) get[T](index []int) T {
	offset := t.offset_index(index)
	return t.data.get[T](offset)
}

// get_nth returns a scalar value from a Tensor at the provided index

// get_nth exposes this operation as part of the public API.

// get_nth exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) get_nth[T](n int) T {
	if t.is_row_major_contiguous() && n >= 0 && n < t.size {
		return t.data.get[T](n)
	}
	index := t.nth_index(n)
	return t.get[T](index)
}

// offset_index returns the index to a Tensor's data at
// a given index

// offset_index exposes this operation as part of the public API.

// offset_index exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) offset_index[T](index []int) int {
	mut offset := 0
	for i in 0 .. t.rank() {
		mut j := index[i]
		if j < 0 {
			j += t.shape[i]
		}
		offset += j * t.strides[i]
		if t.strides[i] < 0 {
			offset += t.shape[i] - 1
		}
	}

	if offset < 0 {
		offset = t.size - 1 + offset
	}

	return offset
}

// nth_index returns the nth index of a Tensor's shape
// for `n == 2` and a `shape` of `[2, 2]` the _nth index_ is `[1, 0]`
// and for a `shape` of `[2, 3]` and `n == 3` the _nth index_ is `[0, 1, 1]`
// in sorted order.
pub fn (t &Tensor[T]) nth_index[T](n int) []int {
	rank := t.rank()
	mut index := []int{len: rank}
	for i in 0 .. rank {
		index[i] = 0
	}
	mut i := 0
	for {
		if i == n {
			return index
		}
		i += 1
		for j := rank - 1; j >= 0; j -= 1 {
			if index[j] < t.shape[j] - 1 {
				index[j] += 1
				break
			}
			index[j] = 0
		}
	}

	return index
}
