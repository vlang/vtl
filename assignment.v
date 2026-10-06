module vtl

// set copies a scalar value into a Tensor at the provided index

// set exposes this operation as part of the public API.

// set exposes this operation as part of the public API.
@[inline]
pub fn (mut t Tensor[T]) set[T](index []int, val T) {
	offset := t.offset_index(index)
	t.data.set[T](offset, val)
}

// set_nth copies a scalar value into a Tensor at the provided offset

// set_nth exposes this operation as part of the public API.

// set_nth exposes this operation as part of the public API.
@[inline]
pub fn (mut t Tensor[T]) set_nth[T](n int, val T) {
	index := t.nth_index(n)
	t.set[T](index, val)
}

// fill fills an entire Tensor with a given value

// fill exposes this operation as part of the public API.

// fill exposes this operation as part of the public API.
@[inline]
pub fn (mut t Tensor[T]) fill[T](val T) &Tensor[T] {
	t.data.fill[T](val)
	return t
}

// assign sets the values of an Tensor equal to the values of another
// Tensor of the same shape
pub fn (mut t Tensor[T]) assign[T](other &Tensor[T]) !&Tensor[T] {
	mut iters, _ := t.iterators[T]([other])!
	for {
		vals, i := iters.next() or { break }
		t.set(i, vals[1])
	}
	return t
}

// put_along_axis writes values at indices along axis. Indices and values must
// have the same rank as the target and values must match the index shape. The
// index dimensions outside axis may be smaller than the target dimensions.
// Negative indices count from the end.
// When an index is repeated, the last value in row-major iteration order wins.
pub fn (mut t Tensor[T]) put_along_axis[T](indices &Tensor[int], values &Tensor[T], axis int) ! {
	axis_index := validate_axis_update_shapes(t.shape, indices, values, axis, 'put_along_axis')!
	normalized_indices := normalize_axis_update_indices(indices, t.shape[axis_index], 'put_along_axis')!
	update_values := values.to_array()
	mut index_iter := indices.iterator[int]()
	mut position := 0
	for {
		_, mut index := index_iter.next() or { break }
		index[axis_index] = normalized_indices[position]
		t.set(index, update_values[position])
		position++
	}
}

// scatter_add adds values at indices along axis. Duplicate indices accumulate
// in row-major iteration order; negative indices count from the end.
pub fn (mut t Tensor[T]) scatter_add[T](indices &Tensor[int], values &Tensor[T], axis int) ! {
	axis_index := validate_axis_update_shapes(t.shape, indices, values, axis, 'scatter_add')!
	normalized_indices := normalize_axis_update_indices(indices, t.shape[axis_index], 'scatter_add')!
	update_values := values.to_array()
	mut index_iter := indices.iterator[int]()
	mut position := 0
	for {
		_, mut index := index_iter.next() or { break }
		index[axis_index] = normalized_indices[position]
		t.set(index, t.get(index) + update_values[position])
		position++
	}
}

fn normalize_axis_update_indices(indices &Tensor[int], axis_size int, operation string) ![]int {
	mut normalized_indices := []int{cap: indices.size}
	mut index_iter := indices.iterator[int]()
	for {
		selected, _ := index_iter.next() or { break }
		normalized := if selected < 0 { selected + axis_size } else { selected }
		if normalized < 0 || normalized >= axis_size {
			return error('${operation} index ${selected} is out of range for axis size ${axis_size}')
		}
		normalized_indices << normalized
	}
	return normalized_indices
}

fn validate_axis_update_shapes[T](target_shape []int, indices &Tensor[int], values &Tensor[T], axis int, operation string) !int {
	rank := target_shape.len
	axis_index := if axis < 0 { axis + rank } else { axis }
	if rank == 0 || axis_index < 0 || axis_index >= rank {
		return error('${operation} axis ${axis} is out of range for rank ${rank}')
	}
	if indices.rank() != rank || values.rank() != rank {
		return error('${operation} indices and values must have rank ${rank}')
	}
	if indices.shape != values.shape {
		return error('${operation} values shape must match indices shape')
	}
	for dimension in 0 .. rank {
		if dimension != axis_index && indices.shape[dimension] > target_shape[dimension] {
			return error('${operation} index shape exceeds target shape outside the selected axis')
		}
	}
	return axis_index
}
