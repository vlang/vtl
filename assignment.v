module vtl

// PutMode controls how flat indices outside a tensor's range are handled.
pub enum PutMode {
	raise
	wrap
	clip
}

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

// put replaces values at row-major logical flat indices. Values repeat when
// there are fewer values than indices. Negative indices count from the end.
// Invalid indices return an error without changing the tensor.
pub fn (mut t Tensor[T]) put[T](indices &Tensor[int], values &Tensor[T]) ! {
	t.put_with_mode[T](indices, values, .raise)!
}

// putmask replaces values where mask is true, in row-major logical order.
// Update values repeat cyclically by row-major flat position, matching NumPy's
// putmask semantics. The mask must match the tensor shape exactly.
pub fn (mut t Tensor[T]) putmask[T](mask &Tensor[bool], values &Tensor[T]) ! {
	if mask.shape != t.shape {
		return error('putmask: mask shape ${mask.shape} must match tensor shape ${t.shape}')
	}
	mut selected := []bool{len: t.size}
	mut selected_count := 0
	for flat_index in 0 .. t.size {
		selected[flat_index] = mask.get_nth[bool](flat_index)
		if selected[flat_index] {
			selected_count++
		}
	}
	if selected_count == 0 {
		return
	}
	if values.size == 0 {
		return error('putmask: at least one value is required for selected positions')
	}
	mut update_values := []T{}
	if tensor_shares_storage[T, T](t, values) {
		update_values = values.to_array()
	}
	for flat_index in 0 .. t.size {
		if selected[flat_index] {
			value := if update_values.len > 0 {
				update_values[flat_index % values.size]
			} else {
				values.get_nth[T](flat_index % values.size)
			}
			t.set_nth[T](flat_index, value)
		}
	}
}

// put_with_mode replaces values at row-major logical flat indices, using mode
// to handle indices outside the flattened tensor's range. Repeated indices
// follow row-major write order, so the last value wins. All indices are
// validated before any writes are made.
pub fn (mut t Tensor[T]) put_with_mode[T](indices &Tensor[int], values &Tensor[T], mode PutMode) ! {
	if indices.size == 0 {
		return
	}
	if t.size == 0 {
		return error('put cannot index an empty tensor')
	}
	if values.size == 0 {
		return error('put requires at least one value when indices are non-empty')
	}
	mut selected_indices := []int{}
	if tensor_shares_storage[T, int](t, indices) {
		selected_indices = indices.to_array()
	}
	for position in 0 .. indices.size {
		selected := if selected_indices.len > 0 {
			selected_indices[position]
		} else {
			indices.get_nth[int](position)
		}
		_ = normalize_put_index(selected, t.size, mode)!
	}
	mut update_values := []T{}
	if tensor_shares_storage[T, T](t, values) {
		update_values = values.to_array()
	}
	for position in 0 .. indices.size {
		selected := if selected_indices.len > 0 {
			selected_indices[position]
		} else {
			indices.get_nth[int](position)
		}
		index := normalize_put_index(selected, t.size, mode)!
		update := if update_values.len > 0 {
			update_values[position % values.size]
		} else {
			values.get_nth[T](position % values.size)
		}
		t.set_nth[T](index, update)
	}
}

fn tensor_shares_storage[T, U](a &Tensor[T], b &Tensor[U]) bool {
	a_start := unsafe { usize(a.data.data.data) }
	a_end := a_start + usize(a.data.data.len) * usize(sizeof(T))
	b_start := unsafe { usize(b.data.data.data) }
	b_end := b_start + usize(b.data.data.len) * usize(sizeof(U))
	return a_start < b_end && b_start < a_end
}

fn normalize_put_index(selected int, size int, mode PutMode) !int {
	return match mode {
		.raise {
			index := if selected < 0 { selected + size } else { selected }
			if index < 0 || index >= size {
				return error('put index ${selected} is out of range for flattened size ${size}')
			}
			index
		}
		.wrap {
			remainder := selected % size
			if remainder < 0 { remainder + size } else { remainder }
		}
		.clip {
			if selected < 0 {
				0
			} else if selected >= size {
				size - 1
			} else {
				selected
			}
		}
	}
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
