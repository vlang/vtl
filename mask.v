module vtl

// compress selects flattened values at positions where condition is true.
// As in NumPy, a condition shorter than the input stops selection at its
// length; extra condition values are ignored.
pub fn compress[T](condition &Tensor[bool], t &Tensor[T]) !&Tensor[T] {
	if condition.rank() != 1 {
		return error('compress: condition must be one-dimensional')
	}
	count := if condition.size < t.size { condition.size } else { t.size }
	mut values := []T{}
	for index in 0 .. count {
		if condition.get_nth[bool](index) {
			values << t.get_nth[T](index)
		}
	}
	return from_array[T](values, [values.len], memory: .row_major)
}

// compress_axis selects slices along axis using a one-dimensional condition.
// If condition is shorter than the axis, the remaining slices are omitted.
pub fn compress_axis[T](condition &Tensor[bool], t &Tensor[T], axis int) !&Tensor[T] {
	if condition.rank() != 1 {
		return error('compress_axis: condition must be one-dimensional')
	}
	rank := t.rank()
	if rank == 0 {
		return error('compress_axis: tensor must have at least one dimension')
	}
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('compress_axis: axis ${axis} out of bounds for tensor with ${rank} dimensions')
	}
	axis_size := t.shape[axis_index]
	condition_size := if condition.size < axis_size { condition.size } else { axis_size }
	mut selected := []int{}
	for index in 0 .. condition_size {
		if condition.get_nth[bool](index) {
			selected << index
		}
	}
	mut out_shape := t.shape.clone()
	out_shape[axis_index] = selected.len
	mut result := empty[T](out_shape, memory: .row_major)
	for out_linear in 0 .. result.size {
		out_index := result.nth_index(out_linear)
		mut in_index := out_index.clone()
		in_index[axis_index] = selected[out_index[axis_index]]
		result.set_nth(out_linear, t.get[T](in_index))
	}
	return result
}

// masked_select returns the values whose corresponding mask entries are true.
// The mask must be broadcastable to the tensor shape; values are returned in
// row-major logical order, including for non-contiguous views.
pub fn (t &Tensor[T]) masked_select[T](mask &Tensor[bool]) !&Tensor[T] {
	broadcast_mask := mask.broadcast_to(t.shape) or {
		return error('mask shape ${mask.shape} cannot broadcast to tensor shape ${t.shape}')
	}
	mut values := []T{}
	mut value_iter := t.iterator[T]()
	mut mask_iter := broadcast_mask.iterator[bool]()
	for {
		value, _ := value_iter.next() or { break }
		selected, _ := mask_iter.next() or { break }
		if selected {
			values << value
		}
	}
	return from_array[T](values, [values.len], memory: .row_major)
}

// masked_fill returns a tensor with mask-selected values replaced by value.
// The mask must be broadcastable to the tensor shape.
pub fn (t &Tensor[T]) masked_fill[T](mask &Tensor[bool], value T) !&Tensor[T] {
	broadcast_mask := mask.broadcast_to(t.shape) or {
		return error('mask shape ${mask.shape} cannot broadcast to tensor shape ${t.shape}')
	}
	mut result := empty[T](t.shape, memory: .row_major)
	mut value_iter := t.iterator[T]()
	mut mask_iter := broadcast_mask.iterator[bool]()
	for {
		current, index := value_iter.next() or { break }
		selected, _ := mask_iter.next() or { break }
		result.set(index, if selected { value } else { current })
	}
	return result
}
