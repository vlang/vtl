module vtl

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
