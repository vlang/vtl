module vtl

// masked_select returns the values whose corresponding mask entries are true.
// The mask must have exactly the same shape as the tensor; values are returned
// in row-major logical order, including for non-contiguous views.
pub fn (t &Tensor[T]) masked_select[T](mask &Tensor[bool]) !&Tensor[T] {
	if t.shape != mask.shape {
		return error('mask shape ${mask.shape} must match tensor shape ${t.shape}')
	}
	mut values := []T{}
	mut value_iter := t.iterator[T]()
	mut mask_iter := mask.iterator[bool]()
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
// The mask must have exactly the same shape as the tensor.
pub fn (t &Tensor[T]) masked_fill[T](mask &Tensor[bool], value T) !&Tensor[T] {
	if t.shape != mask.shape {
		return error('mask shape ${mask.shape} must match tensor shape ${t.shape}')
	}
	mut result := empty[T](t.shape, memory: .row_major)
	mut value_iter := t.iterator[T]()
	mut mask_iter := mask.iterator[bool]()
	for {
		current, index := value_iter.next() or { break }
		selected, _ := mask_iter.next() or { break }
		result.set(index, if selected { value } else { current })
	}
	return result
}
