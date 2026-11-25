module vtl

pub fn ufunc_unary[T](input &Tensor[T], operation fn (T) T) !&Tensor[T] {
	mut output := zeros_like[T](input)
	ufunc_unary_into[T](input, output, unsafe { nil }, false, operation)!
	return output
}

// ufunc_unary_into preserves masked-out values in the caller-owned output.
pub fn ufunc_unary_into[T](input &Tensor[T], output &Tensor[T], where_mask &Tensor[bool], has_where bool, operation fn (T) T) ! {
	if input.shape != output.shape {
		return error('ufunc_unary_into output shape must match input shape')
	}
	if has_where && (where_mask == unsafe { nil } || where_mask.shape != input.shape) {
		return error('ufunc_unary_into where mask must match input shape')
	}
	for i in 0 .. input.size() {
		if !has_where || where_mask.get_nth(i) {
			output.set_nth(i, operation(input.get_nth(i)))
		}
	}
}

pub fn ufunc_binary[T](left &Tensor[T], right &Tensor[T], operation fn (T, T) T) !&Tensor[T] {
	mut output := zeros_like[T](left)
	ufunc_binary_into[T](left, right, output, unsafe { nil }, false, operation)!
	return output
}

pub fn ufunc_binary_into[T](left &Tensor[T], right &Tensor[T], output &Tensor[T], where_mask &Tensor[bool], has_where bool, operation fn (T, T) T) ! {
	if left.shape != right.shape || left.shape != output.shape {
		return error('ufunc_binary_into operands and output must have identical shapes')
	}
	if has_where && (where_mask == unsafe { nil } || where_mask.shape != left.shape) {
		return error('ufunc_binary_into where mask must match operand shape')
	}
	for i in 0 .. left.size() {
		if !has_where || where_mask.get_nth(i) {
			output.set_nth(i, operation(left.get_nth(i), right.get_nth(i)))
		}
	}
}

// ufunc_reduce applies a binary operation in logical row-major order.
// The identity is returned unchanged for empty inputs.
pub fn ufunc_reduce[T](input &Tensor[T], identity T, operation fn (T, T) T) T {
	mut result := identity
	for i in 0 .. input.size() {
		result = operation(result, input.get_nth(i))
	}
	return result
}

// ufunc_accumulate returns the inclusive prefix operation in logical
// row-major order, preserving the input shape.
pub fn ufunc_accumulate[T](input &Tensor[T], identity T, operation fn (T, T) T) !&Tensor[T] {
	mut output := zeros_like[T](input)
	mut accumulator := identity
	for i in 0 .. input.size() {
		accumulator = operation(accumulator, input.get_nth(i))
		output.set_nth(i, accumulator)
	}
	return output
}

pub fn ufunc_reduce_axis[T](input &Tensor[T], axis int, identity T, operation fn (T, T) T) !&Tensor[T] {
	rank := input.rank()
	mut axis_index := axis
	if axis_index < 0 {
		axis_index += rank
	}
	if axis_index < 0 || axis_index >= rank {
		return error('ufunc_reduce_axis axis ${axis} is out of range for rank ${rank}')
	}
	mut output_shape := input.shape.clone()
	output_shape.delete(axis_index)
	mut output := zeros[T](output_shape)
	for flat in 0 .. output.size() {
		mut output_index := []int{len: output_shape.len}
		decode_flat_coordinate(flat, output_shape, mut output_index)
		mut input_index := []int{len: rank}
		for out_axis, value in output_index {
			input_index[if out_axis < axis_index { out_axis } else { out_axis + 1 }] = value
		}
		mut reduced := identity
		for axis_value in 0 .. input.shape[axis_index] {
			input_index[axis_index] = axis_value
			reduced = operation(reduced, input.get(input_index))
		}
		output.set_nth(flat, reduced)
	}
	return output
}

pub fn ufunc_accumulate_axis[T](input &Tensor[T], axis int, identity T, operation fn (T, T) T) !&Tensor[T] {
	rank := input.rank()
	mut axis_index := axis
	if axis_index < 0 {
		axis_index += rank
	}
	if axis_index < 0 || axis_index >= rank {
		return error('ufunc_accumulate_axis axis ${axis} is out of range for rank ${rank}')
	}
	mut output := zeros_like[T](input)
	mut line_shape := input.shape.clone()
	line_shape.delete(axis_index)
	line_count := ufunc_shape_size(line_shape)
	for flat in 0 .. line_count {
		mut line_index := []int{len: line_shape.len}
		decode_flat_coordinate(flat, line_shape, mut line_index)
		mut input_index := []int{len: rank}
		for line_axis, value in line_index {
			input_index[if line_axis < axis_index { line_axis } else { line_axis + 1 }] = value
		}
		mut accumulator := identity
		for axis_value in 0 .. input.shape[axis_index] {
			input_index[axis_index] = axis_value
			accumulator = operation(accumulator, input.get(input_index))
			output.set(input_index, accumulator)
		}
	}
	return output
}

fn ufunc_shape_size(shape []int) int {
	mut size := 1
	for dimension in shape {
		size *= dimension
	}
	return size
}

// ufunc_outer computes operation(left[i], right[j]) for every flattened
// logical pair and returns a matrix of shape [left.size, right.size].
pub fn ufunc_outer[T](left &Tensor[T], right &Tensor[T], operation fn (T, T) T) !&Tensor[T] {
	mut output := zeros[T]([left.size(), right.size()])
	for i in 0 .. left.size() {
		for j in 0 .. right.size() {
			output.set([i, j], operation(left.get_nth(i), right.get_nth(j)))
		}
	}
	return output
}

// ufunc_reduceat applies a segmented reduction in flattened logical order.
// Each start must be in bounds; non-increasing adjacent starts select one
// element, following NumPy's reduceat segment rule.
pub fn ufunc_reduceat[T](input &Tensor[T], indices []int, operation fn (T, T) T) !&Tensor[T] {
	if indices.len == 0 {
		return error('ufunc_reduceat requires at least one index')
	}
	mut output := zeros[T]([indices.len])
	for segment, start in indices {
		if start < 0 || start >= input.size() {
			return error('ufunc_reduceat index ${start} is out of bounds')
		}
		mut end := input.size()
		if segment + 1 < indices.len {
			next := indices[segment + 1]
			if next < 0 || next > input.size() {
				return error('ufunc_reduceat index ${next} is out of bounds')
			}
			end = if next <= start { start + 1 } else { next }
		}
		mut result := input.get_nth(start)
		for i in start + 1 .. end {
			result = operation(result, input.get_nth(i))
		}
		output.set_nth(segment, result)
	}
	return output
}

// ufunc_at mutates flat logical positions in sequence. Repeated indices are
// applied repeatedly, and updates may contain one broadcast value or one
// value per index. Bounds are validated before the first mutation.
pub fn ufunc_at[T](mut target &Tensor[T], indices []int, updates []T, operation fn (T, T) T) ! {
	if updates.len != 1 && updates.len != indices.len {
		return error('ufunc_at updates must contain one value or match index count')
	}
	for index in indices {
		if index < 0 || index >= target.size() {
			return error('ufunc_at index ${index} is out of bounds')
		}
	}
	for i, index in indices {
		update := updates[if updates.len == 1 { 0 } else { i }]
		target.set_nth(index, operation(target.get_nth(index), update))
	}
}
