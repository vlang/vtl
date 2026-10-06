module vtl

// flip returns a copy of t with the selected axes reversed. With no axes, all
// axes are reversed, matching NumPy's flip default. Negative axes are accepted.
pub fn (t &Tensor[T]) flip[T](axes ...int) !&Tensor[T] {
	rank := t.rank()
	mut flip_axis := []bool{len: rank}
	if axes.len == 0 {
		for i in 0 .. rank {
			flip_axis[i] = true
		}
	} else {
		for axis in axes {
			mut normalized := axis
			if normalized < 0 {
				normalized += rank
			}
			if normalized < 0 || normalized >= rank {
				return error('flip: axis ${axis} out of bounds for tensor with ${rank} dimensions')
			}
			flip_axis[normalized] = !flip_axis[normalized]
		}
	}
	mut result := empty[T](t.shape, memory: t.memory)
	mut index := []int{len: rank}
	mut source_index := []int{len: rank}
	for linear in 0 .. t.size {
		mut remainder := linear
		for dim := rank - 1; dim >= 0; dim-- {
			index[dim] = remainder % t.shape[dim]
			remainder /= t.shape[dim]
		}
		for dim in 0 .. rank {
			source_index[dim] = if flip_axis[dim] {
				t.shape[dim] - 1 - index[dim]
			} else {
				index[dim]
			}
		}
		result.set_nth(linear, t.get(source_index))
	}
	return result
}

// repeat returns a copy with each value repeated the requested number of
// times. If axis is omitted the tensor is flattened first, matching NumPy.
pub fn repeat[T](t &Tensor[T], repeats int) !&Tensor[T] {
	if repeats < 0 {
		return error('repeat: repeats must be non-negative')
	}
	out_shape := [t.size * repeats]
	mut result := empty[T](out_shape, memory: t.memory)
	for i in 0 .. result.size {
		result.set_nth(i, t.get_nth(i / repeats))
	}
	return result
}

// repeat_axis repeats each value along axis and preserves the other dimensions.
pub fn repeat_axis[T](t &Tensor[T], repeats int, axis int) !&Tensor[T] {
	if repeats < 0 {
		return error('repeat_axis: repeats must be non-negative')
	}

	mut normalized_axis := axis
	if normalized_axis < 0 {
		normalized_axis += t.rank()
	}
	if normalized_axis < 0 || normalized_axis >= t.rank() {
		return error('repeat_axis: axis ${axis} out of bounds for tensor with ${t.rank()} dimensions')
	}
	mut out_shape := t.shape.clone()
	out_shape[normalized_axis] *= repeats
	mut result := empty[T](out_shape, memory: t.memory)
	mut index := []int{len: t.rank()}
	mut source_index := []int{len: t.rank()}
	for linear in 0 .. result.size {
		mut remainder := linear
		for dim := t.rank() - 1; dim >= 0; dim-- {
			index[dim] = remainder % out_shape[dim]
			remainder /= out_shape[dim]
		}
		for dim in 0 .. t.rank() {
			source_index[dim] = if dim == normalized_axis {
				index[dim] / repeats
			} else {
				index[dim]
			}
		}
		result.set_nth(linear, t.get(source_index))
	}
	return result
}

// tile repeats t as a block according to reps. Repetition dimensions align
// from the right; shorter reps are padded with ones on the left, as in NumPy.
pub fn tile[T](t &Tensor[T], reps []int) !&Tensor[T] {
	for repeat_count in reps {
		if repeat_count < 0 {
			return error('tile: repetitions must be non-negative')
		}
	}
	out_rank := if reps.len > t.rank() { reps.len } else { t.rank() }
	input_padding := out_rank - t.rank()
	repeat_padding := out_rank - reps.len
	mut out_shape := []int{len: out_rank, init: 1}
	for dim in 0 .. out_rank {
		input_dim := if dim < input_padding { 1 } else { t.shape[dim - input_padding] }
		repeat_count := if dim < repeat_padding { 1 } else { reps[dim - repeat_padding] }
		out_shape[dim] = input_dim * repeat_count
	}
	mut result := empty[T](out_shape, memory: t.memory)
	mut output_index := []int{len: out_rank}
	mut source_index := []int{len: t.rank()}
	for linear in 0 .. result.size {
		mut remainder := linear
		for dim := out_rank - 1; dim >= 0; dim-- {
			output_index[dim] = remainder % out_shape[dim]
			remainder /= out_shape[dim]
		}
		for dim in 0 .. t.rank() {
			source_index[dim] = output_index[dim + input_padding] % t.shape[dim]
		}
		result.set_nth(linear, t.get(source_index))
	}
	return result
}
