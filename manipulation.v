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

// rot90 rotates a tensor counterclockwise by k quarter-turns in its first two
// dimensions, matching NumPy's default axes=(0, 1).
pub fn rot90[T](t &Tensor[T]) !&Tensor[T] {
	return rot90_axes[T](t, 1, [0, 1])
}

// rot90_k rotates in the first two dimensions by k counterclockwise quarter-turns.
pub fn rot90_k[T](t &Tensor[T], k int) !&Tensor[T] {
	return rot90_axes[T](t, k, [0, 1])
}

// rot90_axes rotates counterclockwise in the plane defined by two distinct axes.
pub fn rot90_axes[T](t &Tensor[T], k int, axes []int) !&Tensor[T] {
	if t.rank() < 2 {
		return error('rot90: tensor must have at least two dimensions')
	}
	if axes.len != 2 {
		return error('rot90: exactly two axes are required')
	}
	mut axis0 := axes[0]
	mut axis1 := axes[1]
	if axis0 < 0 {
		axis0 += t.rank()
	}
	if axis1 < 0 {
		axis1 += t.rank()
	}
	if axis0 < 0 || axis0 >= t.rank() || axis1 < 0 || axis1 >= t.rank() {
		return error('rot90: axes ${axes} out of bounds for tensor with ${t.rank()} dimensions')
	}
	if axis0 == axis1 {
		return error('rot90: axes must be different')
	}
	turns := ((k % 4) + 4) % 4
	mut out_shape := t.shape.clone()
	if turns % 2 == 1 {
		out_shape[axis0] = t.shape[axis1]
		out_shape[axis1] = t.shape[axis0]
	}
	mut result := empty[T](out_shape, memory: t.memory)
	mut output_index := []int{len: t.rank()}
	mut source_index := []int{len: t.rank()}
	for linear in 0 .. result.size {
		mut remainder := linear
		for dim := t.rank() - 1; dim >= 0; dim-- {
			output_index[dim] = remainder % out_shape[dim]
			remainder /= out_shape[dim]
			source_index[dim] = output_index[dim]
		}
		match turns {
			1 {
				source_index[axis0] = output_index[axis1]
				source_index[axis1] = t.shape[axis1] - 1 - output_index[axis0]
			}
			2 {
				source_index[axis0] = t.shape[axis0] - 1 - output_index[axis0]
				source_index[axis1] = t.shape[axis1] - 1 - output_index[axis1]
			}
			3 {
				source_index[axis0] = t.shape[axis0] - 1 - output_index[axis1]
				source_index[axis1] = output_index[axis0]
			}
			else {}
		}
		result.set_nth(linear, t.get(source_index))
	}
	return result
}
