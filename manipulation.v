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
