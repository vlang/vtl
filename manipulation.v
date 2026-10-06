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
			// Repeated axes reverse twice, as in NumPy.
			flip_axis[normalized] = !flip_axis[normalized]
		}
	}

	mut result := empty[T](t.shape, memory: t.memory)
	mut index := []int{len: rank}
	mut source_index := []int{len: rank}
	for linear in 0 .. t.size {
		mut remainder := linear
		for axis := rank - 1; axis >= 0; axis-- {
			dimension := t.shape[axis]
			index[axis] = remainder % dimension
			remainder /= dimension
		}
		for axis in 0 .. rank {
			source_index[axis] = if flip_axis[axis] {
				t.shape[axis] - 1 - index[axis]
			} else {
				index[axis]
			}
		}
		result.set_nth(linear, t.get(source_index))
	}
	return result
}
