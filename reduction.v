module vtl

// argmax_axis returns the indices of the maximum values along the given axis.
@[inline]
pub fn (t &Tensor[T]) argmax_axis[T](axis int) !&Tensor[int] {
	shape := t.shape
	rank := shape.len
	if rank == 0 {
		return error('argmax_axis: tensor has no dimensions')
	}
	mut na := axis
	if axis < 0 {
		na = axis + rank
	}
	if na < 0 || na >= rank {
		return error('argmax_axis: axis ${axis} out of bounds for shape ${shape}')
	}

	// Compute row-major strides
	mut strides := []int{len: rank}
	strides[rank - 1] = 1
	for i := rank - 2; i >= 0; i-- {
		strides[i] = strides[i + 1] * shape[i + 1]
	}

	mut out_shape := shape.clone()
	out_shape[na] = 1
	mut result := empty[int](out_shape)
	axis_stride := strides[na]

	mut outer_idx := []int{len: rank}
	outer_idx[na] = 0
	for {
		if outer_idx[na] >= shape[na] {
			break
		}
		// Compute base linear index
		mut base_lin := 0
		for i := 0; i < rank; i++ {
			base_lin += outer_idx[i] * strides[i]
		}

		mut best_val := t.get_nth(base_lin)
		mut best_arg := 0
		for j := 1; j < shape[na]; j++ {
			val := t.get_nth(base_lin + j * axis_stride)
			if val > best_val {
				best_val = val
				best_arg = j
			}
		}
		result.set_nth(base_lin, best_arg)

		// Advance outer_idx (skip dimension na)
		mut done := true
		for i := rank - 1; i >= 0; i-- {
			if i == na {
				continue
			}
			outer_idx[i]++
			if outer_idx[i] < shape[i] {
				done = false
				break
			}
			outer_idx[i] = 0
		}
		if done {
			break
		}
	}
	return result
}

// argmin_axis returns the indices of the minimum values along the given axis.
@[inline]
pub fn (t &Tensor[T]) argmin_axis[T](axis int) !&Tensor[int] {
	shape := t.shape
	rank := shape.len
	if rank == 0 {
		return error('argmin_axis: tensor has no dimensions')
	}
	mut na := axis
	if axis < 0 {
		na = axis + rank
	}
	if na < 0 || na >= rank {
		return error('argmin_axis: axis ${axis} out of bounds for shape ${shape}')
	}

	mut strides := []int{len: rank}
	strides[rank - 1] = 1
	for i := rank - 2; i >= 0; i-- {
		strides[i] = strides[i + 1] * shape[i + 1]
	}

	mut out_shape := shape.clone()
	out_shape[na] = 1
	mut result := empty[int](out_shape)
	axis_stride := strides[na]

	mut outer_idx := []int{len: rank}
	outer_idx[na] = 0
	for {
		if outer_idx[na] >= shape[na] {
			break
		}
		mut base_lin := 0
		for i := 0; i < rank; i++ {
			base_lin += outer_idx[i] * strides[i]
		}

		mut best_val := t.get_nth(base_lin)
		mut best_arg := 0
		for j := 1; j < shape[na]; j++ {
			val := t.get_nth(base_lin + j * axis_stride)
			if val < best_val {
				best_val = val
				best_arg = j
			}
		}
		result.set_nth(base_lin, best_arg)

		mut done := true
		for i := rank - 1; i >= 0; i-- {
			if i == na {
				continue
			}
			outer_idx[i]++
			if outer_idx[i] < shape[i] {
				done = false
				break
			}
			outer_idx[i] = 0
		}
		if done {
			break
		}
	}
	return result
}

// max_axis returns the maximum value along the given axis as a reduced tensor.
@[inline]
pub fn (t &Tensor[T]) max_axis[T](axis int) !&Tensor[T] {
	arg_idx := t.argmax_axis(axis)!
	shape := t.shape
	rank := shape.len
	mut na := axis
	if axis < 0 {
		na = axis + rank
	}

	mut strides := []int{len: rank}
	strides[rank - 1] = 1
	for i := rank - 2; i >= 0; i-- {
		strides[i] = strides[i + 1] * shape[i + 1]
	}

	mut out_shape := shape.clone()
	out_shape[na] = 1
	mut result := empty[T](out_shape)
	axis_stride := strides[na]

	mut outer_idx := []int{len: rank}
	outer_idx[na] = 0
	for {
		if outer_idx[na] >= shape[na] {
			break
		}
		mut base_lin := 0
		for i := 0; i < rank; i++ {
			base_lin += outer_idx[i] * strides[i]
		}

		mut arg_idx_pos := outer_idx.clone()
		arg_idx_pos[na] = 0
		arg := arg_idx.get(arg_idx_pos)
		lin := base_lin + arg * axis_stride
		result.set_nth(base_lin, t.get_nth(lin))

		mut done := true
		for i := rank - 1; i >= 0; i-- {
			if i == na {
				continue
			}
			outer_idx[i]++
			if outer_idx[i] < shape[i] {
				done = false
				break
			}
			outer_idx[i] = 0
		}
		if done {
			break
		}
	}
	return result
}

// min_axis returns the minimum value along the given axis as a reduced tensor.
@[inline]
pub fn (t &Tensor[T]) min_axis[T](axis int) !&Tensor[T] {
	arg_idx := t.argmin_axis(axis)!
	shape := t.shape
	rank := shape.len
	mut na := axis
	if axis < 0 {
		na = axis + rank
	}

	mut strides := []int{len: rank}
	strides[rank - 1] = 1
	for i := rank - 2; i >= 0; i-- {
		strides[i] = strides[i + 1] * shape[i + 1]
	}

	mut out_shape := shape.clone()
	out_shape[na] = 1
	mut result := empty[T](out_shape)
	axis_stride := strides[na]

	mut outer_idx := []int{len: rank}
	outer_idx[na] = 0
	for {
		if outer_idx[na] >= shape[na] {
			break
		}
		mut base_lin := 0
		for i := 0; i < rank; i++ {
			base_lin += outer_idx[i] * strides[i]
		}

		mut arg_idx_pos := outer_idx.clone()
		arg_idx_pos[na] = 0
		arg := arg_idx.get(arg_idx_pos)
		lin := base_lin + arg * axis_stride
		result.set_nth(base_lin, t.get_nth(lin))

		mut done := true
		for i := rank - 1; i >= 0; i-- {
			if i == na {
				continue
			}
			outer_idx[i]++
			if outer_idx[i] < shape[i] {
				done = false
				break
			}
			outer_idx[i] = 0
		}
		if done {
			break
		}
	}
	return result
}

// argmax returns the indices of the maximum values along the given axis.
@[inline]
pub fn (t &Tensor[T]) argmax[T](axis int) !&Tensor[int] {
	return t.argmax_axis(axis)
}

// argmin returns the indices of the minimum values along the given axis.
@[inline]
pub fn (t &Tensor[T]) argmin[T](axis int) !&Tensor[int] {
	return t.argmin_axis(axis)
}