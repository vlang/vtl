module vtl

import math

// argmax_axis returns the indices of the maximum values along the given axis.

// argmax_axis exposes this operation as part of the public API.

// argmax_axis exposes this operation as part of the public API.
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

	mut strides := []int{len: rank}
	strides[rank - 1] = 1
	for i := rank - 2; i >= 0; i-- {
		strides[i] = strides[i + 1] * shape[i + 1]
	}

	mut out_shape := shape.clone()
	out_shape[na] = 1
	mut result := empty[int](out_shape)
	axis_stride := strides[na]

	mut out_strides := []int{len: rank}
	out_strides[rank - 1] = 1
	for i := rank - 2; i >= 0; i-- {
		out_strides[i] = out_strides[i + 1] * out_shape[i + 1]
	}

	mut outer_idx := []int{len: rank}
	for {
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
		mut out_lin := 0
		for i := 0; i < rank; i++ {
			out_lin += outer_idx[i] * out_strides[i]
		}
		result.set_nth(out_lin, best_arg)

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

// argmin_axis exposes this operation as part of the public API.

// argmin_axis exposes this operation as part of the public API.
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

	mut out_strides := []int{len: rank}
	out_strides[rank - 1] = 1
	for i := rank - 2; i >= 0; i-- {
		out_strides[i] = out_strides[i + 1] * out_shape[i + 1]
	}

	mut outer_idx := []int{len: rank}
	for {
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
		mut out_lin := 0
		for i := 0; i < rank; i++ {
			out_lin += outer_idx[i] * out_strides[i]
		}
		result.set_nth(out_lin, best_arg)

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

// max_axis exposes this operation as part of the public API.

// max_axis exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) max_axis[T](axis int) !&Tensor[T] {
	shape := t.shape
	rank := shape.len
	mut na := axis
	if axis < 0 {
		na = axis + rank
	}
	if na < 0 || na >= rank {
		return error('max_axis: axis ${axis} out of bounds for shape ${shape}')
	}

	mut strides := []int{len: rank}
	strides[rank - 1] = 1
	for i := rank - 2; i >= 0; i-- {
		strides[i] = strides[i + 1] * shape[i + 1]
	}
	axis_stride := strides[na]

	mut out_shape := shape.clone()
	out_shape[na] = 1
	mut result := empty[T](out_shape)

	// compute strides for the output tensor
	mut out_strides := []int{len: rank}
	out_strides[rank - 1] = 1
	for i := rank - 2; i >= 0; i-- {
		out_strides[i] = out_strides[i + 1] * out_shape[i + 1]
	}

	// iterate over all positions in the output tensor
	mut outer_idx := []int{len: rank}
	for {
		// compute linear index in input (with na=0)
		mut base_lin := 0
		for i := 0; i < rank; i++ {
			base_lin += outer_idx[i] * strides[i]
		}
		// find max along na axis
		mut best_val := t.get_nth(base_lin)
		for j := 1; j < shape[na]; j++ {
			val := t.get_nth(base_lin + j * axis_stride)
			if val > best_val {
				best_val = val
			}
		}
		// compute output linear index
		mut out_lin := 0
		for i := 0; i < rank; i++ {
			out_lin += outer_idx[i] * out_strides[i]
		}
		result.set_nth(out_lin, best_val)

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

// min_axis exposes this operation as part of the public API.

// min_axis exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) min_axis[T](axis int) !&Tensor[T] {
	shape := t.shape
	rank := shape.len
	mut na := axis
	if axis < 0 {
		na = axis + rank
	}
	if na < 0 || na >= rank {
		return error('min_axis: axis ${axis} out of bounds for shape ${shape}')
	}

	mut strides := []int{len: rank}
	strides[rank - 1] = 1
	for i := rank - 2; i >= 0; i-- {
		strides[i] = strides[i + 1] * shape[i + 1]
	}
	axis_stride := strides[na]

	mut out_shape := shape.clone()
	out_shape[na] = 1
	mut result := empty[T](out_shape)

	mut out_strides := []int{len: rank}
	out_strides[rank - 1] = 1
	for i := rank - 2; i >= 0; i-- {
		out_strides[i] = out_strides[i + 1] * out_shape[i + 1]
	}

	mut outer_idx := []int{len: rank}
	for {
		mut base_lin := 0
		for i := 0; i < rank; i++ {
			base_lin += outer_idx[i] * strides[i]
		}
		mut best_val := t.get_nth(base_lin)
		for j := 1; j < shape[na]; j++ {
			val := t.get_nth(base_lin + j * axis_stride)
			if val < best_val {
				best_val = val
			}
		}
		mut out_lin := 0
		for i := 0; i < rank; i++ {
			out_lin += outer_idx[i] * out_strides[i]
		}
		result.set_nth(out_lin, best_val)

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

// max_axis_squeeze returns maximum values with the reduced axis removed,
// matching NumPy's default keepdims=false shape behavior.
pub fn (t &Tensor[T]) max_axis_squeeze[T](axis int) !&Tensor[T] {
	result := t.max_axis[T](axis)!
	na := if axis < 0 { axis + t.rank() } else { axis }
	return squeeze_reduction_axis[T](result, na)
}

// min_axis_squeeze returns minimum values with the reduced axis removed,
// matching NumPy's default keepdims=false shape behavior.
pub fn (t &Tensor[T]) min_axis_squeeze[T](axis int) !&Tensor[T] {
	result := t.min_axis[T](axis)!
	na := if axis < 0 { axis + t.rank() } else { axis }
	return squeeze_reduction_axis[T](result, na)
}

// max_axes reduces several axes by maximum. Negative axes are supported;
// duplicate axes return an error. An empty axes list returns a copy.
pub fn (t &Tensor[T]) max_axes[T](axes []int, keepdims bool) !&Tensor[T] {
	return extrema_axes[T](t, axes, keepdims, true)
}

// min_axes reduces several axes by minimum. Negative axes are supported;
// duplicate axes return an error. An empty axes list returns a copy.
pub fn (t &Tensor[T]) min_axes[T](axes []int, keepdims bool) !&Tensor[T] {
	return extrema_axes[T](t, axes, keepdims, false)
}

fn extrema_axes[T](t &Tensor[T], axes []int, keepdims bool, maximum bool) !&Tensor[T] {
	if axes.len == 0 {
		return t.copy(.row_major)
	}
	mut normalized := []int{cap: axes.len}
	for axis in axes {
		axis_index := if axis < 0 { axis + t.rank() } else { axis }
		if axis_index < 0 || axis_index >= t.rank() {
			return error('axis ${axis} out of bounds for rank ${t.rank()}')
		}
		if axis_index in normalized {
			return error('duplicate axis ${axis}')
		}
		normalized << axis_index
	}
	for i in 0 .. normalized.len {
		for j in i + 1 .. normalized.len {
			if normalized[i] < normalized[j] {
				normalized[i], normalized[j] = normalized[j], normalized[i]
			}
		}
	}
	mut output_shape := t.shape.clone()
	for axis_index in normalized {
		if output_shape[axis_index] == 0 {
			return error('cannot reduce an empty axis')
		}
		if keepdims {
			output_shape[axis_index] = 1
		}
	}
	if !keepdims {
		for axis_index in normalized {
			output_shape.delete(axis_index)
		}
	}
	if 0 in output_shape {
		return empty[T](output_shape)
	}
	mut result := t.copy(.row_major)
	for axis_index in normalized {
		if maximum {
			if keepdims {
				result = result.max_axis[T](axis_index)!
			} else {
				result = result.max_axis_squeeze[T](axis_index)!
			}
		} else if keepdims {
			result = result.min_axis[T](axis_index)!
		} else {
			result = result.min_axis_squeeze[T](axis_index)!
		}
	}
	return result
}

fn squeeze_reduction_axis[T](t &Tensor[T], axis int) !&Tensor[T] {
	mut shape := t.shape.clone()
	shape.delete(axis)
	if shape.len == 0 {
		shape = [1]
	}
	return t.reshape[T](shape)
}

// argmax_axis_squeeze returns the maximum indices with the reduced axis
// removed, matching NumPy's default keepdims=false behavior.
pub fn (t &Tensor[T]) argmax_axis_squeeze[T](axis int) !&Tensor[int] {
	result := t.argmax_axis[T](axis)!
	na := if axis < 0 { axis + t.rank() } else { axis }
	return squeeze_reduction_axis[int](result, na)
}

// argmin_axis_squeeze returns the minimum indices with the reduced axis
// removed, matching NumPy's default keepdims=false behavior.
pub fn (t &Tensor[T]) argmin_axis_squeeze[T](axis int) !&Tensor[int] {
	result := t.argmin_axis[T](axis)!
	na := if axis < 0 { axis + t.rank() } else { axis }
	return squeeze_reduction_axis[int](result, na)
}

// argmax returns the indices of the maximum values along the given axis.

// argmax exposes this operation as part of the public API.

// argmax exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) argmax[T](axis int) !&Tensor[int] {
	return t.argmax_axis(axis)
}

// argmin returns the indices of the minimum values along the given axis.

// argmin exposes this operation as part of the public API.

// argmin exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) argmin[T](axis int) !&Tensor[int] {
	return t.argmin_axis(axis)
}

// nanargmax returns the flattened index of the largest non-NaN value. The
// first index wins ties, matching NumPy. Empty and all-NaN tensors return an error.
pub fn (t &Tensor[T]) nanargmax[T]() !int {
	mut found := false
	mut best_value := f64(0)
	mut best_index := 0
	for index in 0 .. t.size {
		value := f64(t.get_nth(index))
		if math.is_nan(value) {
			continue
		}
		if !found || value > best_value {
			found = true
			best_value = value
			best_index = index
		}
	}
	if !found {
		return error('nanargmax: tensor contains no non-NaN values')
	}
	return best_index
}

// nanargmin returns the flattened index of the smallest non-NaN value. The
// first index wins ties, matching NumPy. Empty and all-NaN tensors return an error.
pub fn (t &Tensor[T]) nanargmin[T]() !int {
	mut found := false
	mut best_value := f64(0)
	mut best_index := 0
	for index in 0 .. t.size {
		value := f64(t.get_nth(index))
		if math.is_nan(value) {
			continue
		}
		if !found || value < best_value {
			found = true
			best_value = value
			best_index = index
		}
	}
	if !found {
		return error('nanargmin: tensor contains no non-NaN values')
	}
	return best_index
}

// cumsum returns the cumulative sum along the given axis.
// Only meaningful for numeric types; bool and string follow their respective + semantics.

// cumsum exposes this operation as part of the public API.

// cumsum exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) cumsum[T](axis int) !&Tensor[T] {
	shape := t.shape
	rank := shape.len
	if rank == 0 {
		return error('cumsum: tensor has no dimensions')
	}
	mut na := axis
	if axis < 0 {
		na = axis + rank
	}
	if na < 0 || na >= rank {
		return error('cumsum: axis ${axis} out of bounds for shape ${shape}')
	}
	mut strides := []int{len: rank}
	strides[rank - 1] = 1
	for i := rank - 2; i >= 0; i-- {
		strides[i] = strides[i + 1] * shape[i + 1]
	}
	axis_stride := strides[na]
	n_axis := shape[na]
	mut result := zeros[T](shape)
	mut outer_idx := []int{len: rank}
	for {
		mut base_lin := 0
		for i := 0; i < rank; i++ {
			base_lin += outer_idx[i] * strides[i]
		}
		$if T is bool {
			mut acc := false
			for j := 0; j < n_axis; j++ {
				lin := base_lin + j * axis_stride
				acc = acc || t.get_nth(lin)
				result.set_nth(lin, acc)
			}
		} $else $if T is string {
			mut acc := ''
			for j := 0; j < n_axis; j++ {
				lin := base_lin + j * axis_stride
				acc = acc + t.get_nth(lin)
				result.set_nth(lin, acc)
			}
		} $else {
			mut acc := cast[T](0)
			for j := 0; j < n_axis; j++ {
				lin := base_lin + j * axis_stride
				acc = acc + t.get_nth(lin)
				result.set_nth(lin, acc)
			}
		}
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

// cumprod returns the cumulative product along the given axis.
// Only meaningful for numeric types; bool follows && semantics.

// cumprod exposes this operation as part of the public API.

// cumprod exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) cumprod[T](axis int) !&Tensor[T] {
	shape := t.shape
	rank := shape.len
	if rank == 0 {
		return error('cumprod: tensor has no dimensions')
	}
	mut na := axis
	if axis < 0 {
		na = axis + rank
	}
	if na < 0 || na >= rank {
		return error('cumprod: axis ${axis} out of bounds for shape ${shape}')
	}
	mut strides := []int{len: rank}
	strides[rank - 1] = 1
	for i := rank - 2; i >= 0; i-- {
		strides[i] = strides[i + 1] * shape[i + 1]
	}
	axis_stride := strides[na]
	n_axis := shape[na]
	mut result := zeros[T](shape)
	mut outer_idx := []int{len: rank}
	for {
		mut base_lin := 0
		for i := 0; i < rank; i++ {
			base_lin += outer_idx[i] * strides[i]
		}
		$if T is bool {
			mut acc := true
			for j := 0; j < n_axis; j++ {
				lin := base_lin + j * axis_stride
				acc = acc && t.get_nth(lin)
				result.set_nth(lin, acc)
			}
		} $else $if T is string {
			// cumprod on string is not meaningful; return zeros (empty strings)
			_ = n_axis
		} $else {
			mut acc := cast[T](1)
			for j := 0; j < n_axis; j++ {
				lin := base_lin + j * axis_stride
				acc = acc * t.get_nth(lin)
				result.set_nth(lin, acc)
			}
		}
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
