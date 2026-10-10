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

	if shape[na] == 0 {
		return error('argmax_axis: cannot reduce an empty axis')
	}

	mut out_shape := shape.clone()
	out_shape[na] = 1
	mut result := empty[int](out_shape)
	if result.size == 0 {
		return result
	}

	mut out_strides := []int{len: rank}
	out_strides[rank - 1] = 1
	for i := rank - 2; i >= 0; i-- {
		out_strides[i] = out_strides[i + 1] * out_shape[i + 1]
	}

	mut outer_idx := []int{len: rank}
	for {
		mut input_idx := outer_idx.clone()
		mut best_val := t.get(input_idx)
		mut best_is_nan := math.is_nan(f64(best_val))
		mut best_arg := 0
		for j := 1; j < shape[na]; j++ {
			input_idx[na] = j
			val := t.get(input_idx)
			if math.is_nan(f64(val)) {
				best_arg = j
				break
			}
			if best_is_nan {
				continue
			}
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

	if shape[na] == 0 {
		return error('argmin_axis: cannot reduce an empty axis')
	}

	mut out_shape := shape.clone()
	out_shape[na] = 1
	mut result := empty[int](out_shape)
	if result.size == 0 {
		return result
	}

	mut out_strides := []int{len: rank}
	out_strides[rank - 1] = 1
	for i := rank - 2; i >= 0; i-- {
		out_strides[i] = out_strides[i + 1] * out_shape[i + 1]
	}

	mut outer_idx := []int{len: rank}
	for {
		mut input_idx := outer_idx.clone()
		mut best_val := t.get(input_idx)
		mut best_is_nan := math.is_nan(f64(best_val))
		mut best_arg := 0
		for j := 1; j < shape[na]; j++ {
			input_idx[na] = j
			val := t.get(input_idx)
			if math.is_nan(f64(val)) {
				best_arg = j
				break
			}
			if best_is_nan {
				continue
			}
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
	return extrema_axis[T](t, axis, true)
}

// min_axis returns the minimum value along the given axis as a reduced tensor.

// min_axis exposes this operation as part of the public API.

// min_axis exposes this operation as part of the public API.
@[inline]
pub fn (t &Tensor[T]) min_axis[T](axis int) !&Tensor[T] {
	return extrema_axis[T](t, axis, false)
}

fn extrema_axis[T](t &Tensor[T], axis int, maximum bool) !&Tensor[T] {
	shape := t.shape
	rank := shape.len
	if rank == 0 {
		return error('extrema_axis: tensor has no dimensions')
	}
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('axis ${axis} out of bounds for shape ${shape}')
	}
	if shape[axis_index] == 0 {
		return error('cannot reduce an empty axis')
	}

	mut output_shape := shape.clone()
	output_shape[axis_index] = 1
	mut result := empty[T](output_shape)
	if result.size == 0 {
		return result
	}
	mut output_strides := []int{len: rank}
	output_strides[rank - 1] = 1
	for dimension := rank - 2; dimension >= 0; dimension-- {
		output_strides[dimension] = output_strides[dimension + 1] * output_shape[dimension + 1]
	}
	mut axis_stride := 1
	for dimension in axis_index + 1 .. rank {
		axis_stride *= shape[dimension]
	}
	mut outer_index := []int{len: rank}
	for {
		mut base_index := 0
		for dimension, coordinate in outer_index {
			base_index = base_index * shape[dimension] + coordinate
		}
		mut best_value := t.get_nth(base_index)
		mut best_is_nan := value_is_nan[T](best_value)
		for position in 1 .. shape[axis_index] {
			value := t.get_nth(base_index + position * axis_stride)
			value_nan := value_is_nan[T](value)
			if value_nan {
				best_value = value
				best_is_nan = true
			} else if !best_is_nan && (if maximum { value > best_value } else { value < best_value }) {
				best_value = value
			}
		}
		mut output_index := 0
		for dimension, coordinate in outer_index {
			output_index += coordinate * output_strides[dimension]
		}
		result.set_nth(output_index, best_value)

		mut done := true
		for dimension := rank - 1; dimension >= 0; dimension-- {
			if dimension == axis_index {
				continue
			}
			outer_index[dimension]++
			if outer_index[dimension] < shape[dimension] {
				done = false
				break
			}
			outer_index[dimension] = 0
		}
		if done {
			break
		}
	}
	return result
}

fn value_is_nan[T](value T) bool {
	$if T is $float {
		return math.is_nan(f64(value))
	} $else {
		return false
	}
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

// argmax_flat returns the flattened index of the first largest value. A NaN
// value is returned immediately, matching NumPy's argmax behavior.
pub fn (t &Tensor[T]) argmax_flat[T]() !int {
	if t.size == 0 {
		return error('argmax_flat: cannot reduce an empty tensor')
	}
	mut best_index := 0
	mut best_value := t.get_nth(0)
	if math.is_nan(f64(best_value)) {
		return best_index
	}
	for index in 1 .. t.size {
		value := t.get_nth(index)
		if math.is_nan(f64(value)) {
			return index
		}
		if value > best_value {
			best_value = value
			best_index = index
		}
	}
	return best_index
}

// argmin_flat returns the flattened index of the first smallest value. A NaN
// value is returned immediately, matching NumPy's argmin behavior.
pub fn (t &Tensor[T]) argmin_flat[T]() !int {
	if t.size == 0 {
		return error('argmin_flat: cannot reduce an empty tensor')
	}
	mut best_index := 0
	mut best_value := t.get_nth(0)
	if math.is_nan(f64(best_value)) {
		return best_index
	}
	for index in 1 .. t.size {
		value := t.get_nth(index)
		if math.is_nan(f64(value)) {
			return index
		}
		if value < best_value {
			best_value = value
			best_index = index
		}
	}
	return best_index
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

// nanargmax_axis returns the index of the largest non-NaN value on axis.
// Set keepdims to retain the reduced axis as a length-one dimension.
pub fn (t &Tensor[T]) nanargmax_axis[T](axis int, keepdims bool) !&Tensor[int] {
	return nanarg_axis[T](t, axis, keepdims, true)
}

// nanargmin_axis returns the index of the smallest non-NaN value on axis.
// Set keepdims to retain the reduced axis as a length-one dimension.
pub fn (t &Tensor[T]) nanargmin_axis[T](axis int, keepdims bool) !&Tensor[int] {
	return nanarg_axis[T](t, axis, keepdims, false)
}

fn nanarg_axis[T](t &Tensor[T], axis int, keepdims bool, maximum bool) !&Tensor[int] {
	rank := t.rank()
	if rank == 0 {
		return error('NaN arg reduction requires a tensor with at least one dimension')
	}
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('NaN arg reduction axis ${axis} out of bounds for rank ${rank}')
	}
	if t.shape[axis_index] == 0 {
		return error('NaN arg reduction cannot reduce an empty axis')
	}
	mut output_shape := []int{cap: rank}
	for dimension, size in t.shape {
		if dimension == axis_index {
			if keepdims {
				output_shape << 1
			}
		} else {
			output_shape << size
		}
	}
	if output_shape.len == 0 {
		output_shape = [1]
	}
	mut result := empty[int](output_shape, memory: .row_major)
	mut output_index := []int{len: output_shape.len}
	for output_flat in 0 .. result.size {
		mut remainder := output_flat
		for dimension := output_shape.len - 1; dimension >= 0; dimension-- {
			output_index[dimension] = remainder % output_shape[dimension]
			remainder /= output_shape[dimension]
		}
		mut input_index := []int{len: rank}
		mut output_dimension := 0
		for dimension in 0 .. rank {
			if dimension == axis_index {
				if keepdims {
					output_dimension++
				}
				continue
			}
			input_index[dimension] = output_index[output_dimension]
			output_dimension++
		}
		mut found := false
		mut best_value := f64(0)
		mut best_index := 0
		for position in 0 .. t.shape[axis_index] {
			input_index[axis_index] = position
			value := f64(t.get(input_index))
			if math.is_nan(value) {
				continue
			}
			if !found || (maximum && value > best_value) || (!maximum && value < best_value) {
				found = true
				best_value = value
				best_index = position
			}
		}
		if !found {
			return error('NaN arg reduction found a slice with no non-NaN values')
		}
		result.set_nth(output_flat, best_index)
	}
	return result
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
