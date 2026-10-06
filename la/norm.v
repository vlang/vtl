module la

import math
import vtl

// vector_norm computes a p-norm over all tensor elements and returns a one
// element f64 tensor. Positive and negative finite p, zero, and +/-infinity are
// supported. Scaling keeps finite results stable for very large or small data.
pub fn vector_norm[T](t &vtl.Tensor[T], ord f64) !&vtl.Tensor[f64] {
	if math.is_nan(ord) {
		return error('vector_norm order must not be NaN')
	}
	if t.size == 0 {
		if ord == 0 || (ord > 0 && !math.is_inf(ord, 1)) {
			return vtl.from_1d([f64(0)])
		}
		return error('vector_norm is undefined for an empty tensor and this order')
	}
	mut index := []int{len: t.rank()}
	value := vector_norm_slice[T](t, mut index, -1, ord)
	return vtl.from_1d([value])
}

// vector_norm_axis computes a p-norm independently along axis and removes the
// reduced dimension. Negative axes count from the end.
pub fn vector_norm_axis[T](t &vtl.Tensor[T], ord f64, axis int) !&vtl.Tensor[f64] {
	return vector_norm_axis_impl[T](t, ord, axis, false)
}

// vector_norm_axis_keepdims retains the reduced dimension with length one.
pub fn vector_norm_axis_keepdims[T](t &vtl.Tensor[T], ord f64, axis int) !&vtl.Tensor[f64] {
	return vector_norm_axis_impl[T](t, ord, axis, true)
}

// vector_norm_axes computes a p-norm over one or more axes. Negative axes
// count from the end; axes must be unique. At least one axis is required. With
// keepdims, reduced dimensions remain with length one.
pub fn vector_norm_axes[T](t &vtl.Tensor[T], ord f64, axes []int, keepdims bool) !&vtl.Tensor[f64] {
	if math.is_nan(ord) {
		return error('vector_norm_axes order must not be NaN')
	}
	if axes.len == 0 {
		return error('vector_norm_axes requires at least one axis')
	}
	mut normalized := []int{cap: axes.len}
	for axis in axes {
		axis_index := if axis < 0 { axis + t.rank() } else { axis }
		if axis_index < 0 || axis_index >= t.rank() {
			return error('vector_norm_axes axis ${axis} is out of bounds for rank ${t.rank()}')
		}
		if axis_index in normalized {
			return error('vector_norm_axes axes must be unique')
		}
		normalized << axis_index
	}
	mut output_shape := []int{cap: t.rank()}
	for dimension, size in t.shape {
		if dimension in normalized {
			if keepdims {
				output_shape << 1
			}
		} else {
			output_shape << size
		}
	}
	mut reduction_size := 1
	for axis in normalized {
		reduction_size *= t.shape[axis]
	}
	if reduction_size == 0 && ord != 0 && (ord < 0 || math.is_inf(ord, 1)) {
		return error('vector_norm_axes is undefined for an empty reduction and this order')
	}
	reduction_shape := normalized.map(t.shape[it])
	mut result := vtl.empty[f64](output_shape, memory: .row_major)
	mut output_index := []int{len: output_shape.len}
	mut input_index := []int{len: t.rank()}
	for output in 0 .. result.size {
		decode_flat_index(output, output_shape, mut output_index)
		mut output_dimension := 0
		for dimension in 0 .. t.rank() {
			if dimension in normalized {
				input_index[dimension] = 0
				if keepdims {
					output_dimension++
				}
			} else {
				input_index[dimension] = output_index[output_dimension]
				output_dimension++
			}
		}
		value := vector_norm_axes_slice[T](t, mut input_index, normalized, reduction_shape, reduction_size, ord)
		result.set(output_index, value)
	}
	return result
}

fn vector_norm_axes_slice[T](t &vtl.Tensor[T], mut index []int, axes []int, reduction_shape []int, reduction_size int, ord f64) f64 {
	mut count_nonzero := 0
	mut has_zero := false
	mut has_nan := false
	mut scale := if ord < 0 { f64(math.inf(1)) } else { 0.0 }
	mut reduction_index := []int{len: axes.len}
	for item in 0 .. reduction_size {
		decode_flat_index(item, reduction_shape, mut reduction_index)
		for i, axis in axes {
			index[axis] = reduction_index[i]
		}
		value := math.abs(f64(t.get[T](index)))
		if value == 0 {
			has_zero = true
			continue
		}
		count_nonzero++
		if math.is_nan(value) {
			has_nan = true
			continue
		}
		if ord < 0 {
			scale = math.min(scale, value)
		} else {
			scale = math.max(scale, value)
		}
	}
	if ord == 0 { return f64(count_nonzero) }
	if has_nan { return math.nan() }
	if math.is_inf(ord, 1) { return scale }
	if math.is_inf(ord, -1) { return if has_zero { 0 } else { scale } }
	if ord > 0 {
		if scale == 0 || math.is_inf(scale, 1) { return scale }
		mut sum := 0.0
		for item in 0 .. reduction_size {
			decode_flat_index(item, reduction_shape, mut reduction_index)
			for i, axis in axes { index[axis] = reduction_index[i] }
			sum += math.pow(math.abs(f64(t.get[T](index))) / scale, ord)
		}
		return scale * math.pow(sum, 1.0 / ord)
	}
	if has_zero { return 0 }
	if math.is_inf(scale, 1) { return scale }
	mut sum := 0.0
	for item in 0 .. reduction_size {
		decode_flat_index(item, reduction_shape, mut reduction_index)
		for i, axis in axes { index[axis] = reduction_index[i] }
		sum += math.pow(scale / math.abs(f64(t.get[T](index))), -ord)
	}
	return scale * math.pow(sum, 1.0 / ord)
}

fn decode_flat_index(line int, shape []int, mut index []int) {
	mut remainder := line
	for dimension := shape.len - 1; dimension >= 0; dimension-- {
		index[dimension] = remainder % shape[dimension]
		remainder /= shape[dimension]
	}
}

fn vector_norm_axis_impl[T](t &vtl.Tensor[T], ord f64, axis int, keepdims bool) !&vtl.Tensor[f64] {
	if math.is_nan(ord) {
		return error('vector_norm_axis order must not be NaN')
	}
	rank := t.rank()
	if rank == 0 {
		return error('vector_norm_axis requires a tensor with at least one dimension')
	}
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('vector_norm_axis ${axis} is out of bounds for rank ${rank}')
	}
	mut output_shape := t.shape.clone()
	if keepdims {
		output_shape[axis_index] = 1
	} else {
		output_shape.delete(axis_index)
	}
	if t.shape[axis_index] == 0 {
		if ord == 0 || (ord > 0 && !math.is_inf(ord, 1)) {
			return vtl.zeros[f64](output_shape)
		}
		return error('vector_norm_axis is undefined for an empty axis and this order')
	}
	mut result := vtl.empty[f64](output_shape, memory: .row_major)
	mut slice_count := 1
	for dimension, size in t.shape {
		if dimension != axis_index {
			slice_count *= size
		}
	}
	mut index := []int{len: rank}
	for slice in 0 .. slice_count {
		decode_norm_slice(slice, t.shape, axis_index, mut index)
		value := vector_norm_slice[T](t, mut index, axis_index, ord)
		index[axis_index] = 0
		if keepdims {
			result.set(index, value)
		} else {
			mut output_index := index[..axis_index].clone()
			output_index << index[axis_index + 1..]
			result.set(output_index, value)
		}
	}
	return result
}

fn vector_norm_slice[T](t &vtl.Tensor[T], mut index []int, axis int, ord f64) f64 {
	length := if axis < 0 { t.size } else { t.shape[axis] }
	mut count_nonzero := 0
	mut has_zero := false
	mut has_nan := false
	mut scale := if ord < 0 { f64(math.inf(1)) } else { 0.0 }
	for position in 0 .. length {
		value := math.abs(f64(norm_value[T](t, mut index, axis, position)))
		if value == 0 {
			has_zero = true
			continue
		}
		count_nonzero++
		if math.is_nan(value) {
			has_nan = true
			continue
		}
		if ord < 0 {
			scale = math.min(scale, value)
		} else {
			scale = math.max(scale, value)
		}
	}
	if ord == 0 {
		return f64(count_nonzero)
	}
	if has_nan {
		return math.nan()
	}
	if math.is_inf(ord, 1) {
		return scale
	}
	if math.is_inf(ord, -1) {
		return if has_zero { 0 } else { scale }
	}
	if ord > 0 {
		if scale == 0 || math.is_inf(scale, 1) {
			return scale
		}
		mut sum := 0.0
		for position in 0 .. length {
			value := math.abs(f64(norm_value[T](t, mut index, axis, position))) / scale
			sum += math.pow(value, ord)
		}
		return scale * math.pow(sum, 1.0 / ord)
	}
	if has_zero {
		return 0
	}
	if math.is_inf(scale, 1) {
		return scale
	}
	mut sum := 0.0
	for position in 0 .. length {
		value := math.abs(f64(norm_value[T](t, mut index, axis, position)))
		sum += math.pow(scale / value, -ord)
	}
	return scale * math.pow(sum, 1.0 / ord)
}

fn norm_value[T](t &vtl.Tensor[T], mut index []int, axis int, position int) T {
	if axis < 0 {
		return t.get_nth[T](position)
	}
	index[axis] = position
	return t.get[T](index)
}

fn decode_norm_slice(line int, shape []int, axis int, mut index []int) {
	mut remainder := line
	for dimension := shape.len - 1; dimension >= 0; dimension-- {
		if dimension == axis {
			index[dimension] = 0
			continue
		}
		index[dimension] = remainder % shape[dimension]
		remainder /= shape[dimension]
	}
}
