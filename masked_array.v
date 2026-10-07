module vtl

import math

// MaskedArray pairs tensor values with a boolean missing-data mask. A true
// mask entry marks the corresponding value as missing, following NumPy's
// numpy.ma convention.
pub struct MaskedArray[T] {
pub:
	values &Tensor[T]
	mask   &Tensor[bool]
}

// MaskedValue carries a reduction result and indicates whether it is masked.
pub struct MaskedValue[T] {
pub:
	value     T
	is_masked bool
}

// masked_array creates a masked tensor. The mask broadcasts against values;
// the returned values and mask are views with the common broadcast shape.
pub fn masked_array[T](values &Tensor[T], mask &Tensor[bool]) !MaskedArray[T] {
	rank := math.max(values.rank(), mask.rank())
	mut values_shape := []int{len: rank, init: 1}
	mut mask_shape := []int{len: rank, init: 1}
	for i, size in values.shape {
		values_shape[rank - values.rank() + i] = size
	}
	for i, size in mask.shape {
		mask_shape[rank - mask.rank() + i] = size
	}
	if !broadcast_equal(values_shape, mask_shape) {
		return error('masked_array: values with shape ${values.shape} and mask with shape ${mask.shape} cannot broadcast')
	}
	mut shape := []int{len: rank}
	for i in 0 .. rank {
		if values_shape[i] == 1 {
			shape[i] = mask_shape[i]
		} else {
			shape[i] = values_shape[i]
		}
	}
	return MaskedArray[T]{
		values: values.broadcast_to[T](shape)!
		mask:   mask.broadcast_to[bool](shape)!
	}
}

// filled returns a row-major copy with masked entries replaced by fill_value.
pub fn (array &MaskedArray[T]) filled[T](fill_value T) &Tensor[T] {
	mut result := empty[T](array.values.shape, memory: .row_major)
	for i in 0 .. result.size {
		value := if array.mask.get_nth(i) { fill_value } else { array.values.get_nth(i) }
		result.set_nth(i, value)
	}
	return result
}

// compressed returns all unmasked values as a one-dimensional copy in
// row-major logical order.
pub fn (array &MaskedArray[T]) compressed[T]() !&Tensor[T] {
	mut visible := empty[bool](array.mask.shape, memory: .row_major)
	for i in 0 .. visible.size {
		visible.set_nth(i, !array.mask.get_nth(i))
	}
	return array.values.masked_select(visible)!
}

// count returns the number of unmasked values.
pub fn (array &MaskedArray[T]) count[T]() int {
	mut valid := 0
	for i in 0 .. array.mask.size {
		if !array.mask.get_nth(i) {
			valid++
		}
	}
	return valid
}

// sum reduces all unmasked values. An entirely masked array returns a masked
// additive identity with value zero.
pub fn (array &MaskedArray[T]) sum[T]() MaskedValue[T] {
	mut total := cast[T](0)
	mut count := 0
	for i in 0 .. array.values.size {
		if !array.mask.get_nth(i) {
			total += array.values.get_nth(i)
			count++
		}
	}
	return MaskedValue[T]{
		value:     total
		is_masked: count == 0
	}
}

// mean reduces all unmasked values. An entirely masked array returns a masked
// NaN result.
pub fn (array &MaskedArray[T]) mean[T]() MaskedValue[f64] {
	mut total := 0.0
	mut count := 0
	for i in 0 .. array.values.size {
		if !array.mask.get_nth(i) {
			total += td(array.values.get_nth(i)).f64()
			count++
		}
	}
	return MaskedValue[f64]{
		value:     if count == 0 { math.nan() } else { total / f64(count) }
		is_masked: count == 0
	}
}

// sum_along_axis reduces unmasked values along one axis. Output mask entries
// are true for slices containing no unmasked values.
pub fn (array &MaskedArray[T]) sum_along_axis[T](axis int, keepdims bool) !MaskedArray[T] {
	return array.sum_along_axes[T]([axis], keepdims)
}

// sum_along_axes reduces unmasked values over several axes. Negative axes are
// supported; duplicate axes are rejected. An empty axes list preserves data.
pub fn (array &MaskedArray[T]) sum_along_axes[T](axes []int, keepdims bool) !MaskedArray[T] {
	if axes.len == 0 {
		return *array
	}
	reduced := normalize_masked_axes(axes, array.values.rank())!
	output_shape := masked_reduction_shape(array.values.shape, reduced, keepdims)
	mut values := zeros[T](output_shape, TensorData{})
	mut output_mask := ones[bool](output_shape, TensorData{})
	mut counts := []int{len: values.size}
	mut input_index := []int{len: array.values.rank()}
	for flat_index in 0 .. array.values.size {
		decode_flat_coordinate(flat_index, array.values.shape, mut input_index)
		if array.mask.get_nth(flat_index) {
			continue
		}
		output_flat_index := masked_axes_output_index(input_index, array.values.shape, reduced)
		values.set_nth(output_flat_index, values.get_nth(output_flat_index) + array.values.get_nth(flat_index))
		counts[output_flat_index]++
	}
	for i in 0 .. values.size {
		output_mask.set_nth(i, counts[i] == 0)
	}
	return MaskedArray[T]{
		values: values
		mask:   output_mask
	}
}

// mean_along_axis computes the mean of unmasked values per axis slice. Slices
// with no unmasked values carry a masked NaN result.
pub fn (array &MaskedArray[T]) mean_along_axis[T](axis int, keepdims bool) !MaskedArray[f64] {
	return array.mean_along_axes[T]([axis], keepdims)
}

// mean_along_axes computes means over several axes. Negative axes are
// supported; duplicate axes are rejected. An empty axes list converts each
// unmasked value to f64 and preserves the mask.
pub fn (array &MaskedArray[T]) mean_along_axes[T](axes []int, keepdims bool) !MaskedArray[f64] {
	if axes.len == 0 {
		mut values := empty[f64](array.values.shape, memory: .row_major)
		for i in 0 .. array.values.size {
			values.set_nth(i, td(array.values.get_nth(i)).f64())
		}
		return MaskedArray[f64]{
			values: values
			mask:   array.mask
		}
	}
	reduced := normalize_masked_axes(axes, array.values.rank())!
	output_shape := masked_reduction_shape(array.values.shape, reduced, keepdims)
	mut values := zeros[f64](output_shape, TensorData{})
	mut output_mask := ones[bool](output_shape, TensorData{})
	mut totals := []f64{len: values.size}
	mut counts := []int{len: values.size}
	mut input_index := []int{len: array.values.rank()}
	for flat_index in 0 .. array.values.size {
		decode_flat_coordinate(flat_index, array.values.shape, mut input_index)
		if array.mask.get_nth(flat_index) {
			continue
		}
		output_flat_index := masked_axes_output_index(input_index, array.values.shape, reduced)
		totals[output_flat_index] += td(array.values.get_nth(flat_index)).f64()
		counts[output_flat_index]++
	}
	for i in 0 .. values.size {
		if counts[i] > 0 {
			values.set_nth(i, totals[i] / f64(counts[i]))
			output_mask.set_nth(i, false)
		} else {
			values.set_nth(i, math.nan())
		}
	}
	return MaskedArray[f64]{
		values: values
		mask:   output_mask
	}
}

fn normalize_masked_axes(axes []int, rank int) ![]bool {
	if rank == 0 {
		return error('masked axis reduction requires at least one dimension')
	}
	mut reduced := []bool{len: rank}
	for axis in axes {
		axis_index := if axis < 0 { axis + rank } else { axis }
		if axis_index < 0 || axis_index >= rank {
			return error('axis ${axis} out of bounds for rank ${rank}')
		}
		if reduced[axis_index] {
			return error('axis ${axis} appears more than once')
		}
		reduced[axis_index] = true
	}
	return reduced
}

fn masked_reduction_shape(input_shape []int, reduced []bool, keepdims bool) []int {
	mut output_shape := []int{cap: input_shape.len}
	for dimension, size in input_shape {
		if reduced[dimension] {
			if keepdims {
				output_shape << 1
			}
		} else {
			output_shape << size
		}
	}
	return output_shape
}

fn masked_axes_output_index(input_index []int, input_shape []int, reduced []bool) int {
	mut output_flat_index := 0
	for dimension, coordinate in input_index {
		if !reduced[dimension] {
			output_flat_index = output_flat_index * input_shape[dimension] + coordinate
		}
	}
	return output_flat_index
}
