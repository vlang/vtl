module vtl

import math

// weighted_average computes a weighted mean over valid values. Weights must
// broadcast to the values shape; masked observations do not contribute either
// their value or their weight. An entirely masked array returns masked NaN.
pub fn (array &MaskedArray[T]) weighted_average[T, W](weights &Tensor[W]) !MaskedValue[f64] {
	if !masked_weights_broadcastable(weights.shape, array.values.shape) {
		return error('weighted_average: weights with shape ${weights.shape} cannot broadcast to values shape ${array.values.shape}')
	}
	broadcast_weights := weights.broadcast_to[W](array.values.shape)!
	mut weighted_sum := 0.0
	mut weight_sum := 0.0
	mut count := 0
	for i in 0 .. array.values.size {
		if array.mask.get_nth(i) {
			continue
		}
		value := td(array.values.get_nth(i)).f64()
		weight := td(broadcast_weights.get_nth(i)).f64()
		weighted_sum += value * weight
		weight_sum += weight
		count++
	}
	if count == 0 {
		return MaskedValue[f64]{
			value:     math.nan()
			is_masked: true
		}
	}
	if weight_sum == 0.0 {
		return error('weighted_average: valid weights must have a non-zero sum')
	}
	return MaskedValue[f64]{
		value:     weighted_sum / weight_sum
		is_masked: false
	}
}

// weighted_average_along_axis computes weighted means per slice. Weights may
// broadcast to the values shape or be a vector matching the selected axis.
pub fn (array &MaskedArray[T]) weighted_average_along_axis[T, W](weights &Tensor[W], axis int, keepdims bool) !MaskedArray[f64] {
	rank := array.values.rank()
	if rank == 0 {
		return error('weighted_average_along_axis: values must have at least one axis')
	}
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('weighted_average_along_axis: axis ${axis} is out of bounds for rank ${rank}')
	}
	if weights.rank() == 1 && weights.shape[0] == array.values.shape[axis_index] {
		mut expanded_shape := []int{len: rank, init: 1}
		expanded_shape[axis_index] = weights.shape[0]
		return array.weighted_average_along_axes[T, W](weights.reshape[W](expanded_shape)!, [axis_index], keepdims)
	}
	return array.weighted_average_along_axes[T, W](weights, [axis_index], keepdims)
}

// weighted_average_along_axes reduces several dimensions. Weights must
// broadcast to the values shape; empty slices are masked and zero-sum valid
// weights return an error.
pub fn (array &MaskedArray[T]) weighted_average_along_axes[T, W](weights &Tensor[W], axes []int, keepdims bool) !MaskedArray[f64] {
	if !masked_weights_broadcastable(weights.shape, array.values.shape) {
		return error('weighted_average_along_axes: weights with shape ${weights.shape} cannot broadcast to values shape ${array.values.shape}')
	}
	broadcast_weights := weights.broadcast_to[W](array.values.shape)!
	if axes.len == 0 {
		mut values := zeros[f64](array.values.shape, TensorData{})
		for flat_index in 0 .. values.size {
			if array.mask.get_nth(flat_index) {
				continue
			}
			weight := td(broadcast_weights.get_nth(flat_index)).f64()
			if weight == 0.0 {
				return error('weighted_average_along_axes: valid weights must be non-zero when axes is empty')
			}
			values.set_nth(flat_index, td(array.values.get_nth(flat_index)).f64())
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
	mut weighted_sums := []f64{len: values.size}
	mut weight_sums := []f64{len: values.size}
	mut counts := []int{len: values.size}
	mut input_index := []int{len: array.values.rank()}
	for flat_index in 0 .. array.values.size {
		if array.mask.get_nth(flat_index) {
			continue
		}
		decode_flat_coordinate(flat_index, array.values.shape, mut input_index)
		output_index := masked_axes_output_index(input_index, array.values.shape, reduced)
		value := td(array.values.get_nth(flat_index)).f64()
		weight := td(broadcast_weights.get_nth(flat_index)).f64()
		weighted_sums[output_index] += value * weight
		weight_sums[output_index] += weight
		counts[output_index]++
	}
	for i in 0 .. values.size {
		if counts[i] == 0 {
			values.set_nth(i, math.nan())
			continue
		}
		if weight_sums[i] == 0.0 {
			return error('weighted_average_along_axes: valid weights must have a non-zero sum in every slice')
		}
		values.set_nth(i, weighted_sums[i] / weight_sums[i])
		output_mask.set_nth(i, false)
	}
	return MaskedArray[f64]{
		values: values
		mask:   output_mask
	}
}

fn masked_weights_broadcastable(weights_shape []int, values_shape []int) bool {
	broadcast_shape := broadcast_shapes(weights_shape, values_shape) or { return false }
	return broadcast_shape == values_shape
}
