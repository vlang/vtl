module stats

import vtl

// average returns the weighted mean of tensors with identical shapes.
pub fn average[T, W](values &vtl.Tensor[T], weights &vtl.Tensor[W]) !f64 {
	if values.shape != weights.shape {
		return error('average: values and weights must have identical shapes')
	}
	if values.size == 0 {
		return error('average: cannot average an empty tensor')
	}
	mut weighted_sum := 0.0
	mut weight_sum := 0.0
	for i in 0 .. values.size {
		value := vtl.cast[f64](values.get_nth[T](i))
		weight := vtl.cast[f64](weights.get_nth[W](i))
		weighted_sum += value * weight
		weight_sum += weight
	}
	if weight_sum == 0.0 {
		return error('average: weights must have a non-zero sum')
	}
	return weighted_sum / weight_sum
}

// average_axis computes weighted means per slice and retains the reduced axis
// with length one. Weights may match the input shape or be a vector matching
// the selected axis length.
pub fn average_axis[T, W](values &vtl.Tensor[T], weights &vtl.Tensor[W], axis int) !&vtl.Tensor[f64] {
	rank := values.rank()
	if rank == 0 {
		return error('average_axis: values must have at least one axis')
	}
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('average_axis: axis ${axis} is out of bounds for rank ${rank}')
	}
	weights_by_axis := weights.rank() == 1 && weights.size == values.shape[axis_index]
	if weights.shape != values.shape && !weights_by_axis {
		return error('average_axis: weights must match the input shape or selected axis length')
	}
	axis_size := values.shape[axis_index]
	if axis_size == 0 {
		return error('average_axis: cannot average an empty axis')
	}
	mut output_shape := values.shape.clone()
	output_shape[axis_index] = 1
	mut output := vtl.empty[f64](output_shape, memory: .row_major)
	for flat_index in 0 .. output.size {
		output_index := output.nth_index(flat_index)
		mut weighted_sum := 0.0
		mut weight_sum := 0.0
		for axis_value in 0 .. axis_size {
			mut input_index := output_index.clone()
			input_index[axis_index] = axis_value
			value := vtl.cast[f64](values.get[T](input_index))
			weight := if weights_by_axis {
				vtl.cast[f64](weights.get_nth[W](axis_value))
			} else {
				vtl.cast[f64](weights.get[W](input_index))
			}
			weighted_sum += value * weight
			weight_sum += weight
		}
		if weight_sum == 0.0 {
			return error('average_axis: weights must have a non-zero sum in every slice')
		}
		output.set(output_index, weighted_sum / weight_sum)
	}
	return output
}

// average_along_axis computes weighted means along one axis. It preserves the
// reduced axis when keepdims is true and removes it otherwise.
pub fn average_along_axis[T, W](values &vtl.Tensor[T], weights &vtl.Tensor[W], axis int, keepdims bool) !&vtl.Tensor[f64] {
	result := average_axis[T, W](values, weights, axis)!
	if keepdims {
		return result
	}
	rank := result.rank()
	axis_index := if axis < 0 { axis + rank } else { axis }
	mut shape := []int{cap: rank - 1}
	for dimension, size in result.shape {
		if dimension != axis_index {
			shape << size
		}
	}
	return result.reshape[f64](shape)
}

// average_along_axes computes weighted means over multiple axes. Weights must
// have the same shape as values. Reduced dimensions follow keepdims.
pub fn average_along_axes[T, W](values &vtl.Tensor[T], weights &vtl.Tensor[W], axes []int, keepdims bool) !&vtl.Tensor[f64] {
	if values.shape != weights.shape {
		return error('average_along_axes: weights must match the input shape')
	}
	rank := values.rank()
	if rank == 0 && axes.len > 0 {
		return error('average_along_axes: values must have at least one axis')
	}
	mut reduced := []bool{len: rank}
	for axis in axes {
		index := if axis < 0 { axis + rank } else { axis }
		if index < 0 || index >= rank {
			return error('average_along_axes: axis ${axis} is out of bounds for rank ${rank}')
		}
		if reduced[index] {
			return error('average_along_axes: axis ${axis} appears more than once')
		}
		if values.shape[index] == 0 {
			return error('average_along_axes: cannot average an empty axis')
		}
		reduced[index] = true
	}
	mut output_shape := []int{cap: rank}
	for dimension, size in values.shape {
		if reduced[dimension] {
			if keepdims {
				output_shape << 1
			}
		} else {
			output_shape << size
		}
	}
	mut output := vtl.empty[f64](output_shape, memory: .row_major)
	mut weighted_sums := []f64{len: output.size}
	mut weight_sums := []f64{len: output.size}
	for flat in 0 .. values.size {
		output_index := weighted_average_axes_index(flat, values.shape, reduced)
		value := vtl.cast[f64](values.get_nth[T](flat))
		weight := vtl.cast[f64](weights.get_nth[W](flat))
		weighted_sums[output_index] += value * weight
		weight_sums[output_index] += weight
	}
	for flat in 0 .. output.size {
		if weight_sums[flat] == 0.0 {
			return error('average_along_axes: weights must have a non-zero sum in every slice')
		}
		output.set_nth(flat, weighted_sums[flat] / weight_sums[flat])
	}
	return output
}

fn weighted_average_axes_index(flat int, shape []int, reduced []bool) int {
	mut remainder := flat
	mut output_index := 0
	mut output_stride := 1
	for dimension := shape.len - 1; dimension >= 0; dimension-- {
		coordinate := remainder % shape[dimension]
		remainder /= shape[dimension]
		if !reduced[dimension] {
			output_index += coordinate * output_stride
			output_stride *= shape[dimension]
		}
	}
	return output_index
}
