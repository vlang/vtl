module stats

import vtl

// average returns the weighted mean of tensors with identical shapes.
pub fn average[T, W](values &vtl.Tensor[T], weights &vtl.Tensor[W]) !f64 {
	return average_as[T, W, f64](values, weights)
}

// average_as computes a weighted mean using U for accumulation and output.
pub fn average_as[T, W, U](values &vtl.Tensor[T], weights &vtl.Tensor[W]) !U {
	if values.shape != weights.shape {
		return error('average: values and weights must have identical shapes')
	}
	if values.size == 0 {
		return error('average: cannot average an empty tensor')
	}
	mut weighted_sum := vtl.cast[U](0)
	mut weight_sum := vtl.cast[U](0)
	for i in 0 .. values.size {
		value := vtl.cast[U](values.get_nth[T](i))
		weight := vtl.cast[U](weights.get_nth[W](i))
		weighted_sum += value * weight
		weight_sum += weight
	}
	if weight_sum == vtl.cast[U](0) {
		return error('average: weights must have a non-zero sum')
	}
	return weighted_sum / weight_sum
}

// average_axis computes weighted means per slice and retains the reduced axis
// with length one. Weights may match the input shape or be a vector matching
// the selected axis length.
pub fn average_axis[T, W](values &vtl.Tensor[T], weights &vtl.Tensor[W], axis int) !&vtl.Tensor[f64] {
	return average_axis_as[T, W, f64](values, weights, axis)
}

// average_axis_as computes weighted means along one axis using U and retains
// the reduced axis with length one.
pub fn average_axis_as[T, W, U](values &vtl.Tensor[T], weights &vtl.Tensor[W], axis int) !&vtl.Tensor[U] {
	return average_along_axis_as[T, W, U](values, weights, axis, true)
}

// average_along_axis_as computes weighted means along one axis using U.
// Weights may match the input shape or be a vector matching the selected axis.
pub fn average_along_axis_as[T, W, U](values &vtl.Tensor[T], weights &vtl.Tensor[W], axis int, keepdims bool) !&vtl.Tensor[U] {
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
	mut output_shape := []int{cap: rank}
	for dimension, size in values.shape {
		if dimension == axis_index {
			if keepdims {
				output_shape << 1
			}
		} else {
			output_shape << size
		}
	}
	mut output := vtl.empty[U](output_shape, memory: .row_major)
	for flat_index in 0 .. output.size {
		output_index := output.nth_index(flat_index)
		mut weighted_sum := vtl.cast[U](0)
		mut weight_sum := vtl.cast[U](0)
		mut input_index := []int{len: rank}
		mut output_dimension := 0
		for dimension in 0 .. rank {
			if dimension != axis_index {
				input_index[dimension] = if keepdims {
					output_index[dimension]
				} else {
					output_index[output_dimension]
				}
				if !keepdims {
					output_dimension++
				}
			}
		}
		for axis_value in 0 .. axis_size {
			input_index[axis_index] = axis_value
			value := vtl.cast[U](values.get[T](input_index))
			weight := if weights_by_axis {
				vtl.cast[U](weights.get_nth[W](axis_value))
			} else {
				vtl.cast[U](weights.get[W](input_index))
			}
			weighted_sum += value * weight
			weight_sum += weight
		}
		if weight_sum == vtl.cast[U](0) {
			return error('average_axis: weights must have a non-zero sum in every slice')
		}
		output.set(output_index, weighted_sum / weight_sum)
	}
	return output
}

// average_along_axis computes weighted means along one axis. It preserves the
// reduced axis when keepdims is true and removes it otherwise.
pub fn average_along_axis[T, W](values &vtl.Tensor[T], weights &vtl.Tensor[W], axis int, keepdims bool) !&vtl.Tensor[f64] {
	return average_along_axis_as[T, W, f64](values, weights, axis, keepdims)
}

// average_along_axes computes weighted means over multiple axes. Weights must
// have the same shape as values. Reduced dimensions follow keepdims.
pub fn average_along_axes[T, W](values &vtl.Tensor[T], weights &vtl.Tensor[W], axes []int, keepdims bool) !&vtl.Tensor[f64] {
	return average_along_axes_as[T, W, f64](values, weights, axes, keepdims)
}

// average_along_axes_as computes weighted means over multiple axes using U.
// Weights must have the same shape as values; reduced dimensions follow keepdims.
pub fn average_along_axes_as[T, W, U](values &vtl.Tensor[T], weights &vtl.Tensor[W], axes []int, keepdims bool) !&vtl.Tensor[U] {
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
	mut output := vtl.empty[U](output_shape, memory: .row_major)
	mut weighted_sums := []U{len: output.size}
	mut weight_sums := []U{len: output.size}
	for flat in 0 .. values.size {
		output_index := weighted_average_axes_index(flat, values.shape, reduced)
		value := vtl.cast[U](values.get_nth[T](flat))
		weight := vtl.cast[U](weights.get_nth[W](flat))
		weighted_sums[output_index] += value * weight
		weight_sums[output_index] += weight
	}
	for flat in 0 .. output.size {
		if weight_sums[flat] == vtl.cast[U](0) {
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
