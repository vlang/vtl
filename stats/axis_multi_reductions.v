module stats

import vtl

// sum_along_axes reduces one or more axes and returns a tensor. Negative axes
// are accepted. An empty axes list returns a copy without reducing dimensions.
// When keepdims is true, each reduced dimension remains with length one.
pub fn sum_along_axes[T](t &vtl.Tensor[T], axes []int, keepdims bool) !&vtl.Tensor[T] {
	return reduce_axes[T](t, axes, keepdims, .sum)
}

// product_along_axes multiplies values along one or more axes. Negative axes
// are accepted. An empty axes list returns a copy without reducing dimensions.
pub fn product_along_axes[T](t &vtl.Tensor[T], axes []int, keepdims bool) !&vtl.Tensor[T] {
	return reduce_axes[T](t, axes, keepdims, .product)
}

fn reduce_axes[T](t &vtl.Tensor[T], axes []int, keepdims bool, operation AxisReduction) !&vtl.Tensor[T] {
	rank := t.rank()
	if axes.len == 0 {
		return t.copy(.row_major)
	}
	if rank == 0 {
		return error('multi-axis reduction requires a tensor with at least one dimension')
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
	if axes.len == rank {
		value := if operation == .sum { sum[T](t) } else { prod[T](t) }
		output_shape := if keepdims { []int{len: rank, init: 1} } else { []int{} }
		mut result := reduction_output[T](output_shape)!
		result.set_nth(0, value)
		return result
	}
	mut output_shape := []int{cap: rank}
	mut output_dimension_by_input := []int{len: rank, init: -1}
	for dimension, size in t.shape {
		if reduced[dimension] {
			if keepdims {
				output_shape << 1
				output_dimension_by_input[dimension] = output_shape.len - 1
			}
		} else {
			output_shape << size
			output_dimension_by_input[dimension] = output_shape.len - 1
		}
	}
	mut result := reduction_output[T](output_shape)!
	identity := if operation == .sum { sum_identity[T]() } else { product_identity[T]() }
	for output_index in 0 .. result.size {
		result.set_nth(output_index, identity)
	}
	mut input_index := []int{len: rank}
	for flat_index in 0 .. t.size {
		mut remainder := flat_index
		for dimension := rank - 1; dimension >= 0; dimension-- {
			dimension_size := t.shape[dimension]
			input_index[dimension] = remainder % dimension_size
			remainder /= dimension_size
		}
		mut output_flat_index := 0
		for dimension, coordinate in input_index {
			output_dimension := output_dimension_by_input[dimension]
			if output_dimension >= 0 {
				output_coordinate := if reduced[dimension] { 0 } else { coordinate }
				output_flat_index = output_flat_index * output_shape[output_dimension] + output_coordinate
			}
		}
		value := t.get(input_index)
		accumulator := result.get_nth(output_flat_index)
		if operation == .sum {
			result.set_nth(output_flat_index, accumulator + value)
		} else {
			result.set_nth(output_flat_index, accumulator * value)
		}
	}
	return result
}
