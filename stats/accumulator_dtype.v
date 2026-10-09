module stats

import vtl

// sum_as reduces all values into an explicitly selected accumulator type.
pub fn sum_as[T, U](t &vtl.Tensor[T]) U {
	mut result := vtl.cast[U](0)
	for flat_index in 0 .. t.size {
		result += vtl.cast[U](t.get_nth(flat_index))
	}
	return result
}

// product_as reduces all values into an explicitly selected accumulator type.
pub fn product_as[T, U](t &vtl.Tensor[T]) U {
	mut result := vtl.cast[U](1)
	for flat_index in 0 .. t.size {
		result *= vtl.cast[U](t.get_nth(flat_index))
	}
	return result
}

// sum_along_axis_as reduces one axis using the requested accumulator type.
pub fn sum_along_axis_as[T, U](t &vtl.Tensor[T], axis int, keepdims bool) !&vtl.Tensor[U] {
	return reduce_axes_as[T, U](t, [axis], keepdims, false)
}

// product_along_axis_as reduces one axis using the requested accumulator type.
pub fn product_along_axis_as[T, U](t &vtl.Tensor[T], axis int, keepdims bool) !&vtl.Tensor[U] {
	return reduce_axes_as[T, U](t, [axis], keepdims, true)
}

// sum_along_axes_as reduces several axes using the requested accumulator type.
pub fn sum_along_axes_as[T, U](t &vtl.Tensor[T], axes []int, keepdims bool) !&vtl.Tensor[U] {
	return reduce_axes_as[T, U](t, axes, keepdims, false)
}

// product_along_axes_as reduces several axes using the requested accumulator type.
pub fn product_along_axes_as[T, U](t &vtl.Tensor[T], axes []int, keepdims bool) !&vtl.Tensor[U] {
	return reduce_axes_as[T, U](t, axes, keepdims, true)
}

fn reduce_axes_as[T, U](t &vtl.Tensor[T], axes []int, keepdims bool, product bool) !&vtl.Tensor[U] {
	rank := t.rank()
	if axes.len > 0 && rank == 0 {
		return error('axis reduction requires a tensor with at least one dimension')
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
	mut output_shape := []int{cap: rank}
	mut output_dimension := []int{len: rank, init: -1}
	for dimension, size in t.shape {
		if reduced[dimension] {
			if keepdims {
				output_shape << 1
				output_dimension[dimension] = output_shape.len - 1
			}
		} else {
			output_shape << size
			output_dimension[dimension] = output_shape.len - 1
		}
	}
	mut result := vtl.empty[U](output_shape, memory: .row_major)
	identity := vtl.cast[U](if product { 1 } else { 0 })
	for flat_index in 0 .. result.size {
		result.set_nth(flat_index, identity)
	}
	mut input_index := []int{len: rank}
	for flat_index in 0 .. t.size {
		mut remainder := flat_index
		for dimension := rank - 1; dimension >= 0; dimension-- {
			input_index[dimension] = remainder % t.shape[dimension]
			remainder /= t.shape[dimension]
		}
		mut result_index := 0
		for dimension, coordinate in input_index {
			mapped_dimension := output_dimension[dimension]
			if mapped_dimension >= 0 {
				mapped_coordinate := if reduced[dimension] { 0 } else { coordinate }
				result_index = result_index * output_shape[mapped_dimension] + mapped_coordinate
			}
		}
		value := vtl.cast[U](t.get(input_index))
		current := result.get_nth(result_index)
		result.set_nth(result_index, if product { current * value } else { current + value })
	}
	return result
}
