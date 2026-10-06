module stats

import vtl

enum AxisReduction {
	sum
	product
}

// sum_along_axis reduces one axis and returns a tensor. When keepdims is true,
// the reduced axis remains in the result with length one.
pub fn sum_along_axis[T](t &vtl.Tensor[T], axis int, keepdims bool) !&vtl.Tensor[T] {
	return reduce_axis[T](t, axis, keepdims, .sum)
}

// product_along_axis multiplies values along one axis. When keepdims is true,
// the reduced axis remains in the result with length one.
pub fn product_along_axis[T](t &vtl.Tensor[T], axis int, keepdims bool) !&vtl.Tensor[T] {
	return reduce_axis[T](t, axis, keepdims, .product)
}

fn reduce_axis[T](t &vtl.Tensor[T], axis int, keepdims bool, operation AxisReduction) !&vtl.Tensor[T] {
	rank := t.rank()
	if rank == 0 {
		return error('axis reduction requires a tensor with at least one dimension')
	}
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('axis ${axis} out of bounds for rank ${rank}')
	}
	mut output_shape := t.shape.clone()
	if keepdims {
		output_shape[axis_index] = 1
	} else {
		output_shape.delete(axis_index)
	}
	mut result := vtl.empty[T](output_shape, memory: .row_major)
	mut slice_count := 1
	for dimension, size in t.shape {
		if dimension != axis_index {
			slice_count *= size
		}
	}
	mut index := []int{len: rank}
	for slice in 0 .. slice_count {
		decode_reduction_slice(slice, t.shape, axis_index, mut index)
		mut reduced := if operation == .sum { vtl.cast[T](0) } else { vtl.cast[T](1) }
		for position in 0 .. t.shape[axis_index] {
			index[axis_index] = position
			value := t.get(index)
			if operation == .sum {
				reduced += value
			} else {
				reduced *= value
			}
		}
		if keepdims {
			index[axis_index] = 0
			result.set(index, reduced)
		} else {
			mut output_index := index[..axis_index].clone()
			output_index << index[axis_index + 1..]
			result.set(output_index, reduced)
		}
	}
	return result
}

fn decode_reduction_slice(line int, shape []int, axis int, mut index []int) {
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
