module stats

import math
import vtl

enum NanReduction {
	sum
	product
	minimum
	maximum
}

// nansum adds all non-NaN elements and returns 0 for empty or all-NaN input.
pub fn nansum[T](t &vtl.Tensor[T]) f64 {
	return nan_reduce_scalar[T](t, .sum)
}

// nanprod multiplies all non-NaN elements and returns 1 for empty or all-NaN input.
pub fn nanprod[T](t &vtl.Tensor[T]) f64 {
	return nan_reduce_scalar[T](t, .product)
}

// nanmin returns the minimum non-NaN element, or NaN if none exists.
pub fn nanmin[T](t &vtl.Tensor[T]) f64 {
	return nan_reduce_scalar[T](t, .minimum)
}

// nanmax returns the maximum non-NaN element, or NaN if none exists.
pub fn nanmax[T](t &vtl.Tensor[T]) f64 {
	return nan_reduce_scalar[T](t, .maximum)
}

// nansum_axis reduces one axis and removes it from the result shape.
pub fn nansum_axis[T](t &vtl.Tensor[T], axis int) !&vtl.Tensor[f64] {
	return nan_reduce_axis[T](t, axis, .sum, false)
}

// nansum_axis_keepdims reduces one axis and keeps it with length one.
pub fn nansum_axis_keepdims[T](t &vtl.Tensor[T], axis int) !&vtl.Tensor[f64] {
	return nan_reduce_axis[T](t, axis, .sum, true)
}

// nanprod_axis reduces one axis and removes it from the result shape.
pub fn nanprod_axis[T](t &vtl.Tensor[T], axis int) !&vtl.Tensor[f64] {
	return nan_reduce_axis[T](t, axis, .product, false)
}

// nanprod_axis_keepdims reduces one axis and keeps it with length one.
pub fn nanprod_axis_keepdims[T](t &vtl.Tensor[T], axis int) !&vtl.Tensor[f64] {
	return nan_reduce_axis[T](t, axis, .product, true)
}

// nanmin_axis reduces one axis and removes it from the result shape. Slices
// containing no non-NaN values produce NaN.
pub fn nanmin_axis[T](t &vtl.Tensor[T], axis int) !&vtl.Tensor[f64] {
	return nan_reduce_axis[T](t, axis, .minimum, false)
}

// nanmin_axis_keepdims reduces one axis and keeps it with length one.
pub fn nanmin_axis_keepdims[T](t &vtl.Tensor[T], axis int) !&vtl.Tensor[f64] {
	return nan_reduce_axis[T](t, axis, .minimum, true)
}

// nanmax_axis reduces one axis and removes it from the result shape. Slices
// containing no non-NaN values produce NaN.
pub fn nanmax_axis[T](t &vtl.Tensor[T], axis int) !&vtl.Tensor[f64] {
	return nan_reduce_axis[T](t, axis, .maximum, false)
}

// nanmax_axis_keepdims reduces one axis and keeps it with length one.
pub fn nanmax_axis_keepdims[T](t &vtl.Tensor[T], axis int) !&vtl.Tensor[f64] {
	return nan_reduce_axis[T](t, axis, .maximum, true)
}

fn nan_reduce_scalar[T](t &vtl.Tensor[T], operation NanReduction) f64 {
	mut has_value := false
	mut result := match operation {
		.sum { 0.0 }
		.product { 1.0 }
		.minimum { math.inf(1) }
		.maximum { math.inf(-1) }
	}
	mut iter := t.iterator[T]()
	for {
		value, _ := iter.next() or { break }
		current := f64(value)
		if math.is_nan(current) {
			continue
		}
		has_value = true
		result = apply_nan_reduction(result, current, operation)
	}
	if !has_value && (operation == .minimum || operation == .maximum) {
		return math.nan()
	}
	return result
}

fn nan_reduce_axis[T](t &vtl.Tensor[T], axis int, operation NanReduction, keepdims bool) !&vtl.Tensor[f64] {
	rank := t.rank()
	if rank == 0 {
		return error('NaN reduction axis requires a tensor with at least one dimension')
	}
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('NaN reduction axis ${axis} out of bounds for rank ${rank}')
	}
	mut output_shape := t.shape.clone()
	if keepdims {
		output_shape[axis_index] = 1
	} else {
		output_shape.delete(axis_index)
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
		decode_nan_reduction_slice(slice, t.shape, axis_index, mut index)
		mut has_value := false
		mut reduced := match operation {
			.sum { 0.0 }
			.product { 1.0 }
			.minimum { math.inf(1) }
			.maximum { math.inf(-1) }
		}
		for position in 0 .. t.shape[axis_index] {
			index[axis_index] = position
			value := f64(t.get(index))
			if math.is_nan(value) {
				continue
			}
			has_value = true
			reduced = apply_nan_reduction(reduced, value, operation)
		}
		if !has_value && (operation == .minimum || operation == .maximum) {
			reduced = math.nan()
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

fn apply_nan_reduction(accumulator f64, value f64, operation NanReduction) f64 {
	return match operation {
		.sum { accumulator + value }
		.product { accumulator * value }
		.minimum { math.min(accumulator, value) }
		.maximum { math.max(accumulator, value) }
	}
}

fn decode_nan_reduction_slice(line int, shape []int, axis int, mut index []int) {
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
