module vtl

import math

// sort returns a copy sorted along the last axis, matching NumPy's default.
pub fn sort[T](t &Tensor[T]) !&Tensor[T] {
	return sort_axis[T](t, -1)
}

// sort_axis returns a copy sorted in ascending order along axis. Floating-point
// NaNs sort after all non-NaN values, and equal values keep their input order.
pub fn sort_axis[T](t &Tensor[T], axis int) !&Tensor[T] {
	axis_index := normalize_sort_axis(t, axis)!
	mut result := empty[T](t.shape, memory: t.memory)
	axis_size := t.shape[axis_index]
	if t.size == 0 || axis_size == 0 {
		return result
	}
	line_count := t.size / axis_size
	mut index := []int{len: t.rank()}
	mut values := []T{len: axis_size}
	for line in 0 .. line_count {
		decode_sort_line(line, t.shape, axis_index, mut index)
		for position in 0 .. axis_size {
			index[axis_index] = position
			values[position] = t.get(index)
		}
		values.sort_with_compare(fn [T](a &T, b &T) int {
			return compare_sort_values[T](*a, *b)
		})
		for position, value in values {
			index[axis_index] = position
			result.set(index, value)
		}
	}
	return result
}

// argsort returns stable ascending sort indices along the last axis.
pub fn argsort[T](t &Tensor[T]) !&Tensor[int] {
	return argsort_axis[T](t, -1)
}

// argsort_axis returns the stable ascending sort indices along axis. Returned
// indices are local to that axis; floating-point NaNs sort to the end.
pub fn argsort_axis[T](t &Tensor[T], axis int) !&Tensor[int] {
	axis_index := normalize_sort_axis(t, axis)!
	mut result := empty[int](t.shape, memory: t.memory)
	axis_size := t.shape[axis_index]
	if t.size == 0 || axis_size == 0 {
		return result
	}
	line_count := t.size / axis_size
	mut index := []int{len: t.rank()}
	mut values := []T{len: axis_size}
	mut positions := []int{len: axis_size}
	for line in 0 .. line_count {
		decode_sort_line(line, t.shape, axis_index, mut index)
		for position in 0 .. axis_size {
			index[axis_index] = position
			values[position] = t.get(index)
			positions[position] = position
		}
		positions.sort_with_compare(fn [values] (a &int, b &int) int {
			return compare_sort_values(values[*a], values[*b])
		})
		for position, value in positions {
			index[axis_index] = position
			result.set(index, value)
		}
	}
	return result
}

fn normalize_sort_axis[T](t &Tensor[T], axis int) !int {
	rank := t.rank()
	if rank == 0 {
		return error('sort: tensor must have at least one dimension')
	}
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('sort: axis ${axis} out of bounds for tensor with ${rank} dimensions')
	}
	return axis_index
}

fn decode_sort_line(line int, shape []int, axis int, mut index []int) {
	mut remainder := line
	for dim := shape.len - 1; dim >= 0; dim-- {
		if dim == axis {
			index[dim] = 0
			continue
		}
		index[dim] = remainder % shape[dim]
		remainder /= shape[dim]
	}
}

fn compare_sort_values[T](a T, b T) int {
	$if T is f32 || T is f64 {
		a_nan := math.is_nan(f64(a))
		b_nan := math.is_nan(f64(b))
		if a_nan {
			return if b_nan { 0 } else { 1 }
		}
		if b_nan {
			return -1
		}
	}
	if a < b {
		return -1
	}
	if a > b {
		return 1
	}
	return 0
}
