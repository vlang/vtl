module stats

import vtl
import math

// gradient_axis computes the numerical gradient along one tensor axis using
// uniform sample spacing. Interior points use centered differences and the
// two boundaries use first-order one-sided differences. The output keeps the
// input shape and uses f64 elements.
pub fn gradient_axis[T](t &vtl.Tensor[T], spacing f64, axis int) !&vtl.Tensor[f64] {
	if t.rank() == 0 {
		return error('gradient_axis: input must have at least one dimension')
	}
	if spacing == 0.0 {
		return error('gradient_axis: spacing must not be zero')
	}
	axis_index := if axis < 0 { axis + t.rank() } else { axis }
	if axis_index < 0 || axis_index >= t.rank() {
		return error('gradient_axis: axis ${axis} out of bounds for rank ${t.rank()}')
	}
	axis_size := t.shape[axis_index]
	if axis_size < 2 {
		return error('gradient_axis: selected axis must have at least two elements')
	}

	mut output := []f64{len: t.size()}
	for flat_index in 0 .. t.size() {
		index := t.nth_index(flat_index)
		position := index[axis_index]
		if position == 0 {
			mut next_index := index.clone()
			next_index[axis_index] = 1
			output[flat_index] = (f64(t.get(next_index)) - f64(t.get(index))) / spacing
		} else if position == axis_size - 1 {
			mut previous_index := index.clone()
			previous_index[axis_index]--
			output[flat_index] = (f64(t.get(index)) - f64(t.get(previous_index))) / spacing
		} else {
			mut previous_index := index.clone()
			mut next_index := index.clone()
			previous_index[axis_index]--
			next_index[axis_index]++
			output[flat_index] = (f64(t.get(next_index)) - f64(t.get(previous_index))) / (2.0 * spacing)
		}
	}
	return vtl.from_array[f64](output, t.shape)
}

// gradient_axis_with_coordinates computes a numerical gradient along one
// tensor axis using the supplied sample coordinates. Coordinates may be
// non-uniform, but must be strictly monotonic. edge_order selects first- or
// second-order one-sided differences at the boundaries.
pub fn gradient_axis_with_coordinates[T](t &vtl.Tensor[T], coordinates []f64, axis int, edge_order int) !&vtl.Tensor[f64] {
	if t.rank() == 0 {
		return error('gradient_axis_with_coordinates: input must have at least one dimension')
	}
	axis_index := if axis < 0 { axis + t.rank() } else { axis }
	if axis_index < 0 || axis_index >= t.rank() {
		return error('gradient_axis_with_coordinates: axis ${axis} out of bounds for rank ${t.rank()}')
	}
	axis_size := t.shape[axis_index]
	if coordinates.len != axis_size {
		return error('gradient_axis_with_coordinates: got ${coordinates.len} coordinates for axis length ${axis_size}')
	}
	if edge_order != 1 && edge_order != 2 {
		return error('gradient_axis_with_coordinates: edge_order must be 1 or 2')
	}
	if axis_size < edge_order + 1 {
		return error('gradient_axis_with_coordinates: axis is too short for edge_order ${edge_order}')
	}
	for coordinate in coordinates {
		if math.is_nan(coordinate) {
			return error('gradient_axis_with_coordinates: coordinates must be strictly monotonic')
		}
	}
	increasing := coordinates[1] > coordinates[0]
	for i in 1 .. coordinates.len {
		if (increasing && coordinates[i] <= coordinates[i - 1]) || (!increasing && coordinates[i] >= coordinates[i - 1]) {
			return error('gradient_axis_with_coordinates: coordinates must be strictly monotonic')
		}
	}

	mut output := []f64{len: t.size()}
	for flat_index in 0 .. t.size() {
		index := t.nth_index(flat_index)
		position := index[axis_index]
		if position == 0 {
			h1 := coordinates[1] - coordinates[0]
			mut next_index := index.clone()
			next_index[axis_index] = 1
			f0 := f64(t.get(index))
			f1 := f64(t.get(next_index))
			if edge_order == 1 {
				output[flat_index] = (f1 - f0) / h1
			} else {
				h2 := coordinates[2] - coordinates[1]
				next_index[axis_index] = 2
				f2 := f64(t.get(next_index))
				output[flat_index] = -(2.0 * h1 + h2) / (h1 * (h1 + h2)) * f0 + (h1 + h2) / (h1 * h2) * f1 - h1 / (h2 * (h1 + h2)) * f2
			}
		} else if position == axis_size - 1 {
			h1 := coordinates[axis_size - 1] - coordinates[axis_size - 2]
			mut previous_index := index.clone()
			previous_index[axis_index] = axis_size - 2
			f_last := f64(t.get(index))
			fn1 := f64(t.get(previous_index))
			if edge_order == 1 {
				output[flat_index] = (f_last - fn1) / h1
			} else {
				h2 := coordinates[axis_size - 2] - coordinates[axis_size - 3]
				previous_index[axis_index] = axis_size - 3
				fn2 := f64(t.get(previous_index))
				output[flat_index] = h1 / (h2 * (h1 + h2)) * fn2 - (h1 + h2) / (h1 * h2) * fn1 + (2.0 * h1 + h2) / (h1 * (h1 + h2)) * f_last
			}
		} else {
			h1 := coordinates[position] - coordinates[position - 1]
			h2 := coordinates[position + 1] - coordinates[position]
			mut previous_index := index.clone()
			mut next_index := index.clone()
			previous_index[axis_index]--
			next_index[axis_index]++
			fm1 := f64(t.get(previous_index))
			f0 := f64(t.get(index))
			fp1 := f64(t.get(next_index))
			output[flat_index] = -h2 / (h1 * (h1 + h2)) * fm1 + (h2 - h1) / (h1 * h2) * f0 + h1 / (h2 * (h1 + h2)) * fp1
		}
	}
	return vtl.from_array[f64](output, t.shape)
}
