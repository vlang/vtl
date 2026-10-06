module stats

import vtl

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
