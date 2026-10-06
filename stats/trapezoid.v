module stats

import vtl

// trapezoid integrates a one-dimensional or N-dimensional tensor along its
// last axis with uniform sample spacing `dx`. The reduced result is f64.
pub fn trapezoid[T](t &vtl.Tensor[T], dx f64) !&vtl.Tensor[f64] {
	return trapezoid_axis[T](t, dx, -1)
}

// trapezoid_axis integrates along `axis` using uniform sample spacing `dx`.
// The reduced axis is removed from the result shape.
pub fn trapezoid_axis[T](t &vtl.Tensor[T], dx f64, axis int) !&vtl.Tensor[f64] {
	return trapezoid_impl[T](t, []f64{}, false, dx, axis)
}

// trapezoid_x_axis integrates along `axis` using the provided one-dimensional
// sample coordinates. Coordinates are consumed in their original order; they
// are not sorted.
pub fn trapezoid_x_axis[T, X](t &vtl.Tensor[T], x &vtl.Tensor[X], axis int) !&vtl.Tensor[f64] {
	if x.rank() != 1 {
		return error('trapezoid_x_axis: x must be one-dimensional')
	}
	axis_index := normalize_trapezoid_axis(t.rank(), axis)!
	if x.size() != t.shape[axis_index] {
		return error('trapezoid_x_axis: x length ${x.size()} does not match axis length ${t.shape[axis_index]}')
	}
	mut x_steps := []f64{len: if x.size() > 0 { x.size() - 1 } else { 0 }}
	for i in 0 .. x_steps.len {
		x_steps[i] = f64(x.get_nth(i + 1)) - f64(x.get_nth(i))
	}
	return trapezoid_impl[T](t, x_steps, true, 1.0, axis_index)
}

fn trapezoid_impl[T](t &vtl.Tensor[T], x_steps []f64, has_x bool, dx f64, axis int) !&vtl.Tensor[f64] {
	if t.rank() == 0 {
		return error('trapezoid: input must have at least one dimension')
	}
	axis_index := normalize_trapezoid_axis(t.rank(), axis)!
	mut output_shape := t.shape.clone()
	axis_size := output_shape[axis_index]
	output_shape.delete(axis_index)
	mut output_size := 1
	for dimension in output_shape {
		output_size *= dimension
	}
	mut output := []f64{len: output_size}
	mut output_index := []int{len: output_shape.len}
	mut input_index := []int{len: t.rank()}
	for flat_index in 0 .. output_size {
		mut remainder := flat_index
		for dimension := output_shape.len - 1; dimension >= 0; dimension-- {
			output_index[dimension] = remainder % output_shape[dimension]
			remainder /= output_shape[dimension]
		}
		mut output_axis := 0
		for input_axis in 0 .. t.rank() {
			if input_axis != axis_index {
				input_index[input_axis] = output_index[output_axis]
				output_axis++
			}
		}
		mut integral := 0.0
		for segment in 0 .. axis_size - 1 {
			input_index[axis_index] = segment
			y0 := f64(t.get(input_index))
			input_index[axis_index] = segment + 1
			y1 := f64(t.get(input_index))
			width := if has_x { x_steps[segment] } else { dx }
			integral += (y0 + y1) * 0.5 * width
		}
		output[flat_index] = integral
	}
	return vtl.from_array[f64](output, output_shape)
}

fn normalize_trapezoid_axis(rank int, axis int) !int {
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('trapezoid: axis ${axis} out of bounds for rank ${rank}')
	}
	return axis_index
}
