module stats

import vtl

// trapezoid integrates a one-dimensional or N-dimensional tensor along its
// last axis with uniform sample spacing `dx`. The reduced result is f64.
pub fn trapezoid[T](t &vtl.Tensor[T], dx f64) !&vtl.Tensor[f64] {
	return trapezoid_as[T, f64](t, dx)
}

// trapezoid_as integrates over the last axis using U for accumulation/output.
pub fn trapezoid_as[T, U](t &vtl.Tensor[T], dx f64) !&vtl.Tensor[U] {
	return trapezoid_axis_as[T, U](t, dx, -1)
}

// trapezoid_axis integrates along `axis` using uniform sample spacing `dx`.
// The reduced axis is removed from the result shape.
pub fn trapezoid_axis[T](t &vtl.Tensor[T], dx f64, axis int) !&vtl.Tensor[f64] {
	return trapezoid_axis_as[T, f64](t, dx, axis)
}

// trapezoid_axis_as integrates along `axis` using U for accumulation/output.
pub fn trapezoid_axis_as[T, U](t &vtl.Tensor[T], dx f64, axis int) !&vtl.Tensor[U] {
	validate_trapezoid_type[U]()!
	return trapezoid_impl[T, U](t, []U{}, false, vtl.cast[U](dx), axis)
}

// trapezoid_x_axis integrates along `axis` using the provided one-dimensional
// sample coordinates. Coordinates are consumed in their original order; they
// are not sorted.
pub fn trapezoid_x_axis[T, X](t &vtl.Tensor[T], x &vtl.Tensor[X], axis int) !&vtl.Tensor[f64] {
	return trapezoid_x_axis_as[T, X, f64](t, x, axis)
}

// trapezoid_x_axis_as integrates along `axis` using U for accumulation/output.
pub fn trapezoid_x_axis_as[T, X, U](t &vtl.Tensor[T], x &vtl.Tensor[X], axis int) !&vtl.Tensor[U] {
	validate_trapezoid_type[U]()!
	if x.rank() != 1 {
		return error('trapezoid_x_axis: x must be one-dimensional')
	}
	axis_index := normalize_trapezoid_axis(t.rank(), axis)!
	if x.size() != t.shape[axis_index] {
		return error('trapezoid_x_axis: x length ${x.size()} does not match axis length ${t.shape[axis_index]}')
	}
	mut x_steps := []U{len: if x.size() > 0 { x.size() - 1 } else { 0 }}
	for i in 0 .. x_steps.len {
		x_steps[i] = vtl.cast[U](x.get_nth(i + 1)) - vtl.cast[U](x.get_nth(i))
	}
	return trapezoid_impl[T, U](t, x_steps, true, vtl.cast[U](1), axis_index)
}

fn trapezoid_impl[T, U](t &vtl.Tensor[T], x_steps []U, has_x bool, dx U, axis int) !&vtl.Tensor[U] {
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
	mut output := []U{len: output_size}
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
		mut integral := vtl.cast[U](0)
		for segment in 0 .. axis_size - 1 {
			input_index[axis_index] = segment
			y0 := vtl.cast[U](t.get(input_index))
			input_index[axis_index] = segment + 1
			y1 := vtl.cast[U](t.get(input_index))
			width := if has_x { x_steps[segment] } else { dx }
			integral += (y0 + y1) * vtl.cast[U](0.5) * width
		}
		output[flat_index] = integral
	}
	return vtl.from_array[U](output, output_shape)
}

fn validate_trapezoid_type[U]() ! {
	if !trapezoid_float_type[U]() {
		return error('trapezoid accumulator/output type must be f32 or f64')
	}
}

fn trapezoid_float_type[U]() bool {
	$if U is f32 || U is f64 {
		return true
	} $else {
		return false
	}
}

fn normalize_trapezoid_axis(rank int, axis int) !int {
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('trapezoid: axis ${axis} out of bounds for rank ${rank}')
	}
	return axis_index
}
