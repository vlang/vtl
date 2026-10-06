module la

import vtl

// tensordot contracts the last `axes` dimensions of a with the first `axes`
// dimensions of b. The contracted dimensions must match pairwise. The result
// contains the remaining dimensions of a followed by the remaining dimensions
// of b, preserving the input element type.
pub fn tensordot[T](a &vtl.Tensor[T], b &vtl.Tensor[T], axes int) !&vtl.Tensor[T] {
	if axes < 0 || axes > a.rank() || axes > b.rank() {
		return error('tensordot: axes ${axes} is invalid for input ranks ${a.rank()} and ${b.rank()}')
	}
	$if T is f32 || T is f64 {
		if axes == 1 && a.rank() == 2 && b.rank() == 2 && a.shape[1] == b.shape[0]
			&& a.shape[1] > 0 {
			return matmul[T](a, b)
		}
	}
	mut axes_a := []int{cap: axes}
	mut axes_b := []int{cap: axes}
	for i in 0 .. axes {
		axes_a << a.rank() - axes + i
		axes_b << i
	}
	return tensordot_axes[T](a, b, axes_a, axes_b)
}

// tensordot_axes contracts the specified axes of a and b pairwise. Negative
// axis numbers count from the end, as in NumPy. Uncontracted dimensions are
// ordered as the remaining axes of a followed by the remaining dimensions of b.
pub fn tensordot_axes[T](a &vtl.Tensor[T], b &vtl.Tensor[T], axes_a []int, axes_b []int) !&vtl.Tensor[T] {
	if axes_a.len != axes_b.len {
		return error('tensordot_axes: axis lists must have the same length')
	}

	mut normalized_a := []int{cap: axes_a.len}
	mut normalized_b := []int{cap: axes_b.len}
	mut contraction_shape := []int{cap: axes_a.len}
	for i, axis_a in axes_a {
		axis_b := axes_b[i]
		normalized_axis_a := if axis_a < 0 { axis_a + a.rank() } else { axis_a }
		normalized_axis_b := if axis_b < 0 { axis_b + b.rank() } else { axis_b }
		if normalized_axis_a < 0 || normalized_axis_a >= a.rank() {
			return error('tensordot_axes: axis ${axis_a} is out of range for input rank ${a.rank()}')
		}
		if normalized_axis_b < 0 || normalized_axis_b >= b.rank() {
			return error('tensordot_axes: axis ${axis_b} is out of range for input rank ${b.rank()}')
		}
		if normalized_axis_a in normalized_a || normalized_axis_b in normalized_b {
			return error('tensordot_axes: axes must not be repeated')
		}
		if a.shape[normalized_axis_a] != b.shape[normalized_axis_b] {
			return error('tensordot_axes: contracted dimensions ${a.shape[normalized_axis_a]} and ${b.shape[normalized_axis_b]} do not match')
		}
		normalized_a << normalized_axis_a
		normalized_b << normalized_axis_b
		contraction_shape << a.shape[normalized_axis_a]
	}

	mut free_a := []int{cap: a.rank() - normalized_a.len}
	mut free_b := []int{cap: b.rank() - normalized_b.len}
	mut output_shape := []int{cap: a.rank() + b.rank() - normalized_a.len - normalized_b.len}
	for axis in 0 .. a.rank() {
		if axis !in normalized_a {
			free_a << axis
			output_shape << a.shape[axis]
		}
	}
	for axis in 0 .. b.rank() {
		if axis !in normalized_b {
			free_b << axis
			output_shape << b.shape[axis]
		}
	}

	output_size := tensordot_shape_size(output_shape)
	contraction_size := tensordot_shape_size(contraction_shape)
	mut result := vtl.zeros[T](output_shape)
	if output_size == 0 || contraction_size == 0 {
		return result
	}
	mut output_index := []int{len: output_shape.len}
	mut contraction_index := []int{len: contraction_shape.len}
	mut index_a := []int{len: a.rank()}
	mut index_b := []int{len: b.rank()}
	for output_linear in 0 .. output_size {
		fill_tensordot_index(output_linear, output_shape, mut output_index)
		for i, axis in free_a {
			index_a[axis] = output_index[i]
		}
		for i, axis in free_b {
			index_b[axis] = output_index[free_a.len + i]
		}
		mut sum := T(0)
		for contraction_linear in 0 .. contraction_size {
			fill_tensordot_index(contraction_linear, contraction_shape, mut contraction_index)
			for i, axis in normalized_a {
				index_a[axis] = contraction_index[i]
			}
			for i, axis in normalized_b {
				index_b[axis] = contraction_index[i]
			}
			sum += a.get(index_a) * b.get(index_b)
		}
		result.set(output_index, sum)
	}
	return result
}

fn tensordot_shape_size(shape []int) int {
	mut size := 1
	for dimension in shape {
		size *= dimension
	}
	return size
}

fn fill_tensordot_index(linear int, shape []int, mut index []int) {
	mut remaining := linear
	for axis := shape.len - 1; axis >= 0; axis-- {
		index[axis] = remaining % shape[axis]
		remaining /= shape[axis]
	}
}
