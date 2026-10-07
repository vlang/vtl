module stats

import math
import vtl

// mean_along_axes computes a mean over several axes. With keepdims, each
// reduced axis remains in the output with length one. An empty axes list
// returns elementwise means, matching NumPy's axis=() behavior.
pub fn mean_along_axes[T](t &vtl.Tensor[T], axes []int, keepdims bool) !&vtl.Tensor[f64] {
	return moments_along_axes[T](t, axes, 0, .mean, false, keepdims)
}

// variance_along_axes computes population or sample variance over several
// axes. ddof is subtracted from the number of observations per output value.
pub fn variance_along_axes[T](t &vtl.Tensor[T], axes []int, ddof int, keepdims bool) !&vtl.Tensor[f64] {
	if ddof < 0 {
		return error('variance_along_axes: ddof must be non-negative')
	}
	return moments_along_axes[T](t, axes, ddof, .variance, false, keepdims)
}

// std_along_axes computes standard deviation over several axes.
pub fn std_along_axes[T](t &vtl.Tensor[T], axes []int, ddof int, keepdims bool) !&vtl.Tensor[f64] {
	variances := variance_along_axes[T](t, axes, ddof, keepdims)!
	return variances.map[T](fn (value f64, _ []int) f64 { return math.sqrt(value) })
}

// nanmean_along_axes computes means over several axes while ignoring NaNs.
pub fn nanmean_along_axes[T](t &vtl.Tensor[T], axes []int, keepdims bool) !&vtl.Tensor[f64] {
	return moments_along_axes[T](t, axes, 0, .mean, true, keepdims)
}

// nanvar_along_axes computes NaN-ignoring variance over several axes.
pub fn nanvar_along_axes[T](t &vtl.Tensor[T], axes []int, ddof int, keepdims bool) !&vtl.Tensor[f64] {
	if ddof < 0 {
		return error('nanvar_along_axes: ddof must be non-negative')
	}
	return moments_along_axes[T](t, axes, ddof, .variance, true, keepdims)
}

// nanstd_along_axes computes NaN-ignoring standard deviation over several axes.
pub fn nanstd_along_axes[T](t &vtl.Tensor[T], axes []int, ddof int, keepdims bool) !&vtl.Tensor[f64] {
	variances := nanvar_along_axes[T](t, axes, ddof, keepdims)!
	return variances.map[T](fn (value f64, _ []int) f64 { return math.sqrt(value) })
}

fn moments_along_axes[T](t &vtl.Tensor[T], axes []int, ddof int, output NanMomentOutput, ignore_nan bool, keepdims bool) !&vtl.Tensor[f64] {
	rank := t.rank()
	if rank == 0 && axes.len > 0 {
		return error('multi-axis moment reduction requires a tensor with at least one dimension')
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
	for dimension, size in t.shape {
		if reduced[dimension] {
			if keepdims {
				output_shape << 1
			}
		} else {
			output_shape << size
		}
	}
	mut result := vtl.empty[f64](output_shape, memory: .row_major)
	mut moments := []NanMoments{len: result.size}
	mut input_index := []int{len: rank}
	for flat_index in 0 .. t.size {
		mut remainder := flat_index
		for dimension := rank - 1; dimension >= 0; dimension-- {
			input_index[dimension] = remainder % t.shape[dimension]
			remainder /= t.shape[dimension]
		}
		mut output_flat_index := 0
		for dimension, coordinate in input_index {
			if !reduced[dimension] {
				output_flat_index = output_flat_index * t.shape[dimension] + coordinate
			}
		}
		value := f64(t.get(input_index))
		if !ignore_nan || !math.is_nan(value) {
			moments[output_flat_index].add(value)
		}
	}
	for i in 0 .. result.size {
		result.set_nth(i, moments[i].output(ddof, output))
	}
	return result
}
