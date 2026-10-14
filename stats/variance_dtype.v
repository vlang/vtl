module stats

import math
import vtl

struct TypedMoments[U] {
mut:
	count int
	mean  U
	m2    U
}

enum TypedMomentOutput {
	mean
	variance
	standard_deviation
}

// variance_as computes population or sample variance using U for accumulation
// and output. ddof is subtracted from the observation count.
pub fn variance_as[T, U](t &vtl.Tensor[T], ddof int) !U {
	if ddof < 0 {
		return error('variance_as: ddof must be non-negative')
	}
	if t.size <= ddof {
		return error('variance_as: observations must exceed ddof')
	}
	mut moments := TypedMoments[U]{}
	for index in 0 .. t.size {
		moments.add(vtl.cast[U](t.get_nth(index)))
	}
	return moments.variance(ddof)
}

// std_as computes standard deviation using U for accumulation and output.
pub fn std_as[T, U](t &vtl.Tensor[T], ddof int) !U {
	variance := variance_as[T, U](t, ddof)!
	return vtl.cast[U](math.sqrt(f64(variance)))
}

// nanmean_as computes a mean while ignoring NaNs, using U for accumulation
// and output. An empty or all-NaN tensor returns NaN for floating-point U.
pub fn nanmean_as[T, U](t &vtl.Tensor[T]) U {
	mut moments := TypedMoments[U]{}
	for index in 0 .. t.size {
		value := t.get_nth(index)
		if !math.is_nan(f64(value)) {
			moments.add(vtl.cast[U](value))
		}
	}
	return moments.mean_or_nan()
}

// nanvariance_as computes variance while ignoring NaNs and using U for
// accumulation and output. Insufficient non-NaN observations return NaN for
// floating-point U, matching the NaN-aware axis reductions.
pub fn nanvariance_as[T, U](t &vtl.Tensor[T], ddof int) !U {
	if ddof < 0 {
		return error('nanvariance_as: ddof must be non-negative')
	}
	mut moments := TypedMoments[U]{}
	for index in 0 .. t.size {
		value := t.get_nth(index)
		if !math.is_nan(f64(value)) {
			moments.add(vtl.cast[U](value))
		}
	}
	return moments.variance_or_nan(ddof)
}

// nanstd_as computes standard deviation while ignoring NaNs and using U for
// accumulation and output.
pub fn nanstd_as[T, U](t &vtl.Tensor[T], ddof int) !U {
	variance := nanvariance_as[T, U](t, ddof)!
	return vtl.cast[U](math.sqrt(f64(variance)))
}

// variance_along_axis_as reduces one axis using U for accumulation and output.
pub fn variance_along_axis_as[T, U](t &vtl.Tensor[T], axis int, ddof int, keepdims bool) !&vtl.Tensor[U] {
	return typed_moments_reduce[T, U](t, [axis], ddof, keepdims, .variance, false)
}

// std_along_axis_as computes standard deviation along one axis using U.
pub fn std_along_axis_as[T, U](t &vtl.Tensor[T], axis int, ddof int, keepdims bool) !&vtl.Tensor[U] {
	return typed_moments_reduce[T, U](t, [axis], ddof, keepdims, .standard_deviation, false)
}

// variance_along_axes_as reduces several axes using U for accumulation/output.
pub fn variance_along_axes_as[T, U](t &vtl.Tensor[T], axes []int, ddof int, keepdims bool) !&vtl.Tensor[U] {
	return typed_moments_reduce[T, U](t, axes, ddof, keepdims, .variance, false)
}

// std_along_axes_as computes standard deviation over several axes using U.
pub fn std_along_axes_as[T, U](t &vtl.Tensor[T], axes []int, ddof int, keepdims bool) !&vtl.Tensor[U] {
	return typed_moments_reduce[T, U](t, axes, ddof, keepdims, .standard_deviation, false)
}

// nanmean_along_axis_as computes a NaN-ignoring mean along one axis using U.
pub fn nanmean_along_axis_as[T, U](t &vtl.Tensor[T], axis int, keepdims bool) !&vtl.Tensor[U] {
	return typed_moments_reduce[T, U](t, [axis], 0, keepdims, .mean, true)
}

// nanvariance_along_axis_as computes NaN-ignoring variance along one axis.
pub fn nanvariance_along_axis_as[T, U](t &vtl.Tensor[T], axis int, ddof int, keepdims bool) !&vtl.Tensor[U] {
	return typed_moments_reduce[T, U](t, [axis], ddof, keepdims, .variance, true)
}

// nanstd_along_axis_as computes NaN-ignoring standard deviation along one axis.
pub fn nanstd_along_axis_as[T, U](t &vtl.Tensor[T], axis int, ddof int, keepdims bool) !&vtl.Tensor[U] {
	return typed_moments_reduce[T, U](t, [axis], ddof, keepdims, .standard_deviation, true)
}

// nanmean_along_axes_as computes a NaN-ignoring mean over several axes using U.
pub fn nanmean_along_axes_as[T, U](t &vtl.Tensor[T], axes []int, keepdims bool) !&vtl.Tensor[U] {
	return typed_moments_reduce[T, U](t, axes, 0, keepdims, .mean, true)
}

// nanvariance_along_axes_as computes NaN-ignoring variance over several axes.
pub fn nanvariance_along_axes_as[T, U](t &vtl.Tensor[T], axes []int, ddof int, keepdims bool) !&vtl.Tensor[U] {
	return typed_moments_reduce[T, U](t, axes, ddof, keepdims, .variance, true)
}

// nanstd_along_axes_as computes NaN-ignoring standard deviation over several axes.
pub fn nanstd_along_axes_as[T, U](t &vtl.Tensor[T], axes []int, ddof int, keepdims bool) !&vtl.Tensor[U] {
	return typed_moments_reduce[T, U](t, axes, ddof, keepdims, .standard_deviation, true)
}

fn typed_moments_reduce[T, U](t &vtl.Tensor[T], axes []int, ddof int, keepdims bool, output TypedMomentOutput, ignore_nan bool) !&vtl.Tensor[U] {
	if ddof < 0 {
		return error('variance reduction: ddof must be non-negative')
	}
	rank := t.rank()
	if rank == 0 && axes.len > 0 {
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
	for dimension, size in t.shape {
		if reduced[dimension] {
			if keepdims {
				output_shape << 1
			}
		} else {
			output_shape << size
		}
	}
	mut result := vtl.empty[U](output_shape, memory: .row_major)
	mut moments := []TypedMoments[U]{len: result.size}
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
		value := t.get(input_index)
		if ignore_nan && math.is_nan(f64(value)) {
			continue
		}
		moments[output_flat_index].add(vtl.cast[U](value))
	}
	for index in 0 .. result.size {
		moment := moments[index]
		value := match output {
			.mean { moment.mean_or_nan() }
			.variance { moment.variance_or_nan(ddof) }
			.standard_deviation { vtl.cast[U](math.sqrt(f64(moment.variance_or_nan(ddof)))) }
		}
		result.set_nth(index, value)
	}
	return result
}

fn (mut moments TypedMoments[U]) add(value U) {
	moments.count++
	delta := value - moments.mean
	moments.mean += delta / vtl.cast[U](moments.count)
	moments.m2 += delta * (value - moments.mean)
}

fn (moments TypedMoments[U]) variance(ddof int) U {
	return moments.m2 / vtl.cast[U](moments.count - ddof)
}

fn (moments TypedMoments[U]) mean_or_nan() U {
	if moments.count == 0 {
		$if U is f32 || U is f64 {
			return U(math.nan())
		} $else {
			return vtl.cast[U](0)
		}
	}
	return moments.mean
}

fn (moments TypedMoments[U]) variance_or_nan(ddof int) U {
	if moments.count <= ddof || moments.count == 0 {
		$if U is f32 || U is f64 {
			return U(math.nan())
		} $else {
			return vtl.cast[U](0)
		}
	}
	return moments.variance(ddof)
}
