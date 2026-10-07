module stats

import math
import vtl

struct NanMoments {
mut:
	count int
	mean  f64
	m2    f64
}

// nanmean computes the arithmetic mean while ignoring NaN values. It returns
// NaN when the tensor contains no non-NaN values.
pub fn nanmean[T](t &vtl.Tensor[T]) f64 {
	mut moments := NanMoments{}
	mut iter := t.iterator[T]()
	for {
		value, _ := iter.next() or { break }
		if !math.is_nan(f64(value)) {
			moments.add(f64(value))
		}
	}
	return moments.mean_or_nan()
}

// nanvar computes population or sample variance while ignoring NaN values.
// ddof is subtracted from the number of non-NaN values; insufficient values
// produce NaN, matching NumPy's nanvar result.
pub fn nanvar[T](t &vtl.Tensor[T], ddof int) !f64 {
	if ddof < 0 {
		return error('nanvar: ddof must be non-negative')
	}
	mut moments := NanMoments{}
	mut iter := t.iterator[T]()
	for {
		value, _ := iter.next() or { break }
		if !math.is_nan(f64(value)) {
			moments.add(f64(value))
		}
	}
	return moments.variance_or_nan(ddof)
}

// nanstd computes standard deviation while ignoring NaN values.
pub fn nanstd[T](t &vtl.Tensor[T], ddof int) !f64 {
	return math.sqrt(nanvar(t, ddof)?)
}

// nanmean_axis computes means along axis and retains the reduced dimension as
// length one. NaN-only slices produce NaN.
pub fn nanmean_axis[T](t &vtl.Tensor[T], axis int) !&vtl.Tensor[f64] {
	return nan_moments_axis[T](t, axis, 0, .mean, true)
}

// mean_axis computes the arithmetic mean along axis and retains that axis with
// length one. NaNs propagate to the corresponding output slice.
pub fn mean_axis[T](t &vtl.Tensor[T], axis int) !&vtl.Tensor[f64] {
	return nan_moments_axis[T](t, axis, 0, .mean, false)
}

// nanvar_axis computes variance along axis and retains the reduced dimension as
// length one. NaN-only or under-sampled slices produce NaN.
pub fn nanvar_axis[T](t &vtl.Tensor[T], axis int, ddof int) !&vtl.Tensor[f64] {
	if ddof < 0 {
		return error('nanvar_axis: ddof must be non-negative')
	}
	return nan_moments_axis[T](t, axis, ddof, .variance, true)
}

// variance_axis computes population or sample variance along axis. The
// reduced axis is retained with length one and NaNs propagate per slice.
pub fn variance_axis[T](t &vtl.Tensor[T], axis int, ddof int) !&vtl.Tensor[f64] {
	if ddof < 0 {
		return error('variance_axis: ddof must be non-negative')
	}
	return nan_moments_axis[T](t, axis, ddof, .variance, false)
}

// nanstd_axis computes standard deviation along axis and retains the reduced
// dimension as length one.
pub fn nanstd_axis[T](t &vtl.Tensor[T], axis int, ddof int) !&vtl.Tensor[f64] {
	variances := nanvar_axis[T](t, axis, ddof)!
	return variances.map[T](fn (value f64, _ []int) f64 { return math.sqrt(value) })
}

// std_axis computes standard deviation along axis and retains the reduced
// dimension with length one.
pub fn std_axis[T](t &vtl.Tensor[T], axis int, ddof int) !&vtl.Tensor[f64] {
	variances := variance_axis[T](t, axis, ddof)!
	return variances.map[T](fn (value f64, _ []int) f64 { return math.sqrt(value) })
}

// mean_along_axis computes means along axis and optionally removes that axis.
pub fn mean_along_axis[T](t &vtl.Tensor[T], axis int, keepdims bool) !&vtl.Tensor[f64] {
	result := mean_axis[T](t, axis)!
	return reshape_reduction_axis[T](result, axis, keepdims)
}

// variance_along_axis computes variance along axis and optionally removes it.
pub fn variance_along_axis[T](t &vtl.Tensor[T], axis int, ddof int, keepdims bool) !&vtl.Tensor[f64] {
	result := variance_axis[T](t, axis, ddof)!
	return reshape_reduction_axis[f64](result, axis, keepdims)
}

// std_along_axis computes standard deviation and optionally removes its axis.
pub fn std_along_axis[T](t &vtl.Tensor[T], axis int, ddof int, keepdims bool) !&vtl.Tensor[f64] {
	result := std_axis[T](t, axis, ddof)!
	return reshape_reduction_axis[f64](result, axis, keepdims)
}

// nanmean_along_axis computes NaN-ignoring means and optionally removes axis.
pub fn nanmean_along_axis[T](t &vtl.Tensor[T], axis int, keepdims bool) !&vtl.Tensor[f64] {
	result := nanmean_axis[T](t, axis)!
	return reshape_reduction_axis[f64](result, axis, keepdims)
}

// nanvar_along_axis computes NaN-ignoring variance and optionally removes axis.
pub fn nanvar_along_axis[T](t &vtl.Tensor[T], axis int, ddof int, keepdims bool) !&vtl.Tensor[f64] {
	result := nanvar_axis[T](t, axis, ddof)!
	return reshape_reduction_axis[f64](result, axis, keepdims)
}

// nanstd_along_axis computes NaN-ignoring std and optionally removes axis.
pub fn nanstd_along_axis[T](t &vtl.Tensor[T], axis int, ddof int, keepdims bool) !&vtl.Tensor[f64] {
	result := nanstd_axis[T](t, axis, ddof)!
	return reshape_reduction_axis[f64](result, axis, keepdims)
}

fn reshape_reduction_axis[T](result &vtl.Tensor[T], axis int, keepdims bool) !&vtl.Tensor[T] {
	if keepdims {
		return result
	}
	rank := result.rank()
	axis_index := if axis < 0 { axis + rank } else { axis }
	mut shape := []int{cap: rank - 1}
	for dimension, size in result.shape {
		if dimension != axis_index {
			shape << size
		}
	}
	return result.reshape[T](shape)
}

enum NanMomentOutput {
	mean
	variance
}

fn nan_moments_axis[T](t &vtl.Tensor[T], axis int, ddof int, output NanMomentOutput, ignore_nan bool) !&vtl.Tensor[f64] {
	rank := t.rank()
	if rank == 0 {
		return error('nan reduction axis requires a tensor with at least one dimension')
	}
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('nan reduction axis ${axis} out of bounds for rank ${rank}')
	}
	mut out_shape := t.shape.clone()
	axis_size := t.shape[axis_index]
	out_shape[axis_index] = 1
	mut result := vtl.empty[f64](out_shape, memory: .row_major)
	mut slice_count := 1
	for dim, dimension in t.shape {
		if dim != axis_index {
			slice_count *= dimension
		}
	}
	mut index := []int{len: rank}
	for slice in 0 .. slice_count {
		decode_nan_slice(slice, t.shape, axis_index, mut index)
		mut moments := NanMoments{}
		for position in 0 .. axis_size {
			index[axis_index] = position
			value := f64(t.get(index))
			if !ignore_nan || !math.is_nan(value) {
				moments.add(value)
			}
		}
		index[axis_index] = 0
		result.set(index, moments.output(ddof, output))
	}
	return result
}

fn (mut moments NanMoments) add(value f64) {
	moments.count++
	delta := value - moments.mean
	moments.mean += delta / f64(moments.count)
	moments.m2 += delta * (value - moments.mean)
}

fn (moments NanMoments) mean_or_nan() f64 {
	return if moments.count == 0 { math.nan() } else { moments.mean }
}

fn (moments NanMoments) variance_or_nan(ddof int) f64 {
	if moments.count <= ddof {
		return math.nan()
	}
	return moments.m2 / f64(moments.count - ddof)
}

fn (moments NanMoments) output(ddof int, output NanMomentOutput) f64 {
	return match output {
		.mean { moments.mean_or_nan() }
		.variance { moments.variance_or_nan(ddof) }
	}
}

fn decode_nan_slice(line int, shape []int, axis int, mut index []int) {
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
