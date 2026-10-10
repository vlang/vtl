module stats

import math
import vtl

// QuantileMethod selects one of NumPy's thirteen supported sample-quantile
// estimators.
pub enum QuantileMethod {
	inverted_cdf
	averaged_inverted_cdf
	closest_observation
	interpolated_inverted_cdf
	hazen
	weibull
	linear
	median_unbiased
	normal_unbiased
	lower
	higher
	midpoint
	nearest
}

// quantile_with_method computes a flattened quantile using the selected
// estimator. It sorts a copy and returns f64, including for integer inputs.
pub fn quantile_with_method[T](t &vtl.Tensor[T], q f64, method QuantileMethod) !f64 {
	validate_quantile(q)!
	if t.size == 0 {
		return error('quantile is undefined for an empty tensor')
	}
	mut values := tensor_float64_values[T](t, false)
	return quantile_method_values(mut values, q, method, true)
}

// nanquantile_with_method computes a flattened quantile while ignoring NaNs.
// It returns NaN if the input has no non-NaN values.
pub fn nanquantile_with_method[T](t &vtl.Tensor[T], q f64, method QuantileMethod) !f64 {
	validate_quantile(q)!
	if t.size == 0 {
		return error('quantile is undefined for an empty tensor')
	}
	mut values := tensor_float64_values[T](t, true)
	if values.len == 0 {
		return math.nan()
	}
	return quantile_method_values(mut values, q, method, true)
}

// quantiles_with_method computes several flattened quantiles from one sorted
// copy of the input values.
pub fn quantiles_with_method[T](t &vtl.Tensor[T], quantiles []f64, method QuantileMethod) !&vtl.Tensor[f64] {
	validate_quantiles(quantiles)!
	if t.size == 0 {
		return error('quantiles are undefined for an empty tensor')
	}
	mut values := tensor_float64_values[T](t, false)
	if values.any(math.is_nan(it)) {
		return vtl.from_1d[f64]([]f64{len: quantiles.len, init: math.nan()})
	}
	values.sort()
	return quantile_values_result(values, quantiles, method)
}

// nanquantiles_with_method computes several flattened quantiles while
// ignoring NaNs. If all values are NaN, every result is NaN.
pub fn nanquantiles_with_method[T](t &vtl.Tensor[T], quantiles []f64, method QuantileMethod) !&vtl.Tensor[f64] {
	validate_quantiles(quantiles)!
	if t.size == 0 {
		return error('quantiles are undefined for an empty tensor')
	}
	mut values := tensor_float64_values[T](t, true)
	if values.len == 0 {
		return vtl.from_1d[f64]([]f64{len: quantiles.len, init: math.nan()})
	}
	values.sort()
	return quantile_values_result(values, quantiles, method)
}

// quantile_axis_with_method reduces one axis and chooses whether to retain it
// as a length-one dimension.
pub fn quantile_axis_with_method[T](t &vtl.Tensor[T], q f64, axis int, method QuantileMethod, keepdims bool) !&vtl.Tensor[f64] {
	validate_quantile(q)!
	return quantile_axis_method_impl[T](t, q, axis, method, false, keepdims)
}

// nanquantile_axis_with_method reduces one axis while ignoring NaNs.
pub fn nanquantile_axis_with_method[T](t &vtl.Tensor[T], q f64, axis int, method QuantileMethod, keepdims bool) !&vtl.Tensor[f64] {
	validate_quantile(q)!
	return quantile_axis_method_impl[T](t, q, axis, method, true, keepdims)
}

// quantiles_axis_with_method computes several quantiles per axis slice. The
// quantile dimension is prepended to the output shape, as in NumPy.
pub fn quantiles_axis_with_method[T](t &vtl.Tensor[T], quantiles []f64, axis int, method QuantileMethod) !&vtl.Tensor[f64] {
	validate_quantiles(quantiles)!
	return quantiles_axis_method_impl[T](t, quantiles, axis, method, false, false)
}

// quantiles_axis_with_method_keepdims retains the reduced axis as length one.
pub fn quantiles_axis_with_method_keepdims[T](t &vtl.Tensor[T], quantiles []f64, axis int, method QuantileMethod, keepdims bool) !&vtl.Tensor[f64] {
	validate_quantiles(quantiles)!
	return quantiles_axis_method_impl[T](t, quantiles, axis, method, false, keepdims)
}

// nanquantiles_axis_with_method computes several NaN-ignoring quantiles per
// axis slice, with the quantile dimension prepended to the output shape.
pub fn nanquantiles_axis_with_method[T](t &vtl.Tensor[T], quantiles []f64, axis int, method QuantileMethod) !&vtl.Tensor[f64] {
	validate_quantiles(quantiles)!
	return quantiles_axis_method_impl[T](t, quantiles, axis, method, true, false)
}

// nanquantiles_axis_with_method_keepdims retains the reduced axis as length one.
pub fn nanquantiles_axis_with_method_keepdims[T](t &vtl.Tensor[T], quantiles []f64, axis int, method QuantileMethod, keepdims bool) !&vtl.Tensor[f64] {
	validate_quantiles(quantiles)!
	return quantiles_axis_method_impl[T](t, quantiles, axis, method, true, keepdims)
}

// percentile_with_method is the 0..100-scale form of quantile_with_method.
pub fn percentile_with_method[T](t &vtl.Tensor[T], percentile f64, method QuantileMethod) !f64 {
	return quantile_with_method[T](t, percentile_quantile(percentile)!, method)
}

// nanpercentile_with_method is the 0..100-scale NaN-ignoring quantile.
pub fn nanpercentile_with_method[T](t &vtl.Tensor[T], percentile f64, method QuantileMethod) !f64 {
	return nanquantile_with_method[T](t, percentile_quantile(percentile)!, method)
}

// percentiles_with_method computes several 0..100-scale percentiles.
pub fn percentiles_with_method[T](t &vtl.Tensor[T], percentiles []f64, method QuantileMethod) !&vtl.Tensor[f64] {
	return quantiles_with_method[T](t, percentiles_to_quantiles(percentiles)!, method)
}

// nanpercentiles_with_method computes several NaN-ignoring 0..100-scale
// percentiles.
pub fn nanpercentiles_with_method[T](t &vtl.Tensor[T], percentiles []f64, method QuantileMethod) !&vtl.Tensor[f64] {
	return nanquantiles_with_method[T](t, percentiles_to_quantiles(percentiles)!, method)
}

// percentile_axis_with_method computes one percentile per axis slice.
pub fn percentile_axis_with_method[T](t &vtl.Tensor[T], percentile f64, axis int, method QuantileMethod, keepdims bool) !&vtl.Tensor[f64] {
	return quantile_axis_with_method[T](t, percentile_quantile(percentile)!, axis, method, keepdims)
}

// nanpercentile_axis_with_method computes NaN-ignoring percentiles per slice.
pub fn nanpercentile_axis_with_method[T](t &vtl.Tensor[T], percentile f64, axis int, method QuantileMethod, keepdims bool) !&vtl.Tensor[f64] {
	return nanquantile_axis_with_method[T](t, percentile_quantile(percentile)!, axis, method, keepdims)
}

// percentiles_axis_with_method computes several percentiles per axis slice.
pub fn percentiles_axis_with_method[T](t &vtl.Tensor[T], percentiles []f64, axis int, method QuantileMethod) !&vtl.Tensor[f64] {
	return quantiles_axis_with_method[T](t, percentiles_to_quantiles(percentiles)!, axis, method)
}

// percentiles_axis_with_method_keepdims retains the reduced axis as length one.
pub fn percentiles_axis_with_method_keepdims[T](t &vtl.Tensor[T], percentiles []f64, axis int, method QuantileMethod, keepdims bool) !&vtl.Tensor[f64] {
	return quantiles_axis_with_method_keepdims[T](t, percentiles_to_quantiles(percentiles)!, axis,
		method, keepdims)
}

// nanpercentiles_axis_with_method computes several NaN-ignoring percentiles per axis slice.
pub fn nanpercentiles_axis_with_method[T](t &vtl.Tensor[T], percentiles []f64, axis int, method QuantileMethod) !&vtl.Tensor[f64] {
	return nanquantiles_axis_with_method[T](t, percentiles_to_quantiles(percentiles)!, axis, method)
}

// nanpercentiles_axis_with_method_keepdims retains the reduced axis as length one.
pub fn nanpercentiles_axis_with_method_keepdims[T](t &vtl.Tensor[T], percentiles []f64, axis int, method QuantileMethod, keepdims bool) !&vtl.Tensor[f64] {
	return nanquantiles_axis_with_method_keepdims[T](t, percentiles_to_quantiles(percentiles)!, axis,
		method, keepdims)
}

fn validate_quantile(q f64) ! {
	if math.is_nan(q) || math.is_inf(q, 0) || q < 0 || q > 1 {
		return error('quantile must be between 0 and 1')
	}
}

fn validate_quantiles(quantiles []f64) ! {
	for q in quantiles {
		validate_quantile(q)!
	}
}

fn quantile_values_result(values []f64, quantiles []f64, method QuantileMethod) !&vtl.Tensor[f64] {
	mut results := []f64{len: quantiles.len}
	for i, q in quantiles {
		results[i] = quantile_method_sorted(values, q, method)
	}
	return vtl.from_1d[f64](results)
}

fn quantile_method_values(mut values []f64, q f64, method QuantileMethod, sort_values bool) f64 {
	if values.any(math.is_nan(it)) {
		return math.nan()
	}
	if values.len == 0 {
		return math.nan()
	}
	lower_index, upper_index, weight := quantile_order_indices(q, values.len, method)
	if sort_values {
		select_quantile_index(mut values, lower_index, 0, values.len - 1)
		if upper_index > lower_index {
			select_quantile_index(mut values, upper_index, lower_index + 1, values.len - 1)
		}
	}
	return values[lower_index] * (1 - weight) + values[upper_index] * weight
}

fn quantile_method_sorted(values []f64, q f64, method QuantileMethod) f64 {
	n := values.len
	if n == 0 {
		return math.nan()
	}
	lower_index, upper_index, weight := quantile_order_indices(q, n, method)
	return values[lower_index] * (1 - weight) + values[upper_index] * weight
}

fn quantile_order_indices(q f64, n int, method QuantileMethod) (int, int, f64) {
	mut position := q * f64(n - 1)
	match method {
		.inverted_cdf, .averaged_inverted_cdf, .closest_observation {
			m := match method {
				.inverted_cdf, .averaged_inverted_cdf { 0.0 }
				else { -0.5 }
			}
			index := q * f64(n) + m - 1
			lower := int(math.floor(index))
			fraction := index - f64(lower)
			weight := match method {
				.inverted_cdf {
					if fraction > 0 { 1.0 } else { 0.0 }
				}
				.averaged_inverted_cdf {
					if fraction > 0 { 1.0 } else { 0.5 }
				}
				.closest_observation {
					if fraction == 0 && lower % 2 == 1 { 0.0 } else { 1.0 }
				}
				else { 0.0 }
			}
			position = f64(lower) + weight
		}
		.interpolated_inverted_cdf { position = q * f64(n) - 1 }
		.hazen { position = q * f64(n) - 0.5 }
		.weibull { position = q * f64(n + 1) - 1 }
		.linear { position = q * f64(n - 1) }
		.median_unbiased { position = q * f64(n) + q / 3 + 1.0 / 3 - 1 }
		.normal_unbiased { position = q * f64(n) + q / 4 + 3.0 / 8 - 1 }
		.lower, .higher, .midpoint, .nearest {
			base := q * f64(n - 1)
			lower := math.floor(base)
			fraction := base - lower
			position = lower + match method {
				.lower { 0.0 }
				.higher { 1.0 }
				.midpoint { 0.5 }
				.nearest {
					if fraction > 0.5 { 1.0 } else { 0.0 }
				}
				else { 0.0 }
			}
		}
	}
	position = math.max(0, math.min(position, f64(n - 1)))
	lower_index := int(math.floor(position))
	upper_index := math.min(lower_index + 1, n - 1)
	weight := position - f64(lower_index)
	return lower_index, upper_index, weight
}

// select_quantile_index partially orders values so the requested order
// statistic is available without sorting the entire input.
fn select_quantile_index(mut values []f64, target int, left int, right int) {
	mut low := left
	mut high := right
	mut depth := 2 * int(math.log2(f64(right - left + 1))) + 1
	for low < high {
		if depth == 0 {
			// Bound worst-case selection cost with the canonical in-place sort.
			values.sort()
			return
		}
		depth--
		middle := low + (high - low) / 2
		pivot := quantile_median_of_three(values[low], values[middle], values[high])
		mut less := low
		mut current := low
		mut greater := high
		for current <= greater {
			if values[current] < pivot {
				values[less], values[current] = values[current], values[less]
				less++
				current++
			} else if values[current] > pivot {
				values[current], values[greater] = values[greater], values[current]
				greater--
			} else {
				current++
			}
		}
		if target < less {
			high = less - 1
		} else if target > greater {
			low = greater + 1
		} else {
			return
		}
	}
}

fn quantile_median_of_three(a f64, b f64, c f64) f64 {
	if a < b {
		if b < c {
			return b
		}
		return if a < c { c } else { a }
	}
	if a < c {
		return a
	}
	return if b < c { c } else { b }
}

fn quantile_axis_method_impl[T](t &vtl.Tensor[T], q f64, axis int, method QuantileMethod, ignore_nan bool, keepdims bool) !&vtl.Tensor[f64] {
	rank := t.rank()
	if rank == 0 {
		return error('quantile axis requires a tensor with at least one dimension')
	}
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('quantile axis ${axis} out of bounds for rank ${rank}')
	}
	axis_size := t.shape[axis_index]
	if axis_size == 0 {
		return error('quantile is undefined for an empty axis')
	}
	mut output_shape := t.shape.clone()
	if keepdims {
		output_shape[axis_index] = 1
	} else {
		output_shape.delete(axis_index)
		if output_shape.len == 0 {
			output_shape = [1]
		}
	}
	mut result := vtl.empty[f64](output_shape, memory: .row_major)
	mut slice_count := 1
	for dim, size in t.shape {
		if dim != axis_index {
			slice_count *= size
		}
	}
	mut index := []int{len: rank}
	mut values := []f64{cap: axis_size}
	for slice in 0 .. slice_count {
		decode_nan_slice(slice, t.shape, axis_index, mut index)
		values.clear()
		mut has_nan := false
		for position in 0 .. axis_size {
			index[axis_index] = position
			value := f64(t.get(index))
			if math.is_nan(value) {
				has_nan = true
			} else {
				values << value
			}
		}
		if !ignore_nan && has_nan {
			values.clear()
		} else {
			values.sort()
		}
		if keepdims {
			index[axis_index] = 0
			result.set(index, quantile_method_sorted(values, q, method))
		} else {
			mut output_index := index[..axis_index].clone()
			output_index << index[axis_index + 1..]
			if output_index.len == 0 {
				output_index = [0]
			}
			result.set(output_index, quantile_method_sorted(values, q, method))
		}
	}
	return result
}

fn quantiles_axis_method_impl[T](t &vtl.Tensor[T], quantiles []f64, axis int, method QuantileMethod, ignore_nan bool, keepdims bool) !&vtl.Tensor[f64] {
	rank := t.rank()
	if rank == 0 {
		return error('quantile axis requires a tensor with at least one dimension')
	}
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('quantile axis ${axis} out of bounds for rank ${rank}')
	}
	axis_size := t.shape[axis_index]
	if axis_size == 0 {
		return error('quantile is undefined for an empty axis')
	}
	mut output_shape := [quantiles.len]
	for dim, size in t.shape {
		if dim == axis_index {
			if keepdims {
				output_shape << 1
			}
		} else {
			output_shape << size
		}
	}
	mut result := vtl.empty[f64](output_shape, memory: .row_major)
	mut slice_count := 1
	for dim, size in t.shape {
		if dim != axis_index {
			slice_count *= size
		}
	}
	mut index := []int{len: rank}
	mut output_index := []int{len: output_shape.len}
	mut values := []f64{cap: axis_size}
	for slice in 0 .. slice_count {
		decode_nan_slice(slice, t.shape, axis_index, mut index)
		values.clear()
		mut has_nan := false
		for position in 0 .. axis_size {
			index[axis_index] = position
			value := f64(t.get(index))
			if math.is_nan(value) {
				has_nan = true
			} else {
				values << value
			}
		}
		if !ignore_nan && has_nan {
			values.clear()
		} else {
			values.sort()
		}
		output_index[0] = 0
		mut output_dimension := 1
		for dimension in 0 .. rank {
			if dimension == axis_index {
				if keepdims {
					output_index[output_dimension] = 0
					output_dimension++
				}
			} else {
				output_index[output_dimension] = index[dimension]
				output_dimension++
			}
		}
		for quantile_index, q in quantiles {
			output_index[0] = quantile_index
			result.set(output_index, quantile_method_sorted(values, q, method))
		}
	}
	return result
}
