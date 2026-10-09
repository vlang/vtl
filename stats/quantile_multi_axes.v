module stats

import math
import vtl

// quantile_axes_with_method reduces multiple axes using a NumPy-compatible
// estimator. keepdims retains each reduced dimension with length one.
pub fn quantile_axes_with_method[T](t &vtl.Tensor[T], q f64, axes []int, method QuantileMethod, keepdims bool) !&vtl.Tensor[f64] {
	return quantiles_multi_axis_impl[T](t, [q], axes, method, false, keepdims, false)
}

// nanquantile_axes_with_method reduces multiple axes while ignoring NaNs.
pub fn nanquantile_axes_with_method[T](t &vtl.Tensor[T], q f64, axes []int, method QuantileMethod, keepdims bool) !&vtl.Tensor[f64] {
	return quantiles_multi_axis_impl[T](t, [q], axes, method, true, keepdims, false)
}

// quantiles_axes_with_method computes multiple estimator values after one sort
// per reduced multi-axis slice. The quantile dimension is prepended.
pub fn quantiles_axes_with_method[T](t &vtl.Tensor[T], quantiles []f64, axes []int, method QuantileMethod) !&vtl.Tensor[f64] {
	return quantiles_multi_axis_impl[T](t, quantiles, axes, method, false, false, true)
}

// quantiles_axes_with_method_keepdims retains reduced dimensions as length one.
pub fn quantiles_axes_with_method_keepdims[T](t &vtl.Tensor[T], quantiles []f64, axes []int, method QuantileMethod, keepdims bool) !&vtl.Tensor[f64] {
	return quantiles_multi_axis_impl[T](t, quantiles, axes, method, false, keepdims, true)
}

// nanquantiles_axes_with_method computes multiple NaN-aware estimator values
// per reduced multi-axis slice.
pub fn nanquantiles_axes_with_method[T](t &vtl.Tensor[T], quantiles []f64, axes []int, method QuantileMethod) !&vtl.Tensor[f64] {
	return quantiles_multi_axis_impl[T](t, quantiles, axes, method, true, false, true)
}

// nanquantiles_axes_with_method_keepdims retains reduced dimensions as length one.
pub fn nanquantiles_axes_with_method_keepdims[T](t &vtl.Tensor[T], quantiles []f64, axes []int, method QuantileMethod, keepdims bool) !&vtl.Tensor[f64] {
	return quantiles_multi_axis_impl[T](t, quantiles, axes, method, true, keepdims, true)
}

// percentile_axes_with_method is the 0..100-scale multi-axis quantile form.
pub fn percentile_axes_with_method[T](t &vtl.Tensor[T], percentile f64, axes []int, method QuantileMethod, keepdims bool) !&vtl.Tensor[f64] {
	return quantile_axes_with_method[T](t, percentile_quantile(percentile)!, axes, method,
		keepdims)
}

// nanpercentile_axes_with_method is the NaN-aware multi-axis percentile form.
pub fn nanpercentile_axes_with_method[T](t &vtl.Tensor[T], percentile f64, axes []int, method QuantileMethod, keepdims bool) !&vtl.Tensor[f64] {
	return nanquantile_axes_with_method[T](t, percentile_quantile(percentile)!, axes,
		method, keepdims)
}

// percentiles_axes_with_method computes multiple multi-axis percentiles.
pub fn percentiles_axes_with_method[T](t &vtl.Tensor[T], percentiles []f64, axes []int, method QuantileMethod) !&vtl.Tensor[f64] {
	return quantiles_axes_with_method[T](t, percentiles_to_quantiles(percentiles)!, axes,
		method)
}

// nanpercentiles_axes_with_method computes multiple NaN-aware multi-axis percentiles.
pub fn nanpercentiles_axes_with_method[T](t &vtl.Tensor[T], percentiles []f64, axes []int, method QuantileMethod) !&vtl.Tensor[f64] {
	return nanquantiles_axes_with_method[T](t, percentiles_to_quantiles(percentiles)!, axes,
		method)
}

// percentiles_axes_with_method_keepdims retains reduced dimensions as length one.
pub fn percentiles_axes_with_method_keepdims[T](t &vtl.Tensor[T], percentiles []f64, axes []int, method QuantileMethod, keepdims bool) !&vtl.Tensor[f64] {
	return quantiles_axes_with_method_keepdims[T](t, percentiles_to_quantiles(percentiles)!, axes,
		method, keepdims)
}

// nanpercentiles_axes_with_method_keepdims is the NaN-aware retained-shape form.
pub fn nanpercentiles_axes_with_method_keepdims[T](t &vtl.Tensor[T], percentiles []f64, axes []int, method QuantileMethod, keepdims bool) !&vtl.Tensor[f64] {
	return nanquantiles_axes_with_method_keepdims[T](t, percentiles_to_quantiles(percentiles)!, axes,
		method, keepdims)
}

fn quantiles_multi_axis_impl[T](t &vtl.Tensor[T], quantiles []f64, axes []int, method QuantileMethod, ignore_nan bool, keepdims bool, multi bool) !&vtl.Tensor[f64] {
	validate_quantiles(quantiles)!
	rank := t.rank()
	if rank == 0 || axes.len == 0 {
		return error('multi-axis quantile requires a ranked tensor and at least one axis')
	}
	mut reduced := []bool{len: rank}
	for axis in axes {
		axis_index := if axis < 0 { axis + rank } else { axis }
		if axis_index < 0 || axis_index >= rank {
			return error('quantile axis ${axis} out of bounds for rank ${rank}')
		}
		if reduced[axis_index] {
			return error('quantile axis ${axis} appears more than once')
		}
		if t.shape[axis_index] == 0 {
			return error('quantile is undefined for an empty axis')
		}
		reduced[axis_index] = true
	}
	mut output_shape := []int{cap: rank + 1}
	if multi {
		output_shape << quantiles.len
	}
	for dimension, size in t.shape {
		if reduced[dimension] {
			if keepdims {
				output_shape << 1
			}
		} else {
			output_shape << size
		}
	}
	if output_shape.len == 0 {
		output_shape = [1]
	}
	mut result := vtl.empty[f64](output_shape, memory: .row_major)
	mut slice_count := 1
	mut reduction_count := 1
	for dimension, size in t.shape {
		if reduced[dimension] {
			reduction_count *= size
		} else {
			slice_count *= size
		}
	}
	mut input_index := []int{len: rank}
	mut output_index := []int{len: output_shape.len}
	mut values := []f64{cap: reduction_count}
	for slice in 0 .. slice_count {
		decode_quantile_multi_slice(slice, t.shape, reduced, mut input_index)
		values.clear()
		mut has_nan := false
		for flat_reduced in 0 .. reduction_count {
			decode_quantile_reduced_index(flat_reduced, t.shape, reduced, mut input_index)
			value := f64(t.get(input_index))
			if math.is_nan(value) {
				has_nan = true
				if ignore_nan {
					continue
				}
			}
			values << value
		}
		if !ignore_nan && has_nan {
			values.clear()
		} else {
			values.sort()
		}
		decode_quantile_multi_slice(slice, t.shape, reduced, mut input_index)
		mut results := []f64{len: quantiles.len}
		for quantile_index, q in quantiles {
			results[quantile_index] = quantile_method_sorted(values, q, method)
		}
		if multi {
			output_index[0] = 0
			mut output_dimension := 1
			for dimension in 0 .. rank {
				if reduced[dimension] {
					if keepdims {
						output_index[output_dimension] = 0
						output_dimension++
					}
				} else {
					output_index[output_dimension] = input_index[dimension]
					output_dimension++
				}
			}
			for quantile_index, value in results {
				output_index[0] = quantile_index
				result.set(output_index, value)
			}
		} else {
			mut output_dimension := 0
			for dimension in 0 .. rank {
				if reduced[dimension] {
					if keepdims {
						output_index[output_dimension] = 0
						output_dimension++
					}
				} else {
					output_index[output_dimension] = input_index[dimension]
					output_dimension++
				}
			}
			result.set(output_index, results[0])
		}
	}
	return result
}

fn decode_quantile_multi_slice(line int, shape []int, reduced []bool, mut index []int) {
	mut remainder := line
	for dimension := shape.len - 1; dimension >= 0; dimension-- {
		if reduced[dimension] {
			index[dimension] = 0
		} else {
			index[dimension] = remainder % shape[dimension]
			remainder /= shape[dimension]
		}
	}
}

fn decode_quantile_reduced_index(line int, shape []int, reduced []bool, mut index []int) {
	mut remainder := line
	for dimension := shape.len - 1; dimension >= 0; dimension-- {
		if reduced[dimension] {
			index[dimension] = remainder % shape[dimension]
			remainder /= shape[dimension]
		}
	}
}
