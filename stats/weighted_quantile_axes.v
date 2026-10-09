module stats

import math
import vtl

// quantile_weighted_axes reduces multiple axes with same-shape weights or a
// compact weight tensor whose dimensions follow the supplied axes order.
pub fn quantile_weighted_axes[T](values &vtl.Tensor[T], weights &vtl.Tensor[f64], q f64, axes []int, keepdims bool) !&vtl.Tensor[f64] {
	return weighted_quantiles_multi_axis[T](values, weights, [q], axes, keepdims, false, false)
}

// nanquantile_weighted_axes reduces multiple axes while ignoring NaNs and the
// weights paired with them.
pub fn nanquantile_weighted_axes[T](values &vtl.Tensor[T], weights &vtl.Tensor[f64], q f64, axes []int, keepdims bool) !&vtl.Tensor[f64] {
	return weighted_quantiles_multi_axis[T](values, weights, [q], axes, keepdims, true, false)
}

// percentile_weighted_axes is the 0..100-scale multi-axis weighted quantile.
pub fn percentile_weighted_axes[T](values &vtl.Tensor[T], weights &vtl.Tensor[f64], percentile f64, axes []int, keepdims bool) !&vtl.Tensor[f64] {
	return quantile_weighted_axes[T](values, weights, percentile_quantile(percentile)!, axes,
		keepdims)
}

// nanpercentile_weighted_axes is the NaN-aware 0..100-scale multi-axis form.
pub fn nanpercentile_weighted_axes[T](values &vtl.Tensor[T], weights &vtl.Tensor[f64], percentile f64, axes []int, keepdims bool) !&vtl.Tensor[f64] {
	return nanquantile_weighted_axes[T](values, weights, percentile_quantile(percentile)!,
		axes, keepdims)
}

// quantiles_weighted_axes computes several weighted quantiles per multi-axis
// slice. The quantile dimension is prepended to the output shape.
pub fn quantiles_weighted_axes[T](values &vtl.Tensor[T], weights &vtl.Tensor[f64], quantiles []f64, axes []int) !&vtl.Tensor[f64] {
	return weighted_quantiles_multi_axis[T](values, weights, quantiles, axes, false, false, true)
}

// nanquantiles_weighted_axes computes multiple NaN-aware weighted quantiles
// per multi-axis slice.
pub fn nanquantiles_weighted_axes[T](values &vtl.Tensor[T], weights &vtl.Tensor[f64], quantiles []f64, axes []int) !&vtl.Tensor[f64] {
	return weighted_quantiles_multi_axis[T](values, weights, quantiles, axes, false, true, true)
}

// percentiles_weighted_axes computes multi-axis percentiles on the 0..100 scale.
pub fn percentiles_weighted_axes[T](values &vtl.Tensor[T], weights &vtl.Tensor[f64], percentiles []f64, axes []int) !&vtl.Tensor[f64] {
	return quantiles_weighted_axes[T](values, weights, percentiles_to_quantiles(percentiles)!,
		axes)
}

// nanpercentiles_weighted_axes computes NaN-aware multi-axis percentiles.
pub fn nanpercentiles_weighted_axes[T](values &vtl.Tensor[T], weights &vtl.Tensor[f64], percentiles []f64, axes []int) !&vtl.Tensor[f64] {
	return nanquantiles_weighted_axes[T](values, weights, percentiles_to_quantiles(percentiles)!,
		axes)
}

fn weighted_quantiles_multi_axis[T](values &vtl.Tensor[T], weights &vtl.Tensor[f64], quantiles []f64, axes []int, keepdims bool, ignore_nan bool, multi bool) !&vtl.Tensor[f64] {
	validate_quantiles(quantiles)!
	rank := values.rank()
	if rank == 0 || axes.len == 0 {
		return error('weighted multi-axis quantile requires a ranked tensor and at least one axis')
	}
	mut normalized_axes := []int{cap: axes.len}
	mut reduced := []bool{len: rank}
	mut reduced_shape := []int{cap: axes.len}
	for axis in axes {
		axis_index := if axis < 0 { axis + rank } else { axis }
		if axis_index < 0 || axis_index >= rank {
			return error('weighted quantile axis ${axis} out of bounds for rank ${rank}')
		}
		if reduced[axis_index] {
			return error('weighted quantile axis ${axis} appears more than once')
		}
		if values.shape[axis_index] == 0 {
			return error('weighted quantile is undefined for an empty axis')
		}
		reduced[axis_index] = true
		normalized_axes << axis_index
		reduced_shape << values.shape[axis_index]
	}
	if weights.shape != values.shape && weights.shape != reduced_shape {
		return error('weights must match the values shape or the reduced axes shape in axes order')
	}
	mut output_shape := []int{cap: rank + 1}
	if multi {
		output_shape << quantiles.len
	}
	for dimension, size in values.shape {
		if reduced[dimension] {
			if keepdims && !multi {
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
	mut input_index := []int{len: rank}
	mut reduced_index := []int{len: axes.len}
	mut output_index := []int{len: output_shape.len}
	mut slice_count := 1
	mut reduction_count := 1
	for dimension, size in values.shape {
		if reduced[dimension] {
			reduction_count *= size
		} else {
			slice_count *= size
		}
	}
	for slice in 0 .. slice_count {
		decode_weighted_multi_slice(slice, values.shape, reduced, mut input_index)
		mut pairs := []WeightedQuantileValue{cap: reduction_count}
		mut total_weight := 0.0
		mut valid_count := 0
		mut has_nan := false
		for flat_reduced in 0 .. reduction_count {
			decode_weighted_reduced_index(flat_reduced, values.shape, normalized_axes, mut input_index, mut reduced_index)
			value := f64(values.get(input_index))
			weight := if weights.shape == values.shape {
				weights.get(input_index)
			} else {
				weights.get(reduced_index)
			}
			validate_weight(weight)!
			if math.is_nan(value) {
				has_nan = true
				if !ignore_nan {
					total_weight += weight
				}
				continue
			}
			valid_count++
			total_weight += weight
			if weight > 0 {
				pairs << WeightedQuantileValue{value, weight}
			}
		}
		if !(ignore_nan && valid_count == 0) {
			validate_weight_total(total_weight)!
		}
		mut values_at_quantiles := []f64{len: quantiles.len, init: math.nan()}
		if !(has_nan && !ignore_nan) && !(ignore_nan && valid_count == 0) {
			pairs.sort_with_compare(weighted_quantile_value_compare)
			if !multi {
				values_at_quantiles[0] = weighted_quantile_sorted(pairs, quantiles[0], total_weight)
			} else {
				values_at_quantiles = weighted_quantiles_sorted(pairs, quantiles, total_weight)
			}
		}
		decode_weighted_multi_slice(slice, values.shape, reduced, mut input_index)
		if multi {
			output_index[0] = 0
			mut output_dimension := 1
			for dimension in 0 .. rank {
				if !reduced[dimension] {
					output_index[output_dimension] = input_index[dimension]
					output_dimension++
				}
			}
			for quantile_index, quantile in values_at_quantiles {
				output_index[0] = quantile_index
				result.set(output_index, quantile)
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
			if output_index.len == 0 {
				result.set_nth(0, values_at_quantiles[0])
			} else {
				result.set(output_index, values_at_quantiles[0])
			}
		}
	}
	return result
}

fn decode_weighted_multi_slice(line int, shape []int, reduced []bool, mut index []int) {
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

fn decode_weighted_reduced_index(line int, shape []int, axes []int, mut input_index []int, mut reduced_index []int) {
	mut remainder := line
	for axis_position := axes.len - 1; axis_position >= 0; axis_position-- {
		axis := axes[axis_position]
		size := shape[axis]
		coordinate := remainder % size
		remainder /= size
		input_index[axis] = coordinate
		reduced_index[axis_position] = coordinate
	}
}
