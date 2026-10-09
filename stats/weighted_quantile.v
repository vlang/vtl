module stats

import math
import vtl

struct WeightedQuantileValue {
	value  f64
	weight f64
}

// quantile_weighted computes NumPy's weighted inverted-CDF quantile.
// Values and weights must have identical shapes; weights must be finite,
// non-negative, and have a positive total.
pub fn quantile_weighted[T](values &vtl.Tensor[T], weights &vtl.Tensor[f64], q f64) !f64 {
	validate_quantile(q)!
	if values.shape != weights.shape {
		return error('quantile_weighted: values and weights must have the same shape')
	}
	if values.size == 0 {
		return error('quantile_weighted: input must not be empty')
	}
	mut pairs := []WeightedQuantileValue{cap: values.size}
	mut total_weight := 0.0
	for i in 0 .. values.size {
		value := f64(values.get_nth(i))
		weight := weights.get_nth(i)
		if math.is_nan(value) {
			return math.nan()
		}
		if math.is_nan(weight) || math.is_inf(weight, 0) || weight < 0 {
			return error('quantile_weighted: weights must be finite and non-negative')
		}
		total_weight += weight
		if weight > 0 {
			pairs << WeightedQuantileValue{value, weight}
		}
	}
	if total_weight <= 0 || math.is_inf(total_weight, 0) {
		return error('quantile_weighted: weights must have a finite positive sum')
	}
	pairs.sort_with_compare(weighted_quantile_value_compare)
	target := q * total_weight
	mut cumulative := 0.0
	for pair in pairs {
		cumulative += pair.weight
		if cumulative >= target {
			return pair.value
		}
	}
	return pairs[pairs.len - 1].value
}

// quantiles_weighted computes several weighted inverted-CDF quantiles after
// sorting the weighted sample once.
pub fn quantiles_weighted[T](values &vtl.Tensor[T], weights &vtl.Tensor[f64], quantiles []f64) !&vtl.Tensor[f64] {
	validate_quantiles(quantiles)!
	if values.shape != weights.shape {
		return error('quantiles_weighted: values and weights must have the same shape')
	}
	if values.size == 0 {
		return error('quantiles_weighted: input must not be empty')
	}
	mut pairs := []WeightedQuantileValue{cap: values.size}
	mut total_weight := 0.0
	for i in 0 .. values.size {
		value := f64(values.get_nth(i))
		weight := weights.get_nth(i)
		if math.is_nan(value) {
			return vtl.from_1d[f64]([]f64{len: quantiles.len, init: math.nan()})
		}
		if math.is_nan(weight) || math.is_inf(weight, 0) || weight < 0 {
			return error('quantiles_weighted: weights must be finite and non-negative')
		}
		total_weight += weight
		if weight > 0 {
			pairs << WeightedQuantileValue{value, weight}
		}
	}
	if total_weight <= 0 || math.is_inf(total_weight, 0) {
		return error('quantiles_weighted: weights must have a finite positive sum')
	}
	pairs.sort_with_compare(weighted_quantile_value_compare)
	mut results := []f64{len: quantiles.len}
	for index, q in quantiles {
		if q <= 0 {
			results[index] = pairs[0].value
			continue
		}
		if q >= 1 {
			results[index] = pairs[pairs.len - 1].value
			continue
		}
		target := q * total_weight
		mut cumulative := 0.0
		for pair in pairs {
			cumulative += pair.weight
			if cumulative >= target {
				results[index] = pair.value
				break
			}
		}
	}
	return vtl.from_1d[f64](results)
}

// quantile_weighted_axis reduces one axis using same-shape weights or a
// one-dimensional weight vector matching that axis.
pub fn quantile_weighted_axis[T](values &vtl.Tensor[T], weights &vtl.Tensor[f64], q f64, axis int, keepdims bool) !&vtl.Tensor[f64] {
	validate_quantile(q)!
	axis_index := validate_weighted_axis(values, weights, axis)!
	rank := values.rank()
	axis_size := values.shape[axis_index]
	mut output_shape := values.shape.clone()
	if keepdims {
		output_shape[axis_index] = 1
	} else {
		output_shape.delete(axis_index)
		if output_shape.len == 0 {
			output_shape = [1]
		}
	}
	mut result := vtl.empty[f64](output_shape, memory: .row_major)
	mut index := []int{len: rank}
	slice_count := weighted_axis_slice_count(values.shape, axis_index)
	for slice in 0 .. slice_count {
		decode_nan_slice(slice, values.shape, axis_index, mut index)
		mut pairs := []WeightedQuantileValue{cap: axis_size}
		mut total_weight := 0.0
		mut has_nan := false
		for position in 0 .. axis_size {
			index[axis_index] = position
			value := f64(values.get(index))
			weight := weighted_axis_weight(weights, index, position)
			validate_weight(weight)!
			total_weight += weight
			if math.is_nan(value) {
				has_nan = true
			} else if weight > 0 {
				pairs << WeightedQuantileValue{value, weight}
			}
		}
		validate_weight_total(total_weight)!
		quantile := if has_nan {
			math.nan()
		} else {
			pairs.sort_with_compare(weighted_quantile_value_compare)
			weighted_quantile_sorted(pairs, q, total_weight)
		}
		if keepdims {
			index[axis_index] = 0
			result.set(index, quantile)
		} else {
			mut output_index := index[..axis_index].clone()
			output_index << index[axis_index + 1..]
			if output_index.len == 0 {
				output_index = [0]
			}
			result.set(output_index, quantile)
		}
	}
	return result
}

// quantiles_weighted_axis computes several weighted quantiles per axis slice.
// The quantile dimension is prepended, matching NumPy's multi-quantile shape.
pub fn quantiles_weighted_axis[T](values &vtl.Tensor[T], weights &vtl.Tensor[f64], quantiles []f64, axis int) !&vtl.Tensor[f64] {
	validate_quantiles(quantiles)!
	axis_index := validate_weighted_axis(values, weights, axis)!
	rank := values.rank()
	axis_size := values.shape[axis_index]
	mut output_shape := [quantiles.len]
	for dimension, size in values.shape {
		if dimension != axis_index {
			output_shape << size
		}
	}
	mut result := vtl.empty[f64](output_shape, memory: .row_major)
	mut index := []int{len: rank}
	mut output_index := []int{len: rank}
	slice_count := weighted_axis_slice_count(values.shape, axis_index)
	for slice in 0 .. slice_count {
		decode_nan_slice(slice, values.shape, axis_index, mut index)
		mut pairs := []WeightedQuantileValue{cap: axis_size}
		mut total_weight := 0.0
		mut has_nan := false
		for position in 0 .. axis_size {
			index[axis_index] = position
			value := f64(values.get(index))
			weight := weighted_axis_weight(weights, index, position)
			validate_weight(weight)!
			total_weight += weight
			if math.is_nan(value) {
				has_nan = true
			} else if weight > 0 {
				pairs << WeightedQuantileValue{value, weight}
			}
		}
		validate_weight_total(total_weight)!
		output_index[0] = 0
		mut output_dimension := 1
		for dimension in 0 .. rank {
			if dimension != axis_index {
				output_index[output_dimension] = index[dimension]
				output_dimension++
			}
		}
		if has_nan {
			for quantile_index in 0 .. quantiles.len {
				output_index[0] = quantile_index
				result.set(output_index, math.nan())
			}
		} else {
			pairs.sort_with_compare(weighted_quantile_value_compare)
			for quantile_index, q in quantiles {
				output_index[0] = quantile_index
				result.set(output_index, weighted_quantile_sorted(pairs, q, total_weight))
			}
		}
	}
	return result
}

fn validate_weighted_axis[T](values &vtl.Tensor[T], weights &vtl.Tensor[f64], axis int) !int {
	rank := values.rank()
	if rank == 0 {
		return error('weighted quantile axis requires a tensor with at least one dimension')
	}
	axis_index := if axis < 0 { axis + rank } else { axis }
	if axis_index < 0 || axis_index >= rank {
		return error('weighted quantile axis ${axis} out of bounds for rank ${rank}')
	}
	if values.shape[axis_index] == 0 {
		return error('weighted quantile is undefined for an empty axis')
	}
	if weights.shape != values.shape && weights.shape != [values.shape[axis_index]] {
		return error('weights must match the values shape or the reduced axis length')
	}
	return axis_index
}

fn weighted_axis_slice_count(shape []int, axis int) int {
	mut count := 1
	for dimension, size in shape {
		if dimension != axis {
			count *= size
		}
	}
	return count
}

fn weighted_axis_weight(weights &vtl.Tensor[f64], index []int, position int) f64 {
	return if weights.rank() == 1 { weights.get([position]) } else { weights.get(index) }
}

fn validate_weight(weight f64) ! {
	if math.is_nan(weight) || math.is_inf(weight, 0) || weight < 0 {
		return error('weights must be finite and non-negative')
	}
}

fn validate_weight_total(total_weight f64) ! {
	if total_weight <= 0 || math.is_inf(total_weight, 0) {
		return error('weights must have a finite positive sum')
	}
}

fn weighted_quantile_sorted(pairs []WeightedQuantileValue, q f64, total_weight f64) f64 {
	if q <= 0 {
		return pairs[0].value
	}
	if q >= 1 {
		return pairs[pairs.len - 1].value
	}
	target := q * total_weight
	mut cumulative := 0.0
	for pair in pairs {
		cumulative += pair.weight
		if cumulative >= target {
			return pair.value
		}
	}
	return pairs[pairs.len - 1].value
}

fn weighted_quantile_value_compare(a &WeightedQuantileValue, b &WeightedQuantileValue) int {
	if a.value < b.value {
		return -1
	}
	if a.value > b.value {
		return 1
	}
	return 0
}
