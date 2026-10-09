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

fn weighted_quantile_value_compare(a &WeightedQuantileValue, b &WeightedQuantileValue) int {
	if a.value < b.value {
		return -1
	}
	if a.value > b.value {
		return 1
	}
	return 0
}
