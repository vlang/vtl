module stats

import math
import vtl

fn test_weighted_quantiles_match_inverted_cdf_semantics() ! {
	values := vtl.from_1d([1.0, 2.0, 3.0, 4.0])!
	weights := vtl.from_1d([1.0, 1.0, 1.0, 7.0])!
	assert quantile_weighted(values, weights, 0.5)! == 4.0
	assert percentile_weighted(values, weights, 50)! == 4.0
	assert quantiles_weighted(values, weights, [0.0, 0.25, 0.5, 1.0])!.to_array() == [
		1.0,
		3.0,
		4.0,
		4.0,
	]
	assert percentiles_weighted(values, weights, [25, 50, 100])!.to_array() == [3.0, 4.0, 4.0]
	zero_weight := vtl.from_1d([0.0, 1.0, 0.0, 1.0])!
	assert quantile_weighted(values, zero_weight, 0.0)! == 2.0
	assert quantile_weighted(values, zero_weight, 0.5)! == 2.0
}

fn test_weighted_quantiles_validate_shape_and_weights() ! {
	values := vtl.from_1d([1.0, 2.0])!
	if _ := quantile_weighted(values, vtl.from_1d([1.0])!, 0.5) {
		assert false, 'mismatched weight shapes must fail'
	}
	if _ := quantile_weighted(values, vtl.from_1d([-1.0, 2.0])!, 0.5) {
		assert false, 'negative weights must fail'
	}
	if _ := quantile_weighted(values, vtl.from_1d([0.0, 0.0])!, 0.5) {
		assert false, 'zero total weight must fail'
	}
	assert quantile_weighted(values, vtl.from_1d([1.0, 1.0])!, 0.5)! == 1.0
}

fn test_weighted_quantiles_reduce_axis_with_shared_or_full_weights() ! {
	values := vtl.from_array([10.0, 7.0, 4.0, 3.0, 2.0, 1.0], [2, 3])!
	full_weights := vtl.from_array([1.0, 1.0, 7.0, 1.0, 1.0, 1.0], [2, 3])!
	by_column := quantile_weighted_axis(values, full_weights, 0.5, 0, false)!
	assert by_column.shape == [3]
	assert by_column.to_array() == [3.0, 2.0, 4.0]
	by_row := quantiles_weighted_axis(values, vtl.from_1d([1.0, 1.0, 7.0])!, [0.0, 0.5, 1.0], -1)!
	assert by_row.shape == [3, 2]
	assert by_row.to_array() == [4.0, 1.0, 4.0, 1.0, 10.0, 3.0]
	kept := quantile_weighted_axis(values, vtl.from_1d([1.0, 1.0, 7.0])!, 0.5, 1, true)!
	assert kept.shape == [2, 1]
	assert kept.to_array() == [4.0, 1.0]
}

fn test_nanweighted_quantiles_skip_values_and_their_weights() ! {
	values := vtl.from_1d([1.0, math.nan(), 3.0, 4.0])!
	weights := vtl.from_1d([1.0, 100.0, 1.0, 2.0])!
	assert nanquantile_weighted(values, weights, 0.5)! == 3.0
	assert nanpercentile_weighted(values, weights, 50)! == 3.0
	assert nanquantiles_weighted(values, weights, [0.25, 0.5, 1.0])!.to_array() == [
		1.0,
		3.0,
		4.0,
	]
	assert nanpercentiles_weighted(values, weights, [25, 50, 100])!.to_array() == [
		1.0,
		3.0,
		4.0,
	]
	all_nan := vtl.from_1d([math.nan(), math.nan()])!
	assert math.is_nan(nanquantile_weighted(all_nan, vtl.from_1d([1.0, 1.0])!, 0.5)!)
	if _ := nanquantile_weighted(vtl.from_1d([1.0, 2.0])!, vtl.from_1d([0.0, 0.0])!, 0.5) {
		assert false, 'zero total weight among non-NaN values must fail'
	}
}

fn test_nanweighted_quantile_axis_preserves_nan_policy_and_shapes() ! {
	values := vtl.from_array([1.0, math.nan(), 3.0, 4.0, 5.0, math.nan()], [2, 3])!
	weights := vtl.from_array([1.0, 10.0, 1.0, 1.0, 1.0, 1.0], [2, 3])!
	ordinary := quantile_weighted_axis(values, weights, 0.5, 1, false)!
	assert math.is_nan(ordinary.get([0]))
	assert math.is_nan(ordinary.get([1]))
	ignored := nanquantile_weighted_axis(values, weights, 0.5, 1, false)!
	assert ignored.to_array() == [1.0, 4.0]
	assert nanpercentile_weighted_axis(values, weights, 50, 1, false)!.to_array() == [
		1.0,
		4.0,
	]
	multiple := nanquantiles_weighted_axis(values, weights, [0.0, 0.5, 1.0], 1)!
	assert multiple.shape == [3, 2]
	assert multiple.to_array() == [1.0, 4.0, 1.0, 4.0, 3.0, 5.0]
	assert percentiles_weighted_axis(values, weights, [0, 50, 100], 1)!.to_array().all(math.is_nan(it))
	assert nanpercentiles_weighted_axis(values, weights, [0, 50, 100], 1)!.to_array() == [
		1.0,
		4.0,
		1.0,
		4.0,
		3.0,
		5.0,
	]
}

fn test_weighted_quantiles_reduce_multiple_axes_with_compact_weights() ! {
	values := vtl.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2])!
	weights := vtl.from_array([1.0, 1.0, 1.0, 3.0], [2, 2])!
	medians := quantile_weighted_axes(values, weights, 0.5, [0, -1], false)!
	assert medians.shape == [2]
	assert medians.to_array() == [5.0, 7.0]
	levels := quantiles_weighted_axes(values, weights, [0.5, 0.0, 1.0], [0, 2])!
	assert levels.shape == [3, 2]
	assert levels.to_array() == [5.0, 7.0, 1.0, 3.0, 6.0, 8.0]
	assert percentile_weighted_axes(values, weights, 50, [0, 2], false)!.to_array() == [
		5.0,
		7.0,
	]
	assert percentiles_weighted_axes(values, weights, [0, 50, 100], [0, 2])!.to_array() == [
		1.0,
		3.0,
		5.0,
		7.0,
		6.0,
		8.0,
	]
	kept := quantile_weighted_axes(values, vtl.ones[f64]([2, 2, 2]), 0.5, [0, 2], true)!
	assert kept.shape == [1, 2, 1]
	assert kept.to_array() == [2.0, 4.0]
	if _ := quantile_weighted_axes(values, weights, 0.5, [0, 0], false) {
		assert false, 'duplicate axes must fail'
	}
	ordered_values := vtl.from_array([
		1.0,
		2.0,
		3.0,
		4.0,
		5.0,
		6.0,
		7.0,
		8.0,
		9.0,
		10.0,
		11.0,
		12.0,
		13.0,
		14.0,
		15.0,
		16.0,
		17.0,
		18.0,
		19.0,
		20.0,
		21.0,
		22.0,
		23.0,
		24.0,
	], [2, 3, 4])!
	ordered_weights := vtl.from_array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0], [4, 2])!
	ordered_axes := quantile_weighted_axes(ordered_values, ordered_weights, 0.5, [2, 0], false)!
	assert ordered_axes.to_array() == [14.0, 18.0, 22.0]
}

fn test_nanweighted_quantiles_reduce_multiple_axes() ! {
	values := vtl.from_array([1.0, math.nan(), 3.0, 4.0, 5.0, 6.0, math.nan(), 8.0], [
		2,
		2,
		2,
	])!
	weights := vtl.ones[f64]([2, 2])
	result := nanquantile_weighted_axes(values, weights, 0.5, [0, 2], false)!
	assert result.to_array() == [5.0, 4.0]
	assert nanpercentile_weighted_axes(values, weights, 50, [0, 2], false)!.to_array() == [
		5.0,
		4.0,
	]
	assert nanpercentiles_weighted_axes(values, weights, [0, 50, 100], [0, 2])!.to_array() == [
		1.0,
		3.0,
		5.0,
		4.0,
		6.0,
		8.0,
	]
	all_nan := vtl.full[f64]([2, 2], math.nan())
	assert math.is_nan(nanquantile_weighted_axes(all_nan, vtl.ones[f64]([2, 2]), 0.5, [
		0,
		1,
	],
		false)!.get_nth(0))
}
