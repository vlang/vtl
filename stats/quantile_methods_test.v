module stats

import math
import vtl

fn test_quantile_methods_match_numpy_estimator_examples() ! {
	values := vtl.from_1d([1.0, 2.0, 3.0, 4.0])!
	methods := [QuantileMethod.inverted_cdf, .averaged_inverted_cdf, .closest_observation,
		.interpolated_inverted_cdf, .hazen, .weibull, .linear, .median_unbiased, .normal_unbiased,
		.lower, .higher, .midpoint, .nearest]
	expected := [1.0, 1.5, 1.0, 1.0, 1.5, 1.25, 1.75, 1.4166666666666667, 1.4375, 1.0, 2.0, 1.5,
		2.0]
	for i, method in methods {
		actual := quantile_with_method(values, 0.25, method)!
		assert math.abs(actual - expected[i]) < 1e-12, '${method}: got ${actual}, expected ${expected[i]}'
	}
	assert quantile_with_method(values, 0.5, .closest_observation)! == 2.0
	assert quantile_with_method(values, 0.5, .nearest)! == 2.0
}

fn test_quantile_methods_nan_axis_and_percentile_apis() ! {
	values := vtl.from_array([1.0, math.nan(), 3.0, 5.0, 7.0, 9.0], [2, 3])!
	ordinary := quantile_axis_with_method(values, 0.5, 1, .linear, false)!
	assert ordinary.shape == [2]
	assert math.is_nan(ordinary.get([0]))
	assert ordinary.get([1]) == 7.0
	ignored := nanquantiles_axis_with_method(values, [0.0, 0.5, 1.0], 1, .linear)!
	assert ignored.shape == [3, 2]
	assert ignored.to_array() == [1.0, 5.0, 2.0, 7.0, 3.0, 9.0]
	assert percentile_with_method(vtl.from_1d([1, 2, 3, 4])!, 25, .linear)! == 1.75
	assert percentiles_with_method(vtl.from_1d([1, 2, 3, 4])!, [25, 75], .linear)!.to_array() == [
		1.75,
		3.25,
	]
}

fn test_quantile_method_keepdims_validation_and_nan_only() ! {
	values := vtl.from_array([1.0, 2.0, 3.0, 4.0], [2, 2])!
	kept := quantile_axis_with_method(values, 0.5, 1, .linear, true)!
	assert kept.shape == [2, 1]
	assert kept.to_array() == [1.5, 3.5]
	all_nan := vtl.from_1d([math.nan(), math.nan()])!
	assert math.is_nan(nanquantile_with_method(all_nan, 0.5, .linear)!)
	if _ := quantile_with_method(values, 1.1, .linear) {
		assert false, 'out-of-range quantile must return an error'
	} else {
		assert true
	}
}

fn test_quantile_methods_reduce_multiple_axes_and_preserve_shapes() ! {
	values := vtl.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2])!
	medians := quantile_axes_with_method(values, 0.5, [0, -1], .linear, false)!
	assert medians.shape == [2]
	assert medians.to_array() == [3.5, 5.5]
	levels := quantiles_axes_with_method(values, [0.5, 0.0, 1.0], [0, 2], .linear)!
	assert levels.shape == [3, 2]
	assert levels.to_array() == [3.5, 5.5, 1.0, 3.0, 6.0, 8.0]
	percentiles := percentiles_axes_with_method(values, [0, 25, 100], [0, 2], .linear)!
	assert percentiles.to_array() == [1.0, 3.0, 1.75, 3.75, 6.0, 8.0]
	kept := quantile_axes_with_method(values, 0.5, [0, 2], .linear, true)!
	assert kept.shape == [1, 2, 1]
	assert kept.to_array() == [3.5, 5.5]
	if _ := quantile_axes_with_method(values, 0.5, [1, 1], .linear, false) {
		assert false, 'duplicate axes must fail'
	}
}

fn test_nanquantile_methods_reduce_multiple_axes() ! {
	values := vtl.from_array([1.0, math.nan(), 3.0, 4.0, 5.0, 6.0, math.nan(), 8.0], [
		2,
		2,
		2,
	])!
	ordinary := quantile_axes_with_method(values, 0.5, [0, 2], .linear, false)!
	assert math.is_nan(ordinary.get([0]))
	assert math.is_nan(ordinary.get([1]))
	ignored := nanquantile_axes_with_method(values, 0.5, [0, 2], .linear, false)!
	assert ignored.to_array() == [5.0, 4.0]
	assert nanpercentiles_axes_with_method(values, [0, 50, 100], [0, 2], .linear)!.to_array() == [
		1.0,
		3.0,
		5.0,
		4.0,
		6.0,
		8.0,
	]
}

fn test_default_multi_axis_quantile_convenience_apis() ! {
	values := vtl.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2])!
	assert quantile_axes(values, 0.5, [0, 2], false)!.to_array() == [3.5, 5.5]
	assert quantiles_axes(values, [0.0, 0.5, 1.0], [0, 2])!.to_array() == [1.0, 3.0, 3.5, 5.5,
		6.0, 8.0]
	assert percentile_axes(values, 50, [0, 2], true)!.shape == [1, 2, 1]
	assert percentiles_axes(values, [0, 50, 100], [0, 2])!.to_array() == [1.0, 3.0, 3.5, 5.5, 6.0,
		8.0]
	nan_values := vtl.from_array([1.0, math.nan(), 3.0, 4.0, 5.0, 6.0, math.nan(), 8.0], [
		2,
		2,
		2,
	])!
	assert nanquantile_axes(nan_values, 0.5, [0, 2], false)!.to_array() == [5.0, 4.0]
	assert nanquantiles_axes(nan_values, [0.0, 1.0], [0, 2])!.to_array() == [1.0, 3.0, 6.0, 8.0]
	assert nanpercentile_axes(nan_values, 50, [0, 2], false)!.to_array() == [5.0, 4.0]
	assert nanpercentiles_axes(nan_values, [0, 100], [0, 2])!.to_array() == [1.0, 3.0, 6.0, 8.0]
}

fn test_multi_quantile_keepdims_shapes_and_values() ! {
	values := vtl.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [2, 2, 2])!
	levels := quantiles_axes_with_method_keepdims(values, [0.0, 0.5, 1.0], [0, 2], .linear,
		true)!
	assert levels.shape == [3, 1, 2, 1]
	assert levels.to_array() == [1.0, 3.0, 3.5, 5.5, 6.0, 8.0]
	assert quantiles_axes_keepdims(values, [0.0, 0.5, 1.0], [0, 2], true)!.shape == [
		3,
		1,
		2,
		1,
	]
	assert percentiles_axes_with_method_keepdims(values, [0, 50, 100], [0, 2], .linear,
		true)!.shape == [3, 1, 2, 1]
	assert percentiles_axes_keepdims(values, [0, 50, 100], [0, 2], true)!.to_array() == [
		1.0,
		3.0,
		3.5,
		5.5,
		6.0,
		8.0,
	]
	nan_values := vtl.from_array([1.0, math.nan(), 3.0, 4.0, 5.0, 6.0, math.nan(), 8.0], [
		2,
		2,
		2,
	])!
	assert nanquantiles_axes_with_method_keepdims(nan_values, [0.0, 1.0], [0, 2], .linear,
		true)!.shape == [2, 1, 2, 1]
	assert nanquantiles_axes_keepdims(nan_values, [0.0, 1.0], [0, 2], true)!.to_array() == [
		1.0,
		3.0,
		6.0,
		8.0,
	]
	assert nanpercentiles_axes_with_method_keepdims(nan_values, [0, 100], [0, 2], .linear,
		true)!.shape == [2, 1, 2, 1]
	assert nanpercentiles_axes_keepdims(nan_values, [0, 100], [0, 2], true)!.to_array() == [
		1.0,
		3.0,
		6.0,
		8.0,
	]
}
