module core

import math
import vtl

fn test_masked_quantile_and_median_ignore_masked_payloads() ! {
	values := vtl.from_1d([1.0, 99, 3, 5])!
	mask := vtl.from_1d[bool]([false, true, false, false])!
	data := vtl.masked_array(values, mask)!
	assert data.quantile(0.0)!.value == 1.0
	assert data.quantile(0.5)!.value == 3.0
	assert data.quantile(1.0)!.value == 5.0
	assert data.percentile(25)!.value == 2.0
	assert data.percentile(75)!.value == 4.0
	assert data.median().value == 3.0
	assert !data.median().is_masked

	integer_data := vtl.masked_array(vtl.from_1d([1, 2, 3, 4])!, vtl.zeros[bool]([4]))!
	assert integer_data.median().value == 2.5

	all_masked := vtl.masked_array(values, vtl.ones[bool](values.shape))!
	result := all_masked.quantile(0.5)!
	assert result.is_masked
	assert math.is_nan(result.value)
}

fn test_masked_quantile_methods_match_numpy_estimators() ! {
	values := vtl.from_1d([1.0, 99, 3, 5])!
	mask := vtl.from_1d[bool]([false, true, false, false])!
	data := vtl.masked_array(values, mask)!
	methods := [vtl.QuantileMethod.inverted_cdf, .averaged_inverted_cdf, .closest_observation,
		.interpolated_inverted_cdf, .hazen, .weibull, .linear, .median_unbiased, .normal_unbiased,
		.lower, .higher, .midpoint, .nearest]
	expected := [1.0, 1.0, 1.0, 1.0, 1.5, 1.0, 2.0, 1.3333333333333335, 1.375, 1.0, 3.0, 2.0, 1.0]
	for i, method in methods {
		result := data.quantile_with_method(0.25, method)!
		assert !result.is_masked
		assert math.abs(result.value - expected[i]) < 1e-12
		assert data.quantile_with_method(0, method)!.value == 1.0
		assert data.quantile_with_method(1, method)!.value == 5.0
	}
	assert data.percentile_with_method(25, .higher)!.value == 3.0
	rows := data.quantile_along_axis_with_method(0, 0.25, .higher, true)!
	assert rows.values.shape == [1]
	assert rows.values.to_array() == [3.0]
	assert rows.mask.to_array() == [false]
}

fn test_masked_quantiles_reduce_axis_and_preserve_empty_slices() ! {
	values := vtl.from_array[f64]([1, 2, 3, 4, 5, 6], [2, 3])!
	mask := vtl.from_array[bool]([false, true, false, true, true, true], [2, 3])!
	data := vtl.masked_array(values, mask)!
	rows := data.quantile_along_axis(1, 0.5, true)!
	assert rows.values.shape == [2, 1]
	assert rows.values.to_array()[0] == 2.0
	assert rows.mask.to_array() == [false, true]
	assert math.is_nan(rows.values.to_array()[1])
	percentile_rows := data.percentile_along_axis(1, 25, false)!
	assert percentile_rows.values.to_array()[0] == 1.5
	assert percentile_rows.mask.to_array() == [false, true]

	squeezed := data.median_along_axis(1, false)!
	assert squeezed.values.shape == [2]
	assert squeezed.mask.to_array() == [false, true]
}

fn test_masked_quantiles_reduce_multiple_axes_and_empty_axis_list() ! {
	values := vtl.from_array[f64]([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])!
	mask := vtl.from_array[bool]([false, true, false, true, true, true, true, false], [
		2,
		2,
		2,
	])!
	data := vtl.masked_array(values, mask)!
	result := data.quantile_along_axes([0, -1], 0.5, true)!
	assert result.values.shape == [1, 2, 1]
	assert result.values.to_array() == [1.0, 5.5]
	assert result.mask.to_array() == [false, false]
	percentile_result := data.percentile_along_axes([0, -1], 50, true)!
	assert percentile_result.values.to_array() == result.values.to_array()
	assert percentile_result.mask.to_array() == result.mask.to_array()
	all_axes := data.quantile_along_axes([0, 1, 2], 0.5, false)!
	assert all_axes.values.shape == []
	assert all_axes.values.to_array() == [3.0]
	assert all_axes.mask.to_array() == [false]

	unchanged := data.quantile_along_axes([], 0.25, false)!
	assert unchanged.values.shape == values.shape
	assert unchanged.values.to_array() == values.as_f64().to_array()
	assert unchanged.mask.to_array() == mask.to_array()
}

fn test_masked_quantile_validates_q_axes_and_propagates_unmasked_nan() ! {
	values := vtl.from_1d([1.0, math.nan(), 3.0])!
	mask := vtl.from_1d[bool]([false, false, true])!
	data := vtl.masked_array(values, mask)!
	global := data.quantile(0.5)!
	assert !global.is_masked
	assert math.is_nan(global.value)

	for q in [-0.1, 1.1, math.nan(), math.inf(1)] {
		if _ := data.quantile(q) {
			assert false, 'quantile must reject q outside [0, 1] and non-finite q'
		}
	}
	for percentile in [-1, 101, math.nan(), math.inf(1)] {
		if _ := data.percentile(percentile) {
			assert false, 'percentile must reject values outside [0, 100] and non-finite inputs'
		}
	}
	if _ := data.median_along_axis(2, false) {
		assert false, 'median must reject an out-of-bounds axis'
	}
	if _ := data.quantile_along_axes([0, -3], 0.5, false) {
		assert false, 'quantile must reject repeated axes'
	}
}
