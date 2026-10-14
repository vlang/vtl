module core

import math
import vtl

fn test_masked_weighted_average_ignores_masked_values_and_weights() ! {
	values := vtl.from_array[f64]([1, 100, 3, 4, 5, 6, 7, 8], [2, 4])!
	mask := vtl.from_array[bool]([false, true, false, false, true, true, true, true], [
		2,
		4,
	])!
	data := vtl.masked_array(values, mask)!
	weights := vtl.from_1d([1.0, 2, 3, 4])!
	result := data.weighted_average(weights)!
	assert !result.is_masked
	assert math.abs(result.value - 3.25) < 1e-12

	all_masked := vtl.masked_array(values, vtl.ones[bool](values.shape))!
	masked_result := all_masked.weighted_average(weights)!
	assert masked_result.is_masked
	assert math.is_nan(masked_result.value)
}

fn test_masked_weighted_average_reduces_axes_and_keeps_empty_slices_masked() ! {
	values := vtl.from_array[f64]([1, 100, 3, 4, 5, 6, 7, 8], [2, 4])!
	mask := vtl.from_array[bool]([false, true, false, false, true, true, true, true], [
		2,
		4,
	])!
	data := vtl.masked_array(values, mask)!
	weights := vtl.from_1d([1.0, 2, 3, 4])!
	rows := data.weighted_average_along_axis(weights, -1, true)!
	assert rows.values.shape == [2, 1]
	assert math.abs(rows.values.to_array()[0] - 3.25) < 1e-12
	assert math.is_nan(rows.values.to_array()[1])
	assert rows.mask.to_array() == [false, true]

	all_axes := data.weighted_average_along_axes(weights, [0, 1], false)!
	assert all_axes.values.shape == []
	assert math.abs(all_axes.values.to_array()[0] - 3.25) < 1e-12
	assert all_axes.mask.to_array() == [false]
}

fn test_masked_weighted_average_validates_shapes_and_zero_weight_sums() ! {
	values := vtl.from_1d([1.0, 2, 3])!
	mask := vtl.zeros[bool](values.shape)
	data := vtl.masked_array(values, mask)!
	if _ := data.weighted_average(vtl.from_1d([1.0, 2])!) {
		assert false, 'incompatible weights must be rejected'
	}
	if _ := data.weighted_average(vtl.from_1d([1.0, -1, 0])!) {
		assert false, 'zero total valid weight must be rejected'
	}
	if _ := data.weighted_average_along_axis(vtl.from_1d([1.0, -1, 0])!, 0, false) {
		assert false, 'zero total weight in a valid slice must be rejected'
	}
	if _ := data.weighted_average_along_axes(vtl.from_1d([1.0, 2])!, [0], false) {
		assert false, 'incompatible axis weights must be rejected'
	}
}
