module stats

import math
import vtl

fn test_multi_axis_moments_match_output_shapes_and_values() ! {
	values := vtl.from_array[f64]([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])!
	means := mean_along_axes(values, [0, -1], false)!
	assert means.shape == [2]
	assert means.to_array() == [3.5, 5.5]
	means_keepdims := mean_along_axes(values, [0, 2], true)!
	assert means_keepdims.shape == [1, 2, 1]
	assert means_keepdims.to_array() == [3.5, 5.5]
	variances := variance_along_axes(values, [0, 2], 0, false)!
	assert variances.to_array() == [4.25, 4.25]
	deviations := std_along_axes(values, [0, 2], 0, true)!
	assert deviations.shape == [1, 2, 1]
	assert math.abs(deviations.get([0, 0, 0]) - math.sqrt(4.25)) < 1e-12
	assert math.abs(deviations.get([0, 1, 0]) - math.sqrt(4.25)) < 1e-12
}

fn test_nan_moments_along_multiple_axes_and_empty_axes() ! {
	values := vtl.from_array[f64]([1, math.nan(), 3, 4, 5, 6, 7, 8], [2, 2, 2])!
	means := nanmean_along_axes(values, [0, 2], false)!
	assert means.to_array() == [4.0, 5.5]
	variances := nanvar_along_axes(values, [0, 2], 0, true)!
	assert variances.shape == [1, 2, 1]
	assert math.abs(variances.get([0, 0, 0]) - (14.0 / 3.0)) < 1e-12
	assert variances.get([0, 1, 0]) == 4.25
	deviations := nanstd_along_axes(values, [0, 2], 0, false)!
	assert math.abs(deviations.get([0]) - math.sqrt(14.0 / 3.0)) < 1e-12
	elementwise_means := mean_along_axes(values, [], false)!
	assert elementwise_means.shape == values.shape
	assert math.is_nan(elementwise_means.get([0, 0, 1]))
	elementwise_variances := variance_along_axes(values, [], 0, true)!
	assert elementwise_variances.shape == values.shape
	assert elementwise_variances.get([1, 1, 1]) == 0.0
}

fn test_multi_axis_moments_reject_duplicate_or_invalid_axes_and_ddof() ! {
	values := vtl.from_array[f64]([1, 2, 3, 4], [2, 2])!
	if _ := mean_along_axes(values, [0, -2], true) {
		assert false, 'multi-axis means must reject duplicate axes'
	}
	if _ := variance_along_axes(values, [2], 0, false) {
		assert false, 'multi-axis variance must reject out-of-range axes'
	}
	if _ := nanvar_along_axes(values, [0], -1, false) {
		assert false, 'multi-axis nan variance must reject negative ddof'
	}
}
