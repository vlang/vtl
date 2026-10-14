module stats

import math
import vtl

fn test_median_axis_keepdims_and_squeeze() ! {
	values := vtl.from_array[f64]([1, 3, 5, 7], [2, 2])!
	kept := median_axis_keepdims(values, 1, true)!
	assert kept.shape == [2, 1]
	assert kept.to_array() == [2.0, 6.0]

	default_shape := median_axis(values, 1)!
	assert default_shape.shape == [2, 1]
	assert median_axis_squeeze(values, 1)!.shape == [2]
	assert median_axis_keepdims(values, 1, false)!.to_array() == [2.0, 6.0]
}

fn test_median_axes_keepdims_matches_linear_quantile() ! {
	volume := vtl.from_array[f64]([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])!
	kept := median_axes(volume, [0, -1], true)!
	assert kept.shape == [1, 2, 1]
	assert kept.to_array() == [3.5, 5.5]

	squeezed := median_axes(volume, [0, 2], false)!
	assert squeezed.shape == [2]
	assert squeezed.to_array() == [3.5, 5.5]
}

fn test_nanmedian_axis_and_axes_keepdims() ! {
	values := vtl.from_array[f64]([1, math.nan(), 5, 3, math.nan(), 7], [2, 3])!
	columns := nanmedian_axis_keepdims(values, 0, true)!
	assert columns.shape == [1, 3]
	assert columns.to_array()[0] == 2.0
	assert math.is_nan(columns.to_array()[1])
	assert columns.to_array()[2] == 6.0

	all_axes := nanmedian_axes(values, [0, 1], true)!
	assert all_axes.shape == [1, 1]
	assert all_axes.to_array() == [4.0]
	assert nanmedian_axis(values, 0)!.shape == [1, 3]
	assert nanmedian_axis_squeeze(values, 0)!.shape == [3]

	all_nan := vtl.from_array[f64]([math.nan(), math.nan(), math.nan(), math.nan()], [
		2,
		2,
	])!
	assert math.is_nan(nanmedian_axes(all_nan, [0, 1], true)!.to_array()[0])
}

fn test_median_axis_and_multi_axis_propagate_nan() ! {
	values := vtl.from_array[f64]([1, math.nan(), 3, 4, 5, 6], [2, 3])!

	rows := median_axis_squeeze(values, 1)!
	assert math.is_nan(rows.get_nth(0))
	assert rows.get_nth(1) == 5.0

	columns := median_axis_squeeze(values, 0)!
	assert columns.get_nth(0) == 2.5
	assert math.is_nan(columns.get_nth(1))
	assert columns.get_nth(2) == 4.5

	all_axes := median_axes(values, [0, 1], false)!
	assert all_axes.shape == [1]
	assert math.is_nan(all_axes.get_nth(0))
}
