module main

import math
import vtl
import vtl.stats

fn main() {
	volume := vtl.from_array[f64]([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])!
	means := stats.mean_along_axes(volume, [0, -1], false)!
	variances := stats.variance_along_axes(volume, [0, 2], 0, true)!
	assert means.shape == [2]
	assert means.to_array() == [3.5, 5.5]
	assert variances.shape == [1, 2, 1]
	assert variances.to_array() == [4.25, 4.25]

	with_nan := vtl.from_array[f64]([1, math.nan(), 3, 4, 5, 6, 7, 8], [2, 2, 2])!
	nan_means := stats.nanmean_along_axes(with_nan, [0, 2], false)!
	assert nan_means.to_array() == [4.0, 5.5]
	println('Multi-axis means: ${means.to_array()}')
	println('Multi-axis variances: ${variances.to_array()}')
	println('NaN-ignoring means: ${nan_means.to_array()}')
}
