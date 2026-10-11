module main

import math
import vtl

fn main() {
	measurements := vtl.from_array[f64]([12.0, math.nan(), 15.0, 18.0, math.nan(), 21.0], [
		2,
		3,
	])!
	missing := vtl.from_array[bool]([false, true, false], [3])!
	data := vtl.masked_array(measurements, missing)!

	cleaned := data.filled(-1.0)
	observed := data.compressed()!
	rows := data.sum_along_axis(1, false)!
	averages := data.mean_along_axis(0, false)!
	median := data.median()
	row_medians := data.median_along_axis(1, false)!

	assert cleaned.to_array() == [12.0, -1.0, 15.0, 18.0, -1.0, 21.0]
	assert observed.to_array() == [12.0, 15.0, 18.0, 21.0]
	assert rows.values.to_array() == [27.0, 39.0]
	assert rows.mask.to_array() == [false, false]
	assert averages.values.to_array()[0] == 15.0
	assert math.is_nan(averages.values.to_array()[1])
	assert averages.values.to_array()[2] == 18.0
	assert averages.mask.to_array() == [false, true, false]
	assert median.value == 16.5 && !median.is_masked
	assert row_medians.values.to_array() == [13.5, 19.5]
	assert row_medians.mask.to_array() == [false, false]
	println('Observed values: ${observed.to_array()}')
	println('Row sums: ${rows.values.to_array()}')
	println('Column means: ${averages.values.to_array()}')
	println('Median: ${median.value}')
	println('Row medians: ${row_medians.values.to_array()}')
}
