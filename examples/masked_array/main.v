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

	assert cleaned.to_array() == [12.0, -1.0, 15.0, 18.0, -1.0, 21.0]
	assert observed.to_array() == [12.0, 15.0, 18.0, 21.0]
	assert rows.values.to_array() == [27.0, 39.0]
	assert rows.mask.to_array() == [false, false]
	assert averages.values.to_array()[0] == 15.0
	assert math.is_nan(averages.values.to_array()[1])
	assert averages.values.to_array()[2] == 18.0
	assert averages.mask.to_array() == [false, true, false]
	println('Observed values: ${observed.to_array()}')
	println('Row sums: ${rows.values.to_array()}')
	println('Column means: ${averages.values.to_array()}')
}
