module main

import math
import vtl
import vtl.stats

fn main() {
	samples := vtl.from_array([1.0, math.nan(), 3.0, 4.0, math.nan(), 6.0], [2, 3])!
	println('NaN-aware sum: ${stats.nansum(samples)}')
	println('NaN-aware product: ${stats.nanprod(samples)}')
	println('Row minima: ${stats.nanmin_axis(samples, 1)!.to_array()}')
	println('Row maxima (keepdims): ${stats.nanmax_axis_keepdims(samples, 1)!.to_array()}')
}
