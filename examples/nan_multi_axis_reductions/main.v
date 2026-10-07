import math
import vtl
import vtl.stats

fn main() {
	values := vtl.from_array[f64]([math.nan(), 2.0, 3.0, 4.0, 5.0, math.nan(), 7.0, 8.0], [
		2,
		2,
		2,
	]) or {
		panic(err)
	}
	sums := stats.nansum_axes(values, [0, -1], false) or { panic(err) }
	products := stats.nanprod_axes(values, [0, 2], true) or { panic(err) }
	minima := stats.nanmin_axes(values, [0, 2], false) or { panic(err) }
	maxima := stats.nanmax_axes(values, [0, 2], false) or { panic(err) }
	println('NaN-aware sums: ${sums.to_array()}')
	println('NaN-aware products: ${products.to_array()} with shape ${products.shape}')
	println('NaN-aware minima: ${minima.to_array()}')
	println('NaN-aware maxima: ${maxima.to_array()}')
}
