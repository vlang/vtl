module main

import vtl
import vtl.ml.metrics

fn main() {
	y_true := vtl.from_1d([0.9, 0.2, 0.1, 0.4, 0.9]) or { panic(err) }
	y := vtl.from_1d([1.0, 0.0, 0.0, 1.0, 1.0]) or { panic(err) }

	mae := metrics.mean_absolute_error(y, y_true) or { panic(err) }
	mse := metrics.mean_squared_error(y, y_true) or { panic(err) }
	mre := metrics.mean_relative_error(y, y_true) or { panic(err) }

	println('mae: ${mae:.6f}')
	println('mse: ${mse:.6f}')
	println('mre: ${mre:.6f}')

	abs_err := metrics.absolute_error(y, y_true) or { panic(err) }
	sq_err := metrics.squared_error(y, y_true) or { panic(err) }
	rel_err := metrics.relative_error(y, y_true) or { panic(err) }

	println('absolute_error: ${abs_err}')
	println('squared_error: ${sq_err}')
	println('relative_error: ${rel_err}')
}
