module main

import vtl
import vtl.la

fn main() {
	// NumPy-style rowvar=true: each row is a measured variable.
	observations := vtl.from_2d([
		[f64(150), 160, 170, 180],
		[50, 55, 65, 80],
	])!
	covariance := la.covariance_matrix[f64](observations, true, 1)!
	correlation := la.correlation_matrix[f64](observations, true)!
	println('Sample covariance: ${covariance.to_array()}')
	println('Pearson correlation: ${correlation.to_array()}')
}
