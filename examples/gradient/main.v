module main

import vtl
import vtl.stats

fn main() {
	// Samples from y = x² at x = 0, 1, 2, 3, 4.
	x := vtl.from_1d([0.0, 1.0, 2.0, 3.0, 4.0])!
	y := vtl.from_1d([0.0, 1.0, 4.0, 9.0, 16.0])!
	dy_dx := stats.gradient_axis[f64](y, 1.0, 0)!
	println('x:      ${x}')
	println('y:      ${y}')
	println('dy/dx:  ${dy_dx}')

	// Non-uniform samples from y = x² use second-order boundary differences.
	x_nonuniform := [0.0, 1.0, 3.0, 6.0]
	y_nonuniform := vtl.from_1d([0.0, 1.0, 9.0, 36.0])!
	dy_nonuniform := stats.gradient_axis_with_coordinates[f64](y_nonuniform, x_nonuniform, 0, 2)!
	println('non-uniform x: ${x_nonuniform}')
	println('non-uniform dy/dx: ${dy_nonuniform}')
}
