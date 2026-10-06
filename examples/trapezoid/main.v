module main

import vtl
import vtl.stats

fn main() {
	values := vtl.from_1d([1.0, 2.0, 3.0])!
	uniform := stats.trapezoid[f64](values, 1.0)!
	x := vtl.from_1d([4.0, 6.0, 8.0])!
	explicit := stats.trapezoid_x_axis[f64, f64](values, x, 0)!
	println('Uniform spacing: ${uniform.get_nth(0)}')
	println('Explicit coordinates: ${explicit.get_nth(0)}')
}
