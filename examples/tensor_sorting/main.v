module main

import vtl
import math

fn main() {
	values := vtl.from_array([3.0, math.nan(), -1.0, 2.0, 2.0, 0.0], [2, 3])!
	println('Values: ${values.to_array()}')
	println('Sorted by row: ${vtl.sort(values)!.to_array()}')
	println('Sort indices: ${vtl.argsort(values)!.to_array()}')
	println('Sorted by column: ${vtl.sort_axis(values, 0)!.to_array()}')
}
