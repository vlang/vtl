module main

import vtl

fn main() {
	values := vtl.from_array([1, 2, 3, 4, 5, 6], [2, 3])!
	lower := vtl.from_1d([2, 2, 2])!
	upper := vtl.from_array([3, 5], [2, 1])!
	clamped := vtl.clip_tensor(values, lower, upper)!
	println('clamped values: ${clamped.to_array()}')
}
