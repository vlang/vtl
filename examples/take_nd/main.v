module main

import vtl

fn main() {
	values := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	indices := vtl.from_array([0, 2, 1, 0], [2, 2])!
	selected := values.take_nd(indices, 1)!
	println('selected shape: ${selected.shape}')
	println('selected values: ${selected.to_array()}')
	flat := values.take_flat(indices)!
	println('flat gather: ${flat.to_array()}')
}
