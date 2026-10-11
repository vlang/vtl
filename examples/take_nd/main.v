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
	mode_indices := vtl.from_1d([-1, 3])!
	wrapped := values.take_nd_with_mode(mode_indices, 1, .wrap)!
	println('wrapped axis gather: ${wrapped.to_array()}')
	clipped := values.take_flat_with_mode(vtl.from_1d([-1, 6])!, .clip)!
	println('clipped flat gather: ${clipped.to_array()}')
}
