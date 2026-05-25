module main

import vtl

fn main() {
	// Row vector [1, 3]
	row := vtl.from_2d([[10, 20, 30]]) or { panic(err) }
	// Column vector [4, 1]
	col := vtl.from_2d([[1], [2], [3], [4]]) or { panic(err) }

	grid := col.add(row) or { panic(err) }

	println('row shape: ${row.shape}')
	println('col shape: ${col.shape}')
	println('grid shape: ${grid.shape}')
	println(grid)

	// Scalar broadcasting
	shifted := grid.add_scalar(5) or { panic(err) }
	println('shifted (+5):')
	println(shifted)
}
