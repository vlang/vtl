module main

import vtl

fn main() {
	values := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	column_mask := vtl.from_1d([false, true, true])!
	selected := values.masked_select(column_mask)!
	println('Selected columns: ${selected.to_array()}')

	row_mask := vtl.from_array([true, false], [2, 1])!
	filled := values.masked_fill(row_mask, -1)!
	println('Filled rows: ${filled.to_array()}')
}
