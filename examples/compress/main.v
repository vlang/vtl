module main

import vtl

fn main() {
	values := vtl.from_2d([[10, 20, 30], [40, 50, 60]])!
	condition := vtl.from_1d([true, false])!

	// With no axis, compress reads the tensor in row-major order.
	println('Flattened: ${vtl.compress(condition, values)!.to_array()}')

	// With an axis, the condition selects rows or columns and preserves rank.
	rows := vtl.from_1d([false, true])!
	println('Selected rows: ${vtl.compress_axis(rows, values, 0)!.to_array()}')
	columns := vtl.from_1d([true, false, true])!
	println('Selected columns: ${vtl.compress_axis(columns, values, 1)!.to_array()}')
}
