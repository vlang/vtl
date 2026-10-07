module main

import vtl

fn main() {
	matrix := vtl.from_array[int]([]int{len: 35, init: index}, [5, 7])!
	rows := vtl.from_1d([0, 2, 4])!
	column_range := vtl.slice_index(1, 3, 1)!
	selected := vtl.mixed_index[int](matrix, [vtl.array_index(rows), column_range])!
	assert selected.shape == [3, 2]
	assert selected.to_array() == [1, 2, 15, 16, 29, 30]

	three_dimensional := vtl.from_array[int]([]int{len: 24, init: index}, [2, 3, 4])!
	outer := vtl.from_1d([0, 1])!
	inner := vtl.from_1d([1, 2])!
	all_rows := vtl.slice_all(1)!
	separated := vtl.mixed_index[int](three_dimensional, [vtl.array_index(outer), all_rows,
		vtl.array_index(inner)])!
	assert separated.shape == [2, 3]
	assert separated.to_array() == [1, 5, 9, 14, 18, 22]
	with_new_axis := vtl.mixed_index[int](three_dimensional, [vtl.ellipsis_index(),
		vtl.newaxis_index()])!
	assert with_new_axis.shape == [2, 3, 4, 1]

	mut last_row := vtl.mixed_index[int](matrix, [vtl.integer_index(-1)])!
	last_row.set([0], 99)
	assert matrix.get([4, 0]) == 99

	println('coordinate and range result: ${selected.to_array()}')
	println('separated coordinate result shape: ${separated.shape}')
	println('ellipsis plus new axis shape: ${with_new_axis.shape}')
	println('basic indexing shares source storage: ${matrix.get([4, 0])}')
}
