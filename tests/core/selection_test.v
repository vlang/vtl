module main

import vtl

fn test_where_selects_and_broadcasts() {
	condition := vtl.from_array([true, false, false, true], [2, 2])!
	x := vtl.from_1d([1, 2])!
	y := vtl.from_2d([[10, 20], [30, 40]])!
	got := vtl.where(condition, x, y)!
	expected := vtl.from_2d([[1, 20], [30, 2]])!
	assert got.array_equal(expected)
}

fn test_where_rejects_incompatible_shapes() {
	condition := vtl.from_1d([true, false, true])!
	x := vtl.from_1d([1, 2])!
	y := vtl.from_1d([3, 4])!
	if _ := vtl.where(condition, x, y) {
		assert false, 'incompatible shapes must return an error'
	} else {
		assert true
	}
}

fn test_where_handles_scalar_choices() {
	condition := vtl.from_1d([true, false])!
	x := vtl.tensor(7, [], memory: .row_major)
	y := vtl.from_1d([3, 4])!
	got := vtl.where(condition, x, y)!
	expected := vtl.from_1d([7, 4])!
	assert got.array_equal(expected)
}
