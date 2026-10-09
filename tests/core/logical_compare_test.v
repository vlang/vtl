module core

import vtl

fn test_equal_and_not_equal_contiguous_inputs() ! {
	left := vtl.from_2d([[1, 2], [3, 4]])!
	right := vtl.from_2d([[1, 0], [3, 5]])!

	assert left.equal(right)!.to_array() == [true, false, true, false]
	assert left.not_equal(right)!.to_array() == [false, true, false, true]
}

fn test_equal_and_not_equal_keep_broadcast_semantics() ! {
	left := vtl.from_2d([[1, 2, 1], [4, 2, 5]])!
	right := vtl.from_1d([1, 2, 3])!

	assert left.equal(right)!.to_array() == [true, true, false, false, true, false]
	assert left.not_equal(right)!.to_array() == [false, false, true, true, false, true]
}

fn test_equal_and_not_equal_keep_strided_view_semantics() ! {
	base := vtl.from_2d([[1, 2], [3, 4]])!
	transposed := base.transpose([1, 0])!

	assert transposed.equal(base)!.to_array() == [true, false, false, true]
	assert transposed.not_equal(base)!.to_array() == [false, true, true, false]
}

fn test_approximate_comparisons_use_contiguous_and_broadcast_paths() ! {
	left := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
	row := vtl.from_1d([1.0, 2.0])!
	expected := [true, true, false, false]

	assert left.tolerance(row, 0.0)!.to_array() == expected
	assert left.close(row)!.to_array() == expected
	assert left.veryclose(row)!.to_array() == expected
	assert left.alike(row)!.to_array() == expected
}

fn test_approximate_comparisons_support_strided_broadcast_views() ! {
	base := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
	transposed := base.transpose([1, 0])!
	column := vtl.from_2d([[1.0], [2.0]])!
	expected := [true, false, true, false]

	assert transposed.tolerance(column, 0.0)!.to_array() == expected
	assert transposed.close(column)!.to_array() == expected
	assert transposed.veryclose(column)!.to_array() == expected
	assert transposed.alike(column)!.to_array() == expected
}
