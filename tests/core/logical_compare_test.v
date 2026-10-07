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
