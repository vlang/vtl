module core

import vtl

fn test_elementwise_logical_ops_broadcast_numeric_tensors() ! {
	values := vtl.from_array([0, 1, -2, 0], [2, 2])!
	other := vtl.from_1d([0, 3])!
	assert values.logical_and(other)!.to_array() == [false, true, false, false]
	assert values.logical_or(other)!.to_array() == [false, true, true, true]
	assert values.logical_xor(other)!.to_array() == [false, false, true, true]
	assert values.logical_not().to_array() == [true, false, false, true]
}

fn test_elementwise_logical_ops_accept_boolean_tensors() ! {
	left := vtl.from_1d([true, false, true])!
	right := vtl.from_1d([false, false, true])!
	assert left.logical_and(right)!.to_array() == [false, false, true]
	assert left.logical_or(right)!.to_array() == [true, false, true]
	assert left.logical_xor(right)!.to_array() == [true, false, false]
	assert right.logical_not().to_array() == [true, true, false]
}
