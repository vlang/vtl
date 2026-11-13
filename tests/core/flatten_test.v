module core

import vtl

fn test_ravel_transposed_tensor_uses_logical_row_major_order() {
	mut tensor := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	transposed := tensor.transpose([1, 0])!
	mut raveled := transposed.ravel()!
	assert raveled.shape == [6]
	assert raveled.to_array() == [1, 4, 2, 5, 3, 6]
	raveled.set_nth(0, 99)
	assert tensor.get_nth(0) == 1
}

fn test_ravel_row_major_contiguous_tensor_shares_storage() {
	mut tensor := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	mut raveled := tensor.ravel()!
	assert raveled.shape == [6]
	raveled.set_nth(0, 99)
	assert tensor.get_nth(0) == 99
}

fn test_flatten_returns_one_dimensional_copy_values() {
	tensor := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	mut flattened := tensor.flatten()!
	assert flattened.shape == [6]
	assert flattened.to_array() == [1, 2, 3, 4, 5, 6]
	flattened.set_nth(0, 99)
	assert tensor.get_nth(0) == 1

	mut transposed_flattened := tensor.transpose([1, 0])!.flatten()!
	assert transposed_flattened.to_array() == [1, 4, 2, 5, 3, 6]
	transposed_flattened.set_nth(0, 88)
	assert tensor.get_nth(0) == 1
}
