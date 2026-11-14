module core

import vtl

fn test_concatenate_flat() {
	a := vtl.ones[f64]([3])
	b := vtl.zeros[f64]([2])
	result := vtl.concatenate[f64]([a, b], axis: 0)!
	expected := vtl.from_1d[f64]([1.0, 1, 1, 0, 0])!
	assert result.array_equal(expected)
}

fn test_concatenate() {
	a := vtl.ones[f64]([2, 2])
	b := vtl.zeros[f64]([2, 2])
	result := vtl.concatenate[f64]([a, b], axis: 0)!
	expected := vtl.from_array([1.0, 1, 1, 1, 0, 0, 0, 0], [4, 2])!
	assert result.array_equal(expected)
}

fn test_concatenate_rejects_axis_equal_to_rank() {
	a := vtl.ones[f64]([2, 2])
	_ := vtl.concatenate[f64]([a], axis: 2) or {
		assert err.msg().contains('axis out of range')
		return
	}
	assert false, 'expected concatenate to reject axis equal to rank'
}

fn test_vstack() {
	a := vtl.ones[f64]([3])
	b := vtl.zeros[f64]([2])
	result := vtl.vstack[f64]([a, b])!
	expected := vtl.from_1d[f64]([1.0, 1, 1, 0, 0])!
	assert result.array_equal(expected)
}

fn test_hstack_flat() {
	a := vtl.ones[f64]([3])
	b := vtl.zeros[f64]([2])
	result := vtl.hstack[f64]([a, b])!
	expected := vtl.from_1d[f64]([1.0, 1, 1, 0, 0])!
	assert result.array_equal(expected)
}

fn test_hstack() {
	a := vtl.ones[f64]([2, 2])
	b := vtl.zeros[f64]([2, 2])
	result := vtl.hstack[f64]([a, b])!
	expected := vtl.from_array([1.0, 1, 0, 0, 1, 1, 0, 0], [2, 4])!
	assert result.array_equal(expected)
}

fn test_dstack_flat() {
	a := vtl.ones[f64]([3])
	b := vtl.zeros[f64]([3])
	result := vtl.dstack[f64]([a, b])!
	expected := vtl.from_array([1.0, 0, 1, 0, 1, 0], [1, 3, 2])!
	assert result.array_equal(expected)
}

fn test_dstack() {
	a := vtl.ones[f64]([2, 2])
	b := vtl.zeros[f64]([2, 2])
	result := vtl.dstack[f64]([a, b])!
	expected := vtl.from_array([1.0, 0, 1, 0, 1, 0, 1, 0], [2, 2, 2])!
	assert result.array_equal(expected)
}

fn test_column_stack_flat() {
	a := vtl.ones[f64]([2])
	b := vtl.zeros[f64]([2])
	result := vtl.column_stack[f64]([a, b])!
	expected := vtl.from_array([1.0, 0, 1, 0], [2, 2])!
	assert result.array_equal(expected)
}

fn test_column_stack_2d() {
	a := vtl.ones[f64]([2, 2])
	b := vtl.zeros[f64]([2, 2])
	result := vtl.column_stack[f64]([a, b])!
	expected := vtl.from_array([1.0, 1, 0, 0, 1, 1, 0, 0], [2, 4])!
	assert result.array_equal(expected)
}

fn test_stack() {
	a := vtl.ones[f64]([2, 2])
	b := vtl.zeros[f64]([2, 2])
	result := vtl.stack[f64]([a, b], axis: 1)!
	expected := vtl.from_array([1.0, 1, 0, 0, 1, 1, 0, 0], [2, 2, 2])!
	assert result.array_equal(expected)
}

fn test_unsqueeze_1() {
	a := vtl.ones[f64]([2, 2])
	result := a.unsqueeze[f64](axis: 1)!
	expected := vtl.from_array([1.0, 1, 1, 1], [2, 1, 2])!
	assert result.array_equal(expected)
}

fn test_unsqueeze_2() {
	a := vtl.ones[f64]([2, 2])
	result := a.unsqueeze[f64](axis: 0)!
	expected := vtl.from_array([1.0, 1, 1, 1], [1, 2, 2])!
	assert result.array_equal(expected)
}

fn test_expand_dims() {
	a := vtl.ones[f64]([2, 2])
	result := a.expand_dims[f64](axis: 1)!
	expected := vtl.from_array([1.0, 1, 1, 1], [2, 1, 2])!
	assert result.array_equal(expected)
}

fn test_axis_insertion_and_squeeze_preserve_transposed_storage() ! {
	mut tensor := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	transposed := tensor.transpose([1, 0])!

	mut unsqueezed := transposed.unsqueeze[int](axis: 0)!
	unsqueezed.set([0, 1, 0], 99)
	assert tensor.get([0, 1]) == 99

	mut squeezed := unsqueezed.squeeze[int](axis: 0)!
	squeezed.set([2, 1], 88)
	assert tensor.get([1, 2]) == 88

	mut expanded := transposed.expand_dims[int](axis: 1)!
	expanded.set([1, 0, 0], 77)
	assert tensor.get([0, 1]) == 77

	mut column_major := vtl.from_2d([[1, 2, 3], [4, 5, 6]], memory: .col_major)!
	mut column_expanded := column_major.unsqueeze[int](axis: 1)!
	assert column_expanded.memory == .col_major
	column_expanded.set([1, 0, 2], 66)
	assert column_major.get([1, 2]) == 66

	mut column_squeezed := column_expanded.squeeze[int](axis: 1)!
	column_squeezed.set([0, 1], 55)
	assert column_major.get([0, 1]) == 55
}
