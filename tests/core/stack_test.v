module main

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
