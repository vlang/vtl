import vnum.num

fn test_concatenate_flat() {
	a := num.ones([3])
	b := num.zeros([2])
	result := num.concatenate([a, b], 0)
	expected := num.from_int_1d([1, 1, 1, 0, 0])
	assert num.allclose(result, expected)
}

fn test_vstack() {
	a := num.ones([3])
	b := num.zeros([2])
	result := num.vstack([a, b])
	expected := num.from_int_1d([1, 1, 1, 0, 0])
	assert num.allclose(result, expected)
}

fn test_hstack_flat() {
	a := num.ones([3])
	b := num.zeros([2])
	result := num.hstack([a, b])
	expected := num.from_int_1d([1, 1, 1, 0, 0])
	assert num.allclose(result, expected)
}

fn test_hstack() {
	a := num.ones([2, 2])
	b := num.zeros([2, 2])
	result := num.hstack([a, b])
	expected := num.from_int([1, 1, 0, 0, 1, 1, 0, 0], [2, 4])
	assert num.allclose(result, expected)
}

fn test_dstack_flat() {
	a := num.ones([3])
	b := num.zeros([3])
	result := num.dstack([a, b])
	expected := num.from_int([1, 0, 1, 0, 1, 0], [1, 3, 2])
	assert num.allclose(result, expected)
}

fn test_dstack() {
	a := num.ones([2, 2])
	b := num.zeros([2, 2])
	result := num.dstack([a, b])
	expected := num.from_int([1, 0, 1, 0, 1, 0, 1, 0], [2, 2, 2])
	assert num.allclose(result, expected)
}

fn test_column_stack_flat() {
	a := num.ones([2])
	b := num.zeros([2])
	result := num.column_stack([a, b])
	expected := num.from_int([1, 0, 1, 0], [2, 2])
	assert num.allclose(result, expected)
}

fn test_column_stack_2d() {
	a := num.ones([2, 2])
	b := num.zeros([2, 2])
	result := num.column_stack([a, b])
	expected := num.from_int([1, 1, 0, 0, 1, 1, 0, 0], [2, 4])
	assert num.allclose(result, expected)
}

fn test_stack() {
	a := num.ones([2, 2])
	b := num.zeros([2, 2])
	result := num.stack([a, b], 1)
	expected := num.from_int([1, 1, 0, 0, 1, 1, 0, 0], [2, 2, 2])
	assert num.allclose(result, expected)
}
