module vtl

fn test_concatenate_flat() {
	a := ones([3])
	b := zeros([2])
	result := concatenate([a, b], axis: 0)
	expected := from_1d([1, 1, 1, 0, 0])
	assert result.equal(expected)
}

fn test_vstack() {
	a := ones([3])
	b := zeros([2])
	result := vstack([a, b])
	expected := from_1d([1, 1, 1, 0, 0])
	assert result.equal(expected)
}

fn test_hstack_flat() {
	a := ones([3])
	b := zeros([2])
	result := hstack([a, b])
	expected := from_1d([1, 1, 1, 0, 0])
	assert result.equal(expected)
}

fn test_hstack() {
	a := ones([2, 2])
	b := zeros([2, 2])
	result := hstack([a, b])
	expected := from_varray([1, 1, 0, 0, 1, 1, 0, 0], [2, 4])
	assert result.equal(expected)
}

fn test_dstack_flat() {
	a := ones([3])
	b := zeros([3])
	result := dstack([a, b])
	expected := from_varray([1, 0, 1, 0, 1, 0], [1, 3, 2])
	assert result.equal(expected)
}

fn test_dstack() {
	a := ones([2, 2])
	b := zeros([2, 2])
	result := dstack([a, b])
	expected := from_varray([1, 0, 1, 0, 1, 0, 1, 0], [2, 2, 2])
	assert result.equal(expected)
}

fn test_column_stack_flat() {
	a := ones([2])
	b := zeros([2])
	result := column_stack([a, b])
	expected := from_varray([1, 0, 1, 0], [2, 2])
	assert result.equal(expected)
}

// fn test_column_stack_2d() {
// 	a := ones([2, 2])
// 	b := zeros([2, 2])
// 	result := column_stack([a, b])
// 	expected := from_varray([1, 1, 0, 0, 1, 1, 0, 0], [2, 4])
// 	assert result.equal(expected)
// }

// fn test_stack() {
// 	a := ones([2, 2])
// 	b := zeros([2, 2])
// 	result := stack([a, b], axis: 1)
// 	expected := from_varray([1, 1, 0, 0, 1, 1, 0, 0], [2, 2, 2])
// 	assert result.equal(expected)
// }
