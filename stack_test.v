module vtl

fn test_concatenate_flat() {
	a := ones<f64>([3])
	b := zeros<f64>([2])
	result := concatenate<f64>([a, b], axis: 0)
	expected := from_1d<f64>([1., 1, 1, 0, 0])
	// assert result.equal(expected)
}

fn test_concatenate() {
	a := ones<f64>([2, 2])
	b := zeros<f64>([2, 2])
	result := concatenate<f64>([a, b], axis: 0)
	expected := from_array<f64>([1., 1, 1, 1, 0, 0, 0, 0], [4, 2])
	// assert result.equal(expected)
}

fn test_vstack() {
	a := ones<f64>([3])
	b := zeros<f64>([2])
	result := vstack<f64>([a, b])
	expected := from_1d<f64>([1., 1, 1, 0, 0])
	// assert result.equal(expected)
}

fn test_hstack_flat() {
	a := ones<f64>([3])
	b := zeros<f64>([2])
	result := hstack<f64>([a, b])
	expected := from_1d<f64>([1., 1, 1, 0, 0])
	// assert result.equal(expected)
}

// fn test_hstack() {
// 	a := ones<f64>([2, 2])
// 	b := zeros<f64>([2, 2])
// 	result := hstack<f64>([a, b])
// 	expected := from_array<f64>([1., 1, 0, 0, 1, 1, 0, 0], [2, 4])
// 	println(result)
// 	println(expected)
// 	assert result.equal(expected)
// }

fn test_dstack_flat() {
	a := ones<f64>([3])
	b := zeros<f64>([3])
	result := dstack<f64>([a, b])
	expected := from_array<f64>([1., 0, 1, 0, 1, 0], [1, 3, 2])
	// assert result.equal(expected)
}

fn test_dstack() {
	a := ones<f64>([2, 2])
	b := zeros<f64>([2, 2])
	result := dstack<f64>([a, b])
	expected := from_array<f64>([1., 0, 1, 0, 1, 0, 1, 0], [2, 2, 2])
	// assert result.equal(expected)
}

// fn test_column_stack_flat() {
// 	a := ones<f64>([2])
// 	b := zeros<f64>([2])
// 	result := column_stack<f64>([a, b])
// 	expected := from_array<f64>([1., 0, 1, 0], [2, 2])
// 	assert result.equal(expected)
// }

// fn test_column_stack_2d() {
// 	a := ones<f64>([2, 2])
// 	b := zeros<f64>([2, 2])
// 	result := column_stack<f64>([a, b])
// 	expected := from_array<f64>([1., 1, 0, 0, 1, 1, 0, 0], [2, 4])
// 	println(result)
// 	println(expected)
// 	assert result.equal(expected)
// }

// fn test_stack() {
// 	a := ones<f64>([2, 2])
// 	b := zeros<f64>([2, 2])
// 	result := stack<f64>([a, b], axis: 1)
// 	expected := from_array<f64>([1., 1, 0, 0, 1, 1, 0, 0], [2, 2, 2])
// 	println(result)
// 	println(expected)
// 	assert result.equal(expected)
// }
