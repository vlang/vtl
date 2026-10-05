module main

import vtl

fn test_tensor_arithmetic_operators() {
	a := vtl.from_1d([f64(1), 2, 3])!
	b := vtl.from_1d([f64(4), 5, 6])!

	add_result := a + b
	assert add_result.array_equal(vtl.from_1d([f64(5), 7, 9])!)
}

fn test_tensor_add_operator_broadcasts() {
	a := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	b := vtl.from_1d([10, 20, 30])!
	result := a + b
	assert result.array_equal(vtl.from_2d([[11, 22, 33], [14, 25, 36]])!)
}
