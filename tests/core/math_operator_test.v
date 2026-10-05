module main

import vtl

fn test_tensor_arithmetic_operators() {
	a := vtl.from_1d([f64(1), 2, 3])!
	b := vtl.from_1d([f64(4), 5, 6])!

	add_result := a + b
	assert add_result.array_equal(vtl.from_1d([f64(5), 7, 9])!)
	multiply_result := a * b
	assert multiply_result.array_equal(vtl.from_1d([f64(4), 10, 18])!)
	divide_result := b / a
	assert divide_result.array_equal(vtl.from_1d([f64(4), 2.5, 2])!)
}

fn test_tensor_add_operator_broadcasts() {
	a := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	b := vtl.from_1d([10, 20, 30])!
	result := a + b
	assert result.array_equal(vtl.from_2d([[11, 22, 33], [14, 25, 36]])!)
}

fn test_tensor_multiply_and_divide_operators_broadcast() {
	a := vtl.from_2d([[1.0, 2, 3], [4, 5, 6]])!
	b := vtl.from_1d([1.0, 2, 3])!
	assert (a * b).array_equal(vtl.from_2d([[1.0, 4, 9], [4, 10, 18]])!)
	assert (a / b).array_equal(vtl.from_2d([[1.0, 1, 1], [4, 2.5, 2]])!)
}
