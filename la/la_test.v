module la

import vtl

fn test_dot_1() {
	a := vtl.from_1d([1.0, 2.0, 3.0])!
	b := vtl.from_1d([4.0, 5.0, 6.0])!
	expected := vtl.from_1d([32.0])!
	result := dot(a, b)!
	assert result.shape == [1]
	assert result.array_equal(expected)
}

fn test_det_1() {
	a := vtl.from_2d([[1.0, 0], [0.0, 1]])!
	expected := vtl.from_1d([1.0])!
	result := det(a)!
	assert result.shape == [1]
	assert result.array_equal(expected)
}

fn test_inv_1() {
	a := vtl.from_2d([[1.0, 0], [0.0, 1]])!
	expected := vtl.from_2d([[1.0, 0], [0.0, 1]])!
	result := inv(a)!
	assert result.shape == [2, 2]
	assert result.array_equal(expected)
}

fn test_matmul_1() {
	a := vtl.from_2d([[1.0, 0], [0.0, 1]])!
	b := vtl.from_2d([[4.0, 1], [2.0, 2]])!
	expected := vtl.from_2d([[4.0, 1], [2.0, 2]])!
	result := matmul(a, b)!
	assert result.shape == [2, 2]
	assert result.array_equal(expected)
}

fn test_matmul_2() {
	a := vtl.seq[f64](2 * 2 * 4).reshape([2, 2, 4])!
	b := vtl.seq[f64](2 * 2 * 4).reshape([2, 4, 2])!
	if _ := matmul(a, b) {
		assert false
	} else {
		assert true
	}
}

fn test_matmul_3() {
	a := vtl.from_2d([[1.0, 0], [0.0, 1]])!
	b := vtl.from_1d([1.0, 2])!
	if _ := matmul(a, b) {
		assert false
	} else {
		assert true
	}
}
