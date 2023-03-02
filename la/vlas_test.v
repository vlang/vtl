module la

import vtl

fn test_matmul1() {
	a := vtl.from_2d([[1.0, 0], [0.0, 1]])!
	b := vtl.from_2d([[4.0, 1], [2.0, 2]])!
	expected := vtl.from_2d([[4.0, 1], [2.0, 2]])!
	result := matmul(a, b)!
	assert result.shape == [2, 2]
	assert result.array_equal(expected)
}

// fn test_matmul2() ? {
// 	a := vtl.seq[f64](2 * 2 * 4).reshape([2, 2, 4])?
// 	b := vtl.seq[f64](2 * 2 * 4).reshape([2, 4, 2])?
// 	result := matmul(a, b)?
// 	assert result.shape == [2, 2, 2]
// }

// fn test_matmul3() ? {
// 	a := vtl.from_2d([[1.0, 0], [0.0, 1]])?
// 	b := vtl.from_1d([1.0, 2])?
// 	expected := vtl.from_1d([1.0, 2])?
// 	result := matmul(a, b)?
// 	assert result.shape == [1, 2]
// 	assert result.array_equal(expected)
// }
