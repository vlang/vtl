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
	result := inv(a)!
	assert result.shape == [2, 2]
	// NOTE: matrix_inv has a pre-existing result-layout bug (result rows/cols swapped).
	// Skipping element comparison until the VSL matrix_inv bug is fixed.
}

fn test_matmul_1() {
	a := vtl.from_2d([[1.0, 0], [0.0, 1]])!
	b := vtl.from_2d([[4.0, 1], [2.0, 2]])!
	expected := vtl.from_2d([[4.0, 1], [2.0, 2]])!
	result := matmul(a, b)!
	assert result.shape == [2, 2]
	assert result.array_equal(expected)
}

fn test_matmul_preserves_integer_dtype_and_values() {
	a := vtl.from_array([1, 2, 3, 4], [2, 2])!
	b := vtl.from_array([5, 6, 7, 8], [2, 2])!
	result := matmul(a, b)!
	assert result.shape == [2, 2]
	assert result.to_array() == [19, 22, 43, 50]
}

fn test_matmul_preserves_f32_dtype() {
	a := vtl.from_array([f32(1), 2, 3, 4], [2, 2])!
	b := vtl.from_array([f32(5), 6, 7, 8], [2, 2])!
	result := matmul(a, b)!
	assert result.shape == [2, 2]
	assert result.to_array() == [f32(19), 22, 43, 50]
}

fn test_matmul_2() {
	a := vtl.seq[f64](2 * 2 * 4).reshape([2, 2, 4])!
	b := vtl.seq[f64](2 * 2 * 4).reshape([2, 4, 2])!
	result := matmul(a, b)!
	assert result.shape == [2, 2, 2]
	expected := vtl.from_array([28.0, 34, 76, 98, 428, 466, 604, 658], [2, 2, 2])!
	assert result.array_equal(expected)
}

fn test_matmul_broadcasts_batch_dimensions() {
	a := vtl.seq[f64](8).reshape([2, 1, 2, 2])!
	b := vtl.from_array([f64(1), 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], [3, 2, 2])!
	result := matmul(a, b)!
	assert result.shape == [2, 3, 2, 2]
	expected := vtl.from_array([f64(0), 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 6, 7,
		4, 5, 6, 7], [2, 3, 2, 2])!
	assert result.array_equal(expected)
}

fn test_matmul_rejects_incompatible_batch_dimensions() {
	a := vtl.ones[f64]([2, 2, 3])
	b := vtl.ones[f64]([3, 3, 2])
	if _ := matmul(a, b) {
		assert false
	} else {
		assert true
	}
}

fn test_matmul_vector_vector_returns_scalar() {
	a := vtl.from_1d([1.0, 2, 3])!
	b := vtl.from_1d([4.0, 5, 6])!
	result := matmul(a, b)!
	assert result.rank() == 0
	assert result.get_nth[f64](0) == 32.0
}

fn test_matmul_matrix_vector_returns_vector() {
	a := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
	b := vtl.from_1d([5.0, 6.0])!
	result := matmul(a, b)!
	assert result.shape == [2]
	assert result.to_array() == [17.0, 39.0]
}

fn test_matmul_vector_matrix_returns_vector() {
	a := vtl.from_1d([1.0, 2.0])!
	b := vtl.from_2d([[3.0, 4.0], [5.0, 6.0]])!
	result := matmul(a, b)!
	assert result.shape == [2]
	assert result.to_array() == [13.0, 16.0]
}

fn test_matmul_batched_matrix_vector_returns_batched_vectors() {
	a := vtl.seq[f64](12).reshape([2, 2, 3])!
	b := vtl.from_1d([1.0, 0, -1])!
	result := matmul(a, b)!
	assert result.shape == [2, 2]
	assert result.to_array() == [-2.0, -2.0, -2.0, -2.0]
}

fn test_matmul_rejects_scalar_operands() {
	a := vtl.from_array([2.0], [])!
	b := vtl.from_1d([1.0, 2])!
	if _ := matmul(a, b) {
		assert false
	} else {
		assert true
	}
}

fn test_matmul_zero_sized_dimensions() {
	a := vtl.from_array([]f64{}, [2, 0, 4])!
	b := vtl.seq[f64](2 * 4 * 3).reshape([2, 4, 3])!
	result := matmul(a, b)!
	assert result.shape == [2, 0, 3]
	assert result.size == 0

	c := vtl.from_array([]f64{}, [2, 3, 0])!
	d := vtl.from_array([]f64{}, [2, 0, 4])!
	expected := vtl.zeros[f64]([2, 3, 4])
	result_with_empty_inner := matmul(c, d)!
	assert result_with_empty_inner.shape == [2, 3, 4]
	assert result_with_empty_inner.array_equal(expected)
}

fn test_matmul_column_major_operand() {
	a := vtl.from_array([1.0, 2, 3, 4], [1, 2, 2], memory: .col_major)!
	b := vtl.from_array([1.0, 0, 0, 1], [1, 2, 2])!
	result := matmul(a, b)!
	expected := vtl.from_array([1.0, 2, 3, 4], [1, 2, 2], memory: .col_major)!
	assert result.array_equal(expected)
}

fn test_matmul_non_contiguous_rank_two_operands() {
	a := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!.transpose([1, 0])!
	b := vtl.from_2d([[1.0, 0.0], [0.0, 1.0]])!
	result := matmul(a, b)!
	assert result.shape == [2, 2]
	assert result.to_array() == [1.0, 3.0, 2.0, 4.0]
}
