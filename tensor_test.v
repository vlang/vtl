module vtl

fn test_rank_1() {
	t := from_1d([1, 2, 3, 4, 5])!
	assert t.rank() == 1
}

fn test_rank_2() {
	t := from_2d([[1, 2, 3], [4, 5, 6]])!
	assert t.rank() == 2
}

fn test_size_1() {
	t := from_1d([1, 2, 3, 4, 5])!
	assert t.size() == 5
}

fn test_size_2() {
	t := from_2d([[1, 2, 3], [4, 5, 6]])!
	assert t.size() == 6
}

fn test_is_matrix_1() {
	t := from_1d([1, 2, 3, 4, 5])!
	assert !t.is_matrix()
}

fn test_is_matrix_2() {
	t := from_2d([[1, 2, 3], [4, 5, 6]])!
	assert t.is_matrix()
}

fn test_is_square_matrix_1() {
	t := from_1d([1, 2, 3, 4, 5])!
	assert !t.is_square_matrix()
}

fn test_is_square_matrix_2() {
	t := from_2d([[1, 2, 3], [4, 5, 6]])!
	assert !t.is_square_matrix()
}

fn test_is_square_matrix_3() {
	t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.is_square_matrix()
}

fn test_is_vector_1() {
	t := from_1d([1, 2, 3, 4, 5])!
	assert t.is_vector()
}

fn test_is_vector_2() {
	t := from_2d([[1, 2, 3], [4, 5, 6]])!
	assert !t.is_vector()
}

fn test_is_vector_3() {
	t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert !t.is_vector()
}

fn test_is_row_major_1() {
	t := from_1d([1, 2, 3, 4, 5])!
	assert t.is_row_major()
}

fn test_is_row_major_2() {
	t := from_2d([[1, 2, 3], [4, 5, 6]])!
	assert t.is_row_major()
}

fn test_is_row_major_3() {
	t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.is_row_major()
}

fn test_is_col_major_1() {
	t := from_1d([1, 2, 3, 4, 5])!
	assert !t.is_col_major()
}

fn test_is_col_major_2() {
	t := from_2d([[1, 2, 3], [4, 5, 6]])!
	assert !t.is_col_major()
}

fn test_is_col_major_3() {
	t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert !t.is_col_major()
}

fn test_is_col_major_4() {
	t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]], memory: .col_major)!
	assert t.is_col_major()
}

fn test_is_row_major_contiguous_1() {
	t := from_1d([1, 2, 3, 4, 5])!
	assert t.is_row_major_contiguous()
}

fn test_is_row_major_contiguous_2() {
	t := from_2d([[1, 2, 3], [4, 5, 6]])!
	assert t.is_row_major_contiguous()
}

fn test_is_col_major_contiguous_1() {
	t := from_1d([1, 2, 3, 4, 5])!
	assert t.is_col_major_contiguous()
}

fn test_is_col_major_contiguous_2() {
	t := from_2d([[1, 2, 3], [4, 5, 6]])!
	assert !t.is_col_major_contiguous()
}

fn test_is_col_major_contiguous_3() {
	t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]], memory: .col_major)!
	assert t.is_col_major_contiguous()
}

fn test_is_contiguous_1() {
	t := from_1d([1, 2, 3, 4, 5])!
	assert t.is_contiguous()
}

fn test_is_contiguous_2() {
	t := from_2d([[1, 2, 3], [4, 5, 6]])!
	assert t.is_contiguous()
}

fn test_to_array_1() {
	t := from_1d([1, 2, 3, 4, 5])!
	assert t.to_array() == [1, 2, 3, 4, 5]
}

fn test_to_array_2() {
	t := from_2d([[1, 2, 3], [4, 5, 6]])!
	assert t.to_array() == [1, 2, 3, 4, 5, 6]
}

fn test_to_array_3() {
	t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.to_array() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
}

fn test_copy_1() {
	t := from_1d([1, 2, 3, 4, 5])!
	assert t.copy(.row_major).array_equal(t)
}

fn test_copy_2() {
	t := from_2d([[1, 2, 3], [4, 5, 6]])!
	assert t.copy(.row_major).array_equal(t)
}

fn test_copy_3() {
	t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.copy(.row_major).array_equal(t)
}

fn test_copy_4() {
	t := from_1d([1, 2, 3, 4, 5])!
	assert t.copy(.col_major).array_equal(t)
}

fn test_copy_5() {
	t := from_2d([[1, 2, 3], [4, 5, 6]])!
	result := t.copy(.col_major)
	expected := from_2d([[1, 2, 3], [4, 5, 6]], memory: .col_major)!
	assert result.array_equal(expected)
}

fn test_view_1() {
	t := from_1d([1, 2, 3, 4, 5])!
	assert t.view().array_equal(t)
}

fn test_view_2() {
	t := from_2d([[1, 2, 3], [4, 5, 6]])!
	assert t.view().array_equal(t)
}

fn test_view_3() {
	t := from_2d([[1, 2, 3], [4, 5, 6], [7, 8, 9]])!
	assert t.view().array_equal(t)
}
