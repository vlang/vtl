module vtl

fn test_infer_row_major_strides() {
	a := [3, 2, 2]
	expected := [4, 2, 1]
	result := strides_from_shape(a, .row_major)
	assert result == expected
}

fn test_infer_column_major_strides() {
	a := [3, 2, 2]
	expected := [1, 3, 6]
	result := strides_from_shape(a, .col_major)
	assert result == expected
}
