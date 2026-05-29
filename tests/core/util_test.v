module main

import vtl

fn test_infer_row_major_strides() {
	a := [3, 2, 2]
	expected := [4, 2, 1]
	result := vtl.tensor(0, a, memory: .row_major).strides
	assert result == expected
}

fn test_infer_column_major_strides() {
	a := [3, 2, 2]
	expected := [1, 3, 6]
	result := vtl.tensor(0, a, memory: .col_major).strides
	assert result == expected
}
