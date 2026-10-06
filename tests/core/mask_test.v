module core

import vtl

fn test_masked_select_returns_row_major_selected_values() {
	values := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	mask := vtl.from_2d([[true, false, true], [false, true, false]])!
	got := values.masked_select(mask)!
	assert got.shape == [3]
	assert got.array_equal[int](vtl.from_1d([1, 3, 5])!)
}

fn test_masked_select_handles_transposed_views() {
	values := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!.transpose([1, 0])!
	mask := vtl.from_2d([[false, true], [true, false], [false, true]])!
	got := values.masked_select(mask)!
	assert got.array_equal[int](vtl.from_1d([4, 2, 6])!)
}

fn test_masked_fill_replaces_selected_values() {
	values := vtl.from_2d([[1.0, 2.0], [3.0, 4.0]])!
	mask := vtl.from_2d([[false, true], [true, false]])!
	got := values.masked_fill(mask, -1.0)!
	expected := vtl.from_2d([[1.0, -1.0], [-1.0, 4.0]])!
	assert got.array_equal(expected)
}

fn test_masked_select_broadcasts_row_and_column_masks() {
	values := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	row_mask := vtl.from_1d([true, false, true])!
	assert values.masked_select(row_mask)!.to_array() == [1, 3, 4, 6]
	column_mask := vtl.from_1d([false, true, true])!
	assert values.masked_select(column_mask)!.to_array() == [2, 3, 5, 6]
}

fn test_masked_fill_broadcasts_mask_without_changing_input() {
	values := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	mask := vtl.from_array([true, false], [2, 1])!
	got := values.masked_fill(mask, -1)!
	expected := vtl.from_2d([[-1, -1, -1], [4, 5, 6]])!
	assert got.array_equal(expected)
	assert values.to_array() == [1, 2, 3, 4, 5, 6]
}

fn test_masked_operations_reject_shape_mismatch() {
	values := vtl.from_2d([[1, 2], [3, 4]])!
	mask := vtl.from_1d([true, false, true])!
	if _ := values.masked_select(mask) {
		assert false, 'masked_select must reject a differently shaped mask'
	}
	if _ := values.masked_fill(mask, 0) {
		assert false, 'masked_fill must reject a differently shaped mask'
	}
}
