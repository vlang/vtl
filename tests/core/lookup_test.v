module core

import vtl
import math

fn test_isin_returns_mask_for_numeric_and_duplicate_values() {
	elements := vtl.from_2d([[3, 1, 3], [2, 4, 1]])!
	test_elements := vtl.from_1d([1, 3, 3])!
	got := vtl.isin(elements, test_elements)
	assert got.shape == elements.shape
	assert got.to_array() == [true, true, true, false, false, true]
}

fn test_isin_supports_strings_and_empty_search_sets() {
	elements := vtl.from_1d(['v', 'numpy', 'vsl'])!
	choices := vtl.from_1d(['v', 'vtl'])!
	assert vtl.isin(elements, choices).to_array() == [true, false, false]
	empty := vtl.from_1d([]int{})!
	integers := vtl.from_1d([1, 2])!
	assert vtl.isin(integers, empty).to_array() == [false, false]
}

fn test_isin_supports_boolean_tensors() {
	elements := vtl.from_1d([true, false, true])!
	choices := vtl.from_1d([false])!
	assert vtl.isin(elements, choices).to_array() == [false, true, false]
}

fn test_isin_nan_does_not_match_and_views_keep_logical_order() {
	elements := vtl.from_2d([[math.nan(), 2.0], [3.0, 2.0]])!.transpose([1, 0])!
	choices := vtl.from_1d([math.nan(), 2.0])!
	assert vtl.isin(elements, choices).to_array() == [false, false, true, true]
}

fn test_count_nonzero_globally_and_by_axis_with_keepdims() ! {
	tensor := vtl.from_2d([[0, 2, 0], [3, 0, 4]])!
	assert vtl.count_nonzero[int](tensor) == 3
	rows := vtl.count_nonzero_axis[int](tensor, 1, false)!
	assert rows.shape == [2]
	assert rows.to_array() == [1, 2]
	columns := vtl.count_nonzero_axis[int](tensor, 0, false)!
	assert columns.shape == [3]
	assert columns.to_array() == [1, 1, 1]
	columns_keepdims := vtl.count_nonzero_axis[int](tensor, -2, true)!
	assert columns_keepdims.shape == [1, 3]
	assert columns_keepdims.to_array() == [1, 1, 1]
	transposed := vtl.count_nonzero_axis[int](tensor.t()!, 1, true)!
	assert transposed.shape == [3, 1]
	assert transposed.to_array() == [1, 1, 1]
	series := vtl.from_1d([0, 2, 3])!
	assert vtl.count_nonzero_axis[int](series, 0, false)!.get_nth[int](0) == 2
	assert vtl.count_nonzero_axis[int](series, 0, true)!.shape == [1]
}

fn test_count_nonzero_handles_empty_scalar_and_nan() ! {
	empty := vtl.from_array([]int{}, [0, 3])!
	assert vtl.count_nonzero[int](empty) == 0
	assert vtl.count_nonzero_axis[int](empty, 1, false)!.shape == [0]
	scalar := vtl.from_array([7], [])!
	assert vtl.count_nonzero[int](scalar) == 1
	values := vtl.from_1d([0.0, math.nan(), 2.0])!
	assert vtl.count_nonzero[f64](values) == 2
	if _ := vtl.count_nonzero_axis[int](scalar, 0, false) {
		assert false, 'count_nonzero_axis must reject scalar input'
	} else {
		assert true
	}
}

fn test_argwhere_groups_nonzero_coordinates_by_element() ! {
	tensor := vtl.from_2d([[0, 2, 0], [3, 4, 0]])!
	coordinates := vtl.argwhere[int](tensor)!
	assert coordinates.shape == [3, 2]
	assert coordinates.to_array() == [0, 1, 1, 0, 1, 1]
}

fn test_argwhere_handles_views_empty_and_scalar_inputs() ! {
	base := vtl.from_2d([[0, 2, 0], [3, 0, 4]])!
	transposed := vtl.argwhere[int](base.t()!)!
	assert transposed.shape == [3, 2]
	assert transposed.to_array() == [0, 1, 1, 0, 2, 1]

	empty := vtl.from_array([]int{}, [0, 3])!
	empty_coordinates := vtl.argwhere[int](empty)!
	assert empty_coordinates.shape == [0, 2]
	assert empty_coordinates.size == 0

	scalar := vtl.from_array([7], [])!
	scalar_coordinates := vtl.argwhere[int](scalar)!
	assert scalar_coordinates.shape == [1, 0]
	assert scalar_coordinates.size == 0

	zero_scalar := vtl.from_array([0], [])!
	zero_coordinates := vtl.argwhere[int](zero_scalar)!
	assert zero_coordinates.shape == [0, 0]
}

fn test_nonzero_returns_one_index_tensor_per_axis() ! {
	tensor := vtl.from_2d([[0, 2, 0], [3, 0, 4]])!
	indices := vtl.nonzero[int](tensor)!
	assert indices.len == 2
	assert indices[0].to_array() == [0, 1, 1]
	assert indices[1].to_array() == [1, 0, 2]

	transposed := vtl.nonzero[int](tensor.t()!)!
	assert transposed[0].to_array() == [0, 1, 2]
	assert transposed[1].to_array() == [1, 0, 1]

	empty := vtl.nonzero[int](vtl.from_array([]int{}, [0, 3])!)!
	assert empty.len == 2
	assert empty[0].shape == [0]
	assert empty[1].shape == [0]

	true_scalar := vtl.nonzero[int](vtl.from_array([7], [])!)!
	assert true_scalar.len == 1
	assert true_scalar[0].to_array() == [0]
	false_scalar := vtl.nonzero[int](vtl.from_array([0], [])!)!
	assert false_scalar.len == 1
	assert false_scalar[0].shape == [0]
}

fn test_advanced_index_pairs_coordinate_arrays() ! {
	tensor := vtl.from_2d([[10, 11, 12], [20, 21, 22]])!
	rows := vtl.from_1d([0, 1])!
	columns := vtl.from_1d([2, 0])!
	got := vtl.advanced_index[int](tensor, [rows, columns])!
	assert got.shape == [2]
	assert got.to_array() == [12, 20]
}

fn test_advanced_index_broadcasts_coordinates_and_supports_negative_values() ! {
	tensor := vtl.from_2d([[10, 11, 12], [20, 21, 22]])!
	rows := vtl.from_array[int]([0, 1], [2, 1])!
	columns := vtl.from_1d([1, 2, 0])!
	got := vtl.advanced_index[int](tensor, [rows, columns])!
	assert got.shape == [2, 3]
	assert got.to_array() == [11, 12, 10, 21, 22, 20]

	negative_rows := vtl.from_1d([-1, 0])!
	negative_columns := vtl.from_1d([-1, -2])!
	negative := vtl.advanced_index[int](tensor, [negative_rows, negative_columns])!
	assert negative.to_array() == [22, 11]
}

fn test_advanced_index_reads_views_and_returns_an_independent_copy() ! {
	tensor := vtl.from_2d([[10, 11, 12], [20, 21, 22]])!
	view := tensor.t()!
	rows := vtl.from_1d([0, 2])!
	columns := vtl.from_1d([1, 0])!
	mut got := vtl.advanced_index[int](view, [rows, columns])!
	assert got.shape == [2]
	assert got.to_array() == [20, 12]
	assert got.is_row_major_contiguous()
	got.set([0], 99)
	assert tensor.get([0, 1]) == 11
}

fn test_advanced_index_handles_scalar_and_empty_coordinates() ! {
	scalar := vtl.from_array[int]([7], [])!
	selected_scalar := vtl.advanced_index[int](scalar, [])!
	assert selected_scalar.shape.len == 0
	assert selected_scalar.get_nth[int](0) == 7

	tensor := vtl.from_2d([[1, 2], [3, 4]])!
	empty_rows := vtl.from_1d([]int{})!
	columns := vtl.from_1d([]int{})!
	empty := vtl.advanced_index[int](tensor, [empty_rows, columns])!
	assert empty.shape == [0]
	assert empty.size == 0
}

fn test_advanced_index_rejects_invalid_coordinate_inputs() {
	tensor := vtl.from_2d([[1, 2], [3, 4]])!
	if _ := vtl.advanced_index[int](tensor, [vtl.from_1d([0])!]) {
		assert false, 'advanced_index must require one coordinate tensor per axis'
	}
	rows := vtl.from_array[int]([0, 1], [2, 1])!
	columns := vtl.from_array[int]([0, 1, 0], [3])!
	depths := vtl.from_1d([0])!
	if _ := vtl.advanced_index[int](tensor, [rows, columns, depths]) {
		assert false, 'advanced_index must reject coordinate count mismatch'
	}
	bad_columns := vtl.from_1d([2])!
	if _ := vtl.advanced_index[int](tensor, [vtl.from_1d([0])!, bad_columns]) {
		assert false, 'advanced_index must reject out-of-range coordinates'
	}
	if _ := vtl.advanced_index[int](tensor, [vtl.from_1d([0, 1])!, vtl.from_1d([0, 1, 0])!]) {
		assert false, 'advanced_index must reject incompatible broadcast shapes'
	}
	if _ := vtl.advanced_index[int](vtl.from_array[int]([1], [])!, [vtl.from_1d([0])!]) {
		assert false, 'advanced_index must reject coordinates on scalar tensors'
	}
}

fn test_take_axis() {
	t := vtl.from_array([0, 1, 2, 3, 4, 5], [2, 3])!
	taken := t.take([2, 0, -1], 1)!
	assert taken.shape == [2, 3]
	assert taken.to_array() == [2, 0, 2, 5, 3, 5]

	rows := t.take([-1, 0], 0)!
	assert rows.shape == [2, 3]
	assert rows.to_array() == [3, 4, 5, 0, 1, 2]

	column_major := vtl.from_array([0, 3, 1, 4, 2, 5], [2, 3], memory: .col_major)!
	column_major_taken := column_major.take([2, 0], 1)!
	assert column_major_taken.is_col_major()
	assert column_major_taken.to_array() == [2, 0, 5, 3]

	columns := t.take([], -1)!
	assert columns.shape == [2, 0]
	assert columns.size() == 0
}

fn test_take_rejects_invalid_axes_and_indices() {
	t := vtl.ones[int]([2, 3])
	if _ := t.take([0], 2) {
		assert false, 'take must reject out-of-range axes'
	} else {
		assert true
	}
	if _ := t.take([3], 1) {
		assert false, 'take must reject out-of-range indices'
	} else {
		assert true
	}
	if _ := t.take([-3], 0) {
		assert false, 'take must reject negative indices below -axis_size'
	} else {
		assert true
	}
}

fn test_take_nd_replaces_axis_with_multidimensional_indices() {
	tensor := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	indices := vtl.from_array([0, 2, 1, 0], [2, 2])!
	got := tensor.take_nd(indices, -1)!
	assert got.shape == [2, 2, 2]
	assert got.to_array() == [1, 3, 2, 1, 4, 6, 5, 4]
	column_major := vtl.from_array([0, 3, 1, 4, 2, 5], [2, 3], memory: .col_major)!
	columns := column_major.take_nd(vtl.from_1d([2, 0])!, 1)!
	assert columns.is_col_major()
	assert columns.to_array() == [2, 0, 5, 3]
	if _ := tensor.take_nd(vtl.from_1d([3])!, 1) {
		assert false, 'take_nd must reject out-of-range indices'
	}
	if _ := tensor.take_nd(indices, 2) {
		assert false, 'take_nd must reject out-of-range axes'
	}
}

fn test_take_nd_supports_scalar_and_empty_index_tensors() {
	tensor := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	scalar_index := vtl.from_array([-1], [])!
	taken_scalar := tensor.take_nd(scalar_index, 1)!
	assert taken_scalar.shape == [2]
	assert taken_scalar.to_array() == [3, 6]
	empty_indices := vtl.from_array([]int{}, [0, 2])!
	empty := tensor.take_nd(empty_indices, 1)!
	assert empty.shape == [2, 0, 2]
	assert empty.size() == 0
}

fn test_take_flat_preserves_index_shape_and_rejects_out_of_range() {
	tensor := vtl.from_2d([[1, 2, 3], [4, 5, 6]])!
	indices := vtl.from_array([5, 0, -2, 3], [2, 2])!
	got := tensor.take_flat(indices)!
	assert got.shape == [2, 2]
	assert got.to_array() == [6, 1, 5, 4]
	transposed := tensor.transpose([1, 0])!
	logical_flat := transposed.take_flat(vtl.from_1d([0, 1, 4, 5])!)!
	assert logical_flat.to_array() == [1, 4, 3, 6]
	bad_indices := vtl.from_1d([6])!
	if _ := tensor.take_flat(bad_indices) {
		assert false, 'take_flat must reject indices outside the flattened tensor'
	}
}

fn test_take_along_axis() {
	t := vtl.from_array([0, 1, 2, 3, 4, 5], [2, 3])!
	indices := vtl.from_array([2, 0, 1, -1], [2, 2])!
	taken := t.take_along_axis(indices, 1)!
	assert taken.shape == [2, 2]
	assert taken.to_array() == [2, 0, 4, 5]

	if _ := t.take_along_axis(vtl.ones[int]([1, 2]), 1) {
		assert false, 'take_along_axis must reject mismatched non-axis dimensions'
	} else {
		assert true
	}
}

fn test_get() {
	t := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [5, 2])!
	assert t.size() == 10
	assert t.get([0, 0]) == 1
	assert t.get([0, 1]) == 2
	assert t.get([1, 0]) == 3
	assert t.get([1, 1]) == 4
	assert t.get([2, 0]) == 5
	assert t.get([2, 1]) == 6
	assert t.get([3, 0]) == 7
	assert t.get([3, 1]) == 8
	assert t.get([4, 0]) == 9
	assert t.get([4, 1]) == 10
}

fn test_get_nth() {
	t := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [5, 2])!
	assert t.size() == 10
	assert t.get_nth(0) == 1
	assert t.get_nth(1) == 2
	assert t.get_nth(2) == 3
	assert t.get_nth(3) == 4
	assert t.get_nth(4) == 5
	assert t.get_nth(5) == 6
	assert t.get_nth(6) == 7
	assert t.get_nth(7) == 8
	assert t.get_nth(8) == 9
	assert t.get_nth(9) == 10
}

fn test_get_nth_and_to_array_preserve_gapped_row_strides() ! {
	base := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8], [4, 2])!
	view := base.slice([0, 2], []int{})!
	gapped := view.as_strided([2, 2], [3, 1])!
	assert !gapped.is_row_major_contiguous()
	assert gapped.get_nth(2) == 4
	assert gapped.to_array() == [1, 2, 4, 5]
}

fn test_offset_index() {
	t := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [5, 2])!
	assert t.size() == 10
	assert t.offset_index([0, 0]) == 0
	assert t.offset_index([0, 1]) == 1
	assert t.offset_index([1, 0]) == 2
	assert t.offset_index([1, 1]) == 3
	assert t.offset_index([2, 0]) == 4
	assert t.offset_index([2, 1]) == 5
	assert t.offset_index([3, 0]) == 6
	assert t.offset_index([3, 1]) == 7
	assert t.offset_index([4, 0]) == 8
	assert t.offset_index([4, 1]) == 9
}

fn test_nth_index() {
	t := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [5, 2])!
	assert t.size() == 10
	assert t.nth_index(0) == [0, 0]
	assert t.nth_index(1) == [0, 1]
	assert t.nth_index(2) == [1, 0]
	assert t.nth_index(3) == [1, 1]
	assert t.nth_index(4) == [2, 0]
	assert t.nth_index(5) == [2, 1]
	assert t.nth_index(6) == [3, 0]
	assert t.nth_index(7) == [3, 1]
	assert t.nth_index(8) == [4, 0]
	assert t.nth_index(9) == [4, 1]
}

fn test_unique_flattens_and_sorts_values() ! {
	tensor := vtl.from_2d([[3, 1, 3], [2, 1, 2]])!
	assert vtl.unique(tensor)!.to_array() == [1, 2, 3]
	assert vtl.unique(vtl.from_1d[int]([]int{})!)!.to_array() == []int{}
}

fn test_unique_counts_reports_sorted_occurrences() ! {
	tensor := vtl.from_1d([4, 2, 4, 1, 2, 4])!
	result := vtl.unique_counts(tensor)!
	assert result.values.to_array() == [1, 2, 4]
	assert result.counts.to_array() == [1, 2, 3]
	assert vtl.unique_inverse(tensor)!.to_array() == [2, 1, 2, 0, 1, 2]
	assert vtl.unique_first_indices(tensor)!.to_array() == [3, 1, 0]
}

fn test_unique_counts_groups_nan_values() ! {
	tensor := vtl.from_1d([math.nan(), 3.0, math.nan(), -2.0, 3.0])!
	result := vtl.unique_counts(tensor)!
	assert result.values.get_nth[f64](0) == -2.0
	assert result.values.get_nth[f64](1) == 3.0
	assert math.is_nan(result.values.get_nth[f64](2))
	assert result.counts.to_array() == [1, 2, 2]
	assert vtl.unique_inverse(tensor)!.to_array() == [2, 1, 2, 0, 1]
	assert vtl.unique_first_indices(tensor)!.to_array() == [3, 1, 0]
}
