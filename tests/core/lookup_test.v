module core

import vtl
import math

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
