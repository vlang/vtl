module core

import vtl
import math

fn test_partition_axis_places_multiple_order_statistics() {
	t := vtl.from_2d([[9, 2, 1, 7], [3, 8, 4, 5]])!
	got := vtl.partition_axis(t, [1, 2], 1)!
	assert got.get([0, 1]) == 2
	assert got.get([0, 2]) == 7
	assert got.get([1, 1]) == 4
	assert got.get([1, 2]) == 5
	for row in 0 .. 2 {
		for col in 0 .. 4 {
			if col < 1 {
				assert got.get([row, col]) <= got.get([row, 1])
			} else if col > 1 {
				assert got.get([row, col]) >= got.get([row, 1])
			}
			if col < 2 {
				assert got.get([row, col]) <= got.get([row, 2])
			} else if col > 2 {
				assert got.get([row, col]) >= got.get([row, 2])
			}
		}
	}
}

fn test_argpartition_indices_match_values_and_nan_order() {
	t := vtl.from_2d([[9.0, 2.0, 1.0, 7.0], [math.nan(), 3.0, 2.0, 1.0]])!
	indices := vtl.argpartition_axis(t, [1], -1)!
	for row in 0 .. 2 {
		pivot_index := indices.get([row, 1])
		pivot_value := t.get([row, pivot_index])
		for col in 0 .. 4 {
			value := t.get([row, indices.get([row, col])])
			if col < 1 {
				assert value <= pivot_value
			} else if col > 1 {
				assert value >= pivot_value || math.is_nan(value)
			}
		}
	}
}

fn test_partition_rejects_invalid_kth_values() {
	t := vtl.from_1d([3, 1, 2])!
	if _ := vtl.partition(t, 3) {
		assert false, 'partition must reject kth equal to the axis size'
	}
	if _ := vtl.partition_axis(t, []int{}, 0) {
		assert false, 'partition_axis must require at least one kth'
	}
}

fn test_sort_and_argsort_default_to_last_axis() {
	t := vtl.from_array([3, 1, 2, 6, 4, 5], [2, 3])!
	assert vtl.sort(t)!.to_array() == [1, 2, 3, 4, 5, 6]
	assert vtl.argsort(t)!.to_array() == [1, 2, 0, 1, 2, 0]
}

fn test_sort_axis_handles_negative_axis_and_stable_ties() {
	t := vtl.from_array([3, 1, 2, 2, 4, 1], [3, 2])!
	assert vtl.sort_axis(t, 0)!.to_array() == [2, 1, 3, 1, 4, 2]
	assert vtl.argsort_axis(t, -2)!.to_array() == [1, 0, 0, 2, 2, 1]
}

fn test_sort_and_argsort_put_nan_values_last() {
	t := vtl.from_1d([math.nan(), 2.0, -1.0, math.nan(), 2.0])!
	sorted := vtl.sort(t)!
	indices := vtl.argsort(t)!
	assert sorted.get_nth(0) == -1.0
	assert sorted.get_nth(1) == 2.0
	assert sorted.get_nth(2) == 2.0
	assert math.is_nan(sorted.get_nth(3))
	assert math.is_nan(sorted.get_nth(4))
	assert indices.to_array() == [2, 1, 4, 0, 3]
}

fn test_sort_rejects_invalid_axes_and_supports_empty_axis() {
	t := vtl.ones[int]([2, 0])
	assert vtl.sort_axis(t, -1)!.shape == [2, 0]
	if _ := vtl.sort_axis(t, 2) {
		assert false, 'sort must reject out-of-range axes'
	} else {
		assert true
	}
	if _ := vtl.sort(vtl.ones[int]([])) {
		assert false, 'sort must reject rank-zero tensors'
	} else {
		assert true
	}
}
