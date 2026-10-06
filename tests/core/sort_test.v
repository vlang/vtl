module core

import vtl
import math

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
