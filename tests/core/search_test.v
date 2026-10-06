module core

import vtl

fn test_searchsorted_sides_and_multidimensional_queries() ! {
	sorted := vtl.from_1d([1, 3, 3, 5])!
	queries := vtl.from_2d([[0, 1, 2], [3, 4, 5], [6, 7, 3]])!
	assert vtl.searchsorted(sorted, queries, .left)!.shape == queries.shape
	assert vtl.searchsorted(sorted, queries, .left)!.to_array() == [0, 0, 1, 1, 3, 3, 4, 4, 1]
	assert vtl.searchsorted(sorted, queries, .right)!.to_array() == [0, 1, 1, 3, 3, 4, 4, 4, 3]
}

fn test_digitize_increasing_and_decreasing_edges() ! {
	values := vtl.from_1d([0, 1, 2, 3, 4, 5, 6])!
	ascending := vtl.from_1d([1, 3, 5])!
	assert vtl.digitize(values, ascending, false)!.to_array() == [0, 1, 1, 2, 2, 3, 3]
	assert vtl.digitize(values, ascending, true)!.to_array() == [0, 0, 1, 1, 2, 2, 3]
	decreasing := vtl.from_1d([5, 3, 1])!
	assert vtl.digitize(values, decreasing, false)!.to_array() == [3, 2, 2, 1, 1, 0, 0]
	assert vtl.digitize(values, decreasing, true)!.to_array() == [3, 3, 2, 2, 1, 1, 0]
}

fn test_search_helpers_reject_invalid_sorted_inputs() ! {
	values := vtl.from_1d([1, 4, 2])!
	queries := vtl.from_1d([2])!
	if _ := vtl.searchsorted(values, queries, .left) {
		assert false, 'unsorted search values must return an error'
	} else {
		assert true
	}
	if _ := vtl.digitize(queries, values, false) {
		assert false, 'non-monotonic bins must return an error'
	} else {
		assert true
	}
}
