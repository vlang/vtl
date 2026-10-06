module core

import vtl

fn test_searchsorted_sides_and_multidimensional_queries() ! {
	sorted := vtl.from_1d([1, 3, 3, 5])!
	queries := vtl.from_2d([[0, 1, 2], [3, 4, 5], [6, 7, 3]])!
	assert vtl.searchsorted(sorted, queries, .left)!.shape == queries.shape
	assert vtl.searchsorted(sorted, queries, .left)!.to_array() == [0, 0, 1, 1, 3, 3, 4, 4, 1]
	assert vtl.searchsorted(sorted, queries, .right)!.to_array() == [0, 1, 1, 3, 3, 4, 4, 4, 3]
}

fn test_searchsorted_descending_sides_and_duplicate_values() ! {
	descending := vtl.from_1d([9, 7, 7, 4, 1])!
	queries := vtl.from_1d([10, 9, 8, 7, 6, 1, 0])!
	assert vtl.searchsorted_descending(descending, queries, .left)!.to_array() == [0, 0, 1, 1,
		3, 4, 5]
	assert vtl.searchsorted_descending(descending, queries, .right)!.to_array() == [0, 1, 1, 3,
		3, 5, 5]
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

fn test_digitize_rejects_non_monotonic_bins() ! {
	values := vtl.from_1d([1, 4, 2])!
	queries := vtl.from_1d([2])!
	if _ := vtl.digitize(queries, values, false) {
		assert false, 'non-monotonic bins must return an error'
	} else {
		assert true
	}
}
