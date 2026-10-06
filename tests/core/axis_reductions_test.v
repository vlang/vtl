module core

import vtl
import vtl.stats as vtl_stats

fn test_sum_and_product_along_axis_keepdims_and_negative_axis() ! {
	values := vtl.from_array([1, 2, 3, 4, 5, 6], [2, 3])!

	row_sums := vtl_stats.sum_along_axis(values, -1, false)!
	assert row_sums.shape == [2]
	assert row_sums.to_array() == [6, 15]
	column_sums := vtl_stats.sum_along_axis(values, 0, true)!
	assert column_sums.shape == [1, 3]
	assert column_sums.to_array() == [5, 7, 9]
	row_products := vtl_stats.product_along_axis(values, 1, true)!
	assert row_products.shape == [2, 1]
	assert row_products.to_array() == [6, 120]
}

fn test_axis_reductions_empty_reduced_dimension() ! {
	values := vtl.from_array([]int{}, [2, 0])!
	assert vtl_stats.sum_along_axis(values, 1, false)!.to_array() == [0, 0]
	assert vtl_stats.product_along_axis(values, 1, true)!.to_array() == [1, 1]
}

fn test_axis_reductions_reject_invalid_axes() ! {
	values := vtl.from_1d([1, 2])!
	if _ := vtl_stats.sum_along_axis(values, 1, false) {
		assert false, 'out-of-range axes must return an error'
	}
	if _ := vtl_stats.product_along_axis(values, -2, false) {
		assert false, 'out-of-range axes must return an error'
	}
	if _ := vtl_stats.sum_along_axis(vtl.from_array([1], []int{})!, 0, true) {
		assert false, 'scalar axis reduction must return an error'
	}
}
