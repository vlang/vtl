module core

import vtl
import vtl.stats as vtl_stats
import math.complex as cmplx

fn test_multi_axis_sum_and_product() ! {
	values := vtl.from_array([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])!
	sums := vtl_stats.sum_along_axes(values, [0, 2], false)!
	assert sums.shape == [2]
	assert sums.to_array() == [14, 22]
	transposed := values.transpose([2, 1, 0])!
	transposed_sums := vtl_stats.sum_along_axes(transposed, [0, 2], false)!
	assert transposed_sums.shape == [2]
	assert transposed_sums.to_array() == [14, 22]

	products := vtl_stats.product_along_axes(values, [-1, 0], true)!
	assert products.shape == [1, 2, 1]
	assert products.to_array() == [60, 672]
}

fn test_multi_axis_empty_axes_and_empty_reduced_dimensions() ! {
	values := vtl.from_array([1, 2, 3, 4], [2, 2])!
	unchanged := vtl_stats.sum_along_axes(values, []int{}, false)!
	assert unchanged.shape == values.shape
	assert unchanged.to_array() == values.to_array()
	scalar := vtl.from_array([5], []int{})!
	unchanged_scalar := vtl_stats.sum_along_axes(scalar, []int{}, false)!
	assert unchanged_scalar.rank() == 0
	assert unchanged_scalar.to_array() == [5]

	empty := vtl.from_array([]int{}, [2, 0, 3])!
	sums := vtl_stats.sum_along_axes(empty, [1], false)!
	products := vtl_stats.product_along_axes(empty, [1], true)!
	assert sums.shape == [2, 3]
	assert sums.to_array() == [0, 0, 0, 0, 0, 0]
	assert products.shape == [2, 1, 3]
	assert products.to_array() == [1, 1, 1, 1, 1, 1]
}

fn test_multi_axis_reduce_all_to_scalar() ! {
	values := vtl.from_array([1, 2, 3, 4], [2, 2])!
	result := vtl_stats.sum_along_axes(values, [0, 1], false)!
	assert result.rank() == 0
	assert result.to_array() == [10]
	kept := vtl_stats.sum_along_axes(values, [0, 1], true)!
	assert kept.shape == [1, 1]
	assert kept.to_array() == [10]
}

fn test_multi_axis_reductions_reject_invalid_axes() ! {
	values := vtl.from_array([1, 2, 3, 4], [2, 2])!
	if _ := vtl_stats.sum_along_axes(values, [0, -2], false) {
		assert false, 'duplicate axes must return an error'
	}
	if _ := vtl_stats.product_along_axes(values, [2], true) {
		assert false, 'out-of-range axes must return an error'
	}
}

fn test_complex_multi_axis_reductions() ! {
	values := vtl.from_array([
		cmplx.Complex{ re: 1, im: 2 },
		cmplx.Complex{ re: 3, im: 4 },
		cmplx.Complex{ re: 5, im: 6 },
		cmplx.Complex{ re: 7, im: 8 },
	], [2, 2])!
	sum := vtl_stats.sum_along_axes(values, [0, 1], false)!
	assert sum.rank() == 0
	assert sum.get_nth(0) == cmplx.Complex{ re: 16, im: 20 }
	product := vtl_stats.product_along_axes(values, [0, 1], true)!
	assert product.shape == [1, 1]
	assert product.get_nth(0) == cmplx.Complex{ re: -755, im: -540 }
}
