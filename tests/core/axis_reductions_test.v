module core

import vtl
import vtl.stats as vtl_stats
import math.complex as cmplx

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
	empty_rows := vtl.from_array([]f64{}, [0, 2])!
	assert vtl_stats.sum_along_axis(empty_rows, 0, true)!.to_array() == [0.0, 0.0]
}

fn test_sum_axis0_contiguous_rank_three() ! {
	values := vtl.from_array[f64]([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])!
	assert vtl_stats.sum_along_axis(values, 0, false)!.shape == [2, 2]
	assert vtl_stats.sum_along_axis(values, 0, false)!.to_array() == [6.0, 8, 10, 12]
	assert vtl_stats.sum_along_axis(values, 0, true)!.shape == [1, 2, 2]
	f32_values := vtl.from_array[f32]([1, 2, 3, 4, 5, 6], [2, 3])!
	assert vtl_stats.sum_along_axis(f32_values, 0, false)!.to_array() == [f32(5), 7, 9]
	wide_f32_values := vtl.from_array[f32]([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
		16, 17, 18, 19, 20, 21, 22], [2, 11])!
	assert vtl_stats.sum_along_axis(wide_f32_values, 0, false)!.to_array() == [
		f32(13),
		15,
		17,
		19,
		21,
		23,
		25,
		27,
		29,
		31,
		33,
	]
}

fn test_sum_axis0_noncontiguous_view_uses_strided_path() ! {
	values := vtl.from_array[f64]([1, 2, 3, 4, 5, 6], [2, 3])!
	transposed := values.transpose([1, 0])!
	assert !transposed.is_row_major_contiguous()
	assert vtl_stats.sum_along_axis(transposed, 0, false)!.to_array() == [6.0, 15]
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

fn test_complex_axis_sum_and_product() ! {
	values := vtl.from_array([
		cmplx.Complex{ re: 1, im: 2 },
		cmplx.Complex{ re: 3, im: 4 },
		cmplx.Complex{ re: 5, im: 6 },
		cmplx.Complex{ re: 7, im: 8 },
	], [2, 2])!
	sums := vtl_stats.sum_along_axis(values, 1, false)!
	assert sums.to_array() == [cmplx.Complex{ re: 4, im: 6 }, cmplx.Complex{ re: 12, im: 14 }]
	column_sums := vtl_stats.sum_along_axis(values, 0, false)!
	assert column_sums.to_array() == [cmplx.Complex{ re: 6, im: 8 }, cmplx.Complex{ re: 10, im: 12 }]
	products := vtl_stats.product_along_axis(values, 1, false)!
	assert products.to_array() == [cmplx.Complex{ re: -5, im: 10 }, cmplx.Complex{ re: -13, im: 82 }]

	empty := vtl.from_array([]cmplx.Complex{}, [2, 0])!
	assert vtl_stats.sum_along_axis(empty, 1, false)!.to_array() == [
		cmplx.Complex{ re: 0, im: 0 },
		cmplx.Complex{ re: 0, im: 0 },
	]
	assert vtl_stats.product_along_axis(empty, 1, false)!.to_array() == [
		cmplx.Complex{ re: 1, im: 0 },
		cmplx.Complex{ re: 1, im: 0 },
	]
}
