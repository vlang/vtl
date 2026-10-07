module core

import math
import vtl

fn test_masked_array_broadcast_fill_compress_and_global_reductions() ! {
	values := vtl.from_array[f64]([1, 2, 3, 4, 5, 6], [2, 3])!
	mask := vtl.from_array[bool]([false, true, false, true, true, true], [2, 3])!
	data := vtl.masked_array(values, mask)!
	assert data.values.shape == [2, 3]
	assert data.count() == 2
	assert data.filled(-1.0).to_array() == [1, -1, 3, -1, -1, -1]
	assert data.compressed()!.to_array() == [1, 3]
	sum := data.sum()
	assert sum.value == 4.0 && !sum.is_masked
	mean := data.mean()
	assert mean.value == 2.0 && !mean.is_masked
	product := data.prod()
	assert product.value == 3.0 && !product.is_masked
	minimum := data.min()
	maximum := data.max()
	assert minimum.value == 1.0 && !minimum.is_masked
	assert maximum.value == 3.0 && !maximum.is_masked
	variance := data.variance(1)!
	assert variance.value == 2.0 && !variance.is_masked
	deviation := data.std(1)!
	assert math.sqrt(2.0) - 1e-12 < deviation.value && deviation.value < math.sqrt(2.0) + 1e-12

	row_mask := vtl.from_array[bool]([true, false, true], [3])!
	broadcast_data := vtl.masked_array(values, row_mask)!
	assert broadcast_data.mask.shape == [2, 3]
	assert broadcast_data.compressed()!.to_array() == [2, 5]
	assert broadcast_data.filled(0.0).to_array() == [0, 2, 0, 0, 5, 0]
}

fn test_masked_array_axis_reductions_preserve_masked_empty_slices() ! {
	values := vtl.from_array[int]([1, 2, 3, 4, 5, 6], [2, 3])!
	mask := vtl.from_array[bool]([false, true, false, true, true, true], [2, 3])!
	data := vtl.masked_array(values, mask)!
	sums := data.sum_along_axis(0, false)!
	assert sums.values.shape == [3]
	assert sums.values.to_array() == [1, 0, 3]
	assert sums.mask.to_array() == [false, true, false]
	means := data.mean_along_axis(-1, true)!
	assert means.values.shape == [2, 1]
	assert means.values.to_array()[0] == 2.0
	assert math.is_nan(means.values.to_array()[1])
	assert means.mask.to_array() == [false, true]
	products := data.prod_along_axis(1, false)!
	assert products.values.to_array() == [3, 1]
	assert products.mask.to_array() == [false, true]
	minima := data.min_along_axis(1, false)!
	maxima := data.max_along_axis(1, false)!
	assert minima.values.to_array() == [1, 0]
	assert minima.mask.to_array() == [false, true]
	assert maxima.values.to_array() == [3, 0]
	assert maxima.mask.to_array() == [false, true]
	variances := data.variance_along_axis(1, 1, false)!
	assert variances.values.to_array()[0] == 2.0
	assert math.is_nan(variances.values.to_array()[1])
	assert variances.mask.to_array() == [false, true]
	deviations := data.std_along_axis(1, 1, false)!
	assert deviations.mask.to_array() == [false, true]
}

fn test_masked_array_all_masked_and_invalid_shapes_or_axes() ! {
	values := vtl.from_1d([2.0, 4.0])!
	all_masked := vtl.masked_array(values, vtl.from_1d[bool]([true, true])!)!
	sum := all_masked.sum()
	assert sum.value == 0 && sum.is_masked
	mean := all_masked.mean()
	assert math.is_nan(mean.value) && mean.is_masked
	product := all_masked.prod()
	assert product.value == 1 && product.is_masked
	assert all_masked.min().is_masked
	assert all_masked.max().is_masked
	assert all_masked.variance(0)!.is_masked
	assert all_masked.std(0)!.is_masked
	assert all_masked.count() == 0
	assert all_masked.compressed()!.size == 0
	if _ := vtl.masked_array(values, vtl.from_1d[bool]([true, false, true])!) {
		assert false, 'masked_array must reject incompatible shapes'
	}
	if _ := all_masked.mean_along_axis(1, false) {
		assert false, 'masked axis reduction must reject out-of-range axes'
	}
}

fn test_masked_array_broadcasts_empty_dimensions() ! {
	values := vtl.zeros[f64]([0], vtl.TensorData{})
	mask := vtl.from_1d[bool]([false])!
	data := vtl.masked_array(values, mask)!
	assert data.values.shape == [0]
	assert data.mask.shape == [0]
	assert data.count() == 0
	assert data.compressed()!.size == 0
}

fn test_masked_array_multi_axis_reductions_and_empty_axes() ! {
	values := vtl.from_array[f64]([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2])!
	mask := vtl.from_array[bool]([false, true, false, true, true, true, true, true], [
		2,
		2,
		2,
	])!
	data := vtl.masked_array(values, mask)!
	sums := data.sum_along_axes([0, 2], false)!
	assert sums.values.shape == [2]
	assert sums.values.to_array() == [1.0, 3.0]
	assert sums.mask.to_array() == [false, false]
	products := data.prod_along_axes([0, 2], false)!
	assert products.values.to_array() == [1.0, 3.0]
	assert products.mask.to_array() == [false, false]
	assert data.min_along_axes([0, 2], false)!.values.to_array() == [1.0, 3.0]
	assert data.max_along_axes([0, 2], false)!.values.to_array() == [1.0, 3.0]
	variances := data.variance_along_axes([0, 2], 0, true)!
	assert variances.values.shape == [1, 2, 1]
	assert variances.values.to_array() == [0.0, 0.0]
	assert variances.mask.to_array() == [false, false]
	if _ := data.variance(-1) {
		assert false, 'variance must reject negative ddof'
	}
	means := data.mean_along_axes([-3, -1], true)!
	assert means.values.shape == [1, 2, 1]
	assert means.values.to_array() == [1.0, 3.0]
	assert means.mask.to_array() == [false, false]
	unchanged := data.sum_along_axes([], false)!
	assert unchanged.values.to_array() == values.to_array()
	assert unchanged.mask.to_array() == mask.to_array()
	elementwise_mean := data.mean_along_axes([], false)!
	assert elementwise_mean.values.to_array() == values.as_f64().to_array()
	assert elementwise_mean.mask.to_array() == mask.to_array()
	if _ := data.sum_along_axes([0, -3], false) {
		assert false, 'multi-axis reductions must reject duplicate axes'
	}
}

fn test_masked_array_broadcast_arithmetic_unions_masks() ! {
	left_values := vtl.from_array[f64]([1, 2, 3, 4, 5, 6], [2, 3])!
	left_mask := vtl.from_1d[bool]([false, true, false])!
	left := vtl.masked_array(left_values, left_mask)!
	right_values := vtl.from_1d[f64]([10, 20, 30])!
	right_mask := vtl.from_array[bool]([true, false], [2, 1])!
	right := vtl.masked_array(right_values, right_mask)!

	sums := left.add(right)!
	assert sums.values.shape == [2, 3]
	assert sums.values.to_array() == [11, 22, 33, 14, 25, 36]
	assert sums.mask.to_array() == [true, true, true, false, true, false]

	differences := left.subtract(right)!
	assert differences.values.to_array() == [-9, -18, -27, -6, -15, -24]
	products := left.multiply(right)!
	assert products.values.to_array() == [10, 40, 90, 40, 100, 180]
	quotients := left.divide(right)!
	assert quotients.values.to_array() == [0.1, 0.1, 0.1, 0.4, 0.25, 0.2]
	less := left.less_than(right)!
	assert less.values.to_array() == [true, true, true, true, true, true]
	assert less.mask.to_array() == sums.mask.to_array()
	equal := left.equal(right)!
	assert equal.values.to_array() == [false, false, false, false, false, false]
	assert left.not_equal(right)!.values.to_array() == [true, true, true, true, true, true]
	assert left.less_equal(right)!.values.to_array() == less.values.to_array()
	assert left.greater_than(right)!.values.to_array() == [false, false, false, false, false, false]
	assert left.greater_equal(right)!.values.to_array() == [false, false, false, false, false,
		false]

	shifted := left.add_scalar(2.0)!
	assert shifted.values.to_array() == [3, 4, 5, 6, 7, 8]
	assert shifted.mask.to_array() == left.mask.to_array()
	assert left.subtract_scalar(1.0)!.values.to_array() == [0, 1, 2, 3, 4, 5]
	assert left.multiply_scalar(2.0)!.values.to_array() == [2, 4, 6, 8, 10, 12]
	assert left.divide_scalar(2.0)!.values.to_array() == [0.5, 1, 1.5, 2, 2.5, 3]
}

fn test_masked_array_mixed_index_keeps_values_and_mask_aligned() ! {
	values := vtl.from_array[int]([1, 2, 3, 4, 5, 6], [2, 3])!
	mask := vtl.from_array[bool]([false, true, false, true, false, true], [2, 3])!
	data := vtl.masked_array(values, mask)!
	row := data.mixed_index([vtl.integer_index(-1), vtl.full_index()])!
	assert row.values.shape == [3]
	assert row.values.to_array() == [4, 5, 6]
	assert row.mask.to_array() == [true, false, true]
	column := data.mixed_index([vtl.full_index(), vtl.integer_index(1)])!
	assert column.values.to_array() == [2, 5]
	assert column.mask.to_array() == [true, false]
	if _ := data.mixed_index([vtl.integer_index(2)]) {
		assert false, 'masked mixed indexing must preserve bounds checks'
	}
}

fn test_masked_array_take_and_put_along_axis_keep_masks_aligned() ! {
	values := vtl.from_array[int]([1, 2, 3, 4, 5, 6], [2, 3])!
	mask := vtl.from_array[bool]([false, true, false, true, false, true], [2, 3])!
	data := vtl.masked_array(values, mask)!
	indices := vtl.from_array[int]([0, 1], [2, 1])!
	updates := vtl.masked_array(vtl.from_array[int]([9, 8], [2, 1])!,
		vtl.from_array[bool]([true, true], [2, 1])!)!

	updated := data.put_along_axis(indices, updates, 1)!
	assert updated.values.to_array() == [9, 2, 3, 4, 8, 6]
	assert updated.mask.to_array() == [true, true, false, true, true, true]
	assert data.values.to_array() == values.to_array()
	assert data.mask.to_array() == mask.to_array()

	taken := updated.take_along_axis(indices, 1)!
	assert taken.values.to_array() == [9, 8]
	assert taken.mask.to_array() == [true, true]
}
