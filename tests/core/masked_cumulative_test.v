module core

import vtl

fn test_masked_cumulative_sum_skips_masked_values_and_preserves_shape() ! {
	values := vtl.from_array([1, 100, 3, 4, 5, 200], [2, 3])!
	mask := vtl.from_array[bool]([false, true, false, false, false, true], [2, 3])!
	data := vtl.masked_array(values, mask)!

	rows := data.cumsum_along_axis(-1)!
	assert rows.values.shape == [2, 3]
	assert rows.values.to_array() == [1, 1, 4, 4, 9, 9]
	assert rows.mask.to_array() == mask.to_array()

	flat := data.cumsum()!
	assert flat.values.shape == [6]
	assert flat.values.to_array() == [1, 1, 4, 8, 13, 13]
	assert flat.mask.to_array() == mask.to_array()
}

fn test_masked_cumulative_product_skips_masked_values() ! {
	values := vtl.from_array([2, 100, 3, 4, 5, 200], [2, 3])!
	mask := vtl.from_array[bool]([false, true, false, false, false, true], [2, 3])!
	data := vtl.masked_array(values, mask)!

	rows := data.cumprod_along_axis(1)!
	assert rows.values.to_array() == [2, 2, 6, 4, 20, 20]
	assert rows.mask.to_array() == mask.to_array()

	flat := data.cumprod()!
	assert flat.values.to_array() == [2, 2, 6, 24, 120, 120]
	assert flat.mask.to_array() == mask.to_array()
}

fn test_masked_cumulative_operations_support_wider_output_types() ! {
	values := vtl.from_array([i8(100), 4, 100, 4], [2, 2])!
	mask := vtl.from_array[bool]([false, true, false, false], [2, 2])!
	data := vtl.masked_array(values, mask)!

	sums := data.cumsum_along_axis_as[i64](1)!
	assert sums.values.to_array() == [100, 100, 100, 104]
	assert sums.mask.to_array() == mask.to_array()

	products := data.cumprod_as[i64]()!
	assert products.values.to_array() == [100, 100, 10000, 40000]
	assert products.mask.to_array() == mask.to_array()
}

fn test_masked_cumulative_operations_handle_all_masked_and_empty_inputs() ! {
	values := vtl.from_array([1, 2, 3, 4], [2, 2])!
	all_masked := vtl.masked_array(values, vtl.ones[bool]([2, 2], vtl.TensorData{}))!
	assert all_masked.cumsum_along_axis(0)!.mask.to_array() == [true, true, true, true]
	assert all_masked.cumprod()!.mask.to_array() == [true, true, true, true]

	empty_values := vtl.zeros[f64]([2, 0], vtl.TensorData{})
	empty_mask := vtl.zeros[bool]([2, 0], vtl.TensorData{})
	empty := vtl.masked_array(empty_values, empty_mask)!
	assert empty.cumsum_along_axis(-1)!.values.shape == [2, 0]
	assert empty.cumprod()!.values.shape == [0]
}

fn test_masked_cumulative_operations_reject_invalid_axes() ! {
	values := vtl.from_array([1, 2, 3, 4], [2, 2])!
	mask := vtl.zeros[bool]([2, 2], vtl.TensorData{})
	data := vtl.masked_array(values, mask)!
	if _ := data.cumsum_along_axis(2) {
		assert false, 'masked cumsum must reject an out-of-range axis'
	}
	if _ := data.cumprod_along_axis(-3) {
		assert false, 'masked cumprod must reject an out-of-range axis'
	}
}
