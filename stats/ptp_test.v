module stats

import math
import vtl

fn test_ptp_global_and_axis_shapes() {
	values := vtl.from_array[f64]([1, 7, 2, 4, 3, 9], [2, 3])!
	assert ptp[f64](values)! == 8

	columns := ptp_along_axis[f64](values, 0, false)!
	assert columns.shape == [3]
	assert columns.to_array() == [3, 4, 7]

	rows := ptp_along_axis[f64](values, -1, true)!
	assert rows.shape == [2, 1]
	assert rows.to_array() == [6, 6]
	all := ptp_along_axes[f64](values, [0, -1], true)!
	assert all.shape == [1, 1]
	assert all.to_array() == [8]
	groups := vtl.from_array[f64]([1, 2, 3, 4, 5, 6, 7, 20], [2, 2, 2])!
	group_ranges := ptp_along_axes[f64](groups, [1, 2], false)!
	assert group_ranges.to_array() == [3, 15]

	f32_values := vtl.from_1d[f32]([-12, 1, 2, 3, 4, 5, 6, 7, 30])!
	assert ptp[f32](f32_values)! == 42

	mut f64_block_values := []f64{len: 21}
	f64_block_values[12] = -20
	f64_block_values[20] = 70
	f64_block_tensor := vtl.from_1d[f64](f64_block_values)!
	assert ptp[f64](f64_block_tensor)! == 90

	mut f32_block_values := []f32{len: 41}
	f32_block_values[19] = -35
	f32_block_values[40] = 80
	f32_block_tensor := vtl.from_1d[f32](f32_block_values)!
	assert ptp[f32](f32_block_tensor)! == 115
}

fn test_ptp_multiple_axes_empty_axes_and_integer_dtype() {
	values := vtl.from_array[i32]([1, 7, 2, 4, 3, 9, 8, 6], [2, 2, 2])!
	spread := ptp_along_axes[i32](values, [0, -1], false)!
	assert spread.shape == [2]
	assert spread.to_array() == [8, 6]

	unchanged := ptp_along_axes[i32](values, [], false)!
	assert unchanged.shape == values.shape
	assert unchanged.to_array() == [0, 0, 0, 0, 0, 0, 0, 0]
	with_nan := vtl.from_1d[f64]([math.nan(), 3])!
	unchanged_nan := ptp_along_axes[f64](with_nan, [], false)!
	assert math.is_nan(unchanged_nan.get_nth(0))
	assert unchanged_nan.get_nth(1) == 0

	overflow := vtl.from_1d[i8]([i8(-128), i8(127)])!
	assert ptp[i8](overflow)! == i8(-1)
}

fn test_ptp_multiple_axes_nan_keepdims_and_strided_views() ! {
	values := vtl.from_array[f64]([math.nan(), 2, 3, 4, 5, math.nan(), 7, 8], [2, 2, 2])!
	spread := ptp_along_axes[f64](values, [0, 2], false)!
	assert spread.shape == [2]
	assert math.is_nan(spread.get_nth(0))
	assert spread.get_nth(1) == 5

	kept := ptp_along_axes[f64](values, [0, 2], true)!
	assert kept.shape == [1, 2, 1]
	assert math.is_nan(kept.get_nth(0))
	assert kept.get_nth(1) == 5

	base := vtl.from_array[f64]([1, 2, 3, 4, 5, 6], [2, 3])!
	transposed := base.transpose([1, 0])!
	strided := ptp_along_axes[f64](transposed, [1], false)!
	assert strided.shape == [3]
	assert strided.to_array() == [3, 3, 3]

	empty_output := vtl.zeros[f64]([0, 2])
	empty_spread := ptp_along_axes[f64](empty_output, [1], false)!
	assert empty_spread.shape == [0]
}

fn test_ptp_propagates_nan_and_rejects_empty_or_invalid_reductions() {
	with_nan := vtl.from_1d[f64]([math.nan(), 3])!
	assert math.is_nan(ptp[f64](with_nan)!)
	mut f32_nan_data := []f32{len: 41, init: 1}
	f32_nan_data[20] = f32(math.nan())
	f32_nan_values := vtl.from_1d[f32](f32_nan_data)!
	assert math.is_nan(f64(ptp[f32](f32_nan_values)!))

	empty := vtl.zeros[f64]([0])
	expect_ptp_error(empty)
	expect_ptp_axis_error(empty, 0)

	values := vtl.from_1d[f64]([1, 2, 3])!
	expect_ptp_axis_error(values, 1)
	ptp_along_axes[f64](values, [0, -1], false) or { return }
	panic('expected duplicate axes to return an error')
}

fn expect_ptp_error(t &vtl.Tensor[f64]) {
	ptp[f64](t) or { return }
	panic('expected an empty tensor to return an error')
}

fn expect_ptp_axis_error(t &vtl.Tensor[f64], axis int) {
	ptp_along_axis[f64](t, axis, false) or { return }
	panic('expected an invalid axis reduction to return an error')
}
