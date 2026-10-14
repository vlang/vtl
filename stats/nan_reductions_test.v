module stats

import math
import vtl

fn test_nan_sum_product_and_extrema_ignore_nan() ! {
	values := vtl.from_1d([2.0, math.nan(), -3.0, 4.0])!
	assert nansum(values) == 3.0
	assert nanprod(values) == -24.0
	assert nanmin(values) == -3.0
	assert nanmax(values) == 4.0
	all_nan := vtl.from_1d([math.nan(), math.nan()])!
	assert nansum(all_nan) == 0.0
	assert nanprod(all_nan) == 1.0
	assert math.is_nan(nanmin(all_nan))
	assert math.is_nan(nanmax(all_nan))
	empty := vtl.from_1d([]f64{})!
	assert nansum(empty) == 0.0
	assert nanprod(empty) == 1.0
	assert math.is_nan(nanmin(empty))
	assert math.is_nan(nanmax(empty))
}

fn test_nan_axis_sum_product_and_extrema_shapes_and_empty_slices() ! {
	values := vtl.from_array([1.0, math.nan(), 3.0, 4.0, 5.0, 6.0], [2, 3])!
	assert nansum_axis(values, 1)!.to_array() == [4.0, 15.0]
	assert nanprod_axis(values, -1)!.to_array() == [3.0, 120.0]
	assert nanmin_axis(values, 0)!.to_array() == [1.0, 5.0, 3.0]
	assert nanmax_axis(values, 0)!.to_array() == [4.0, 5.0, 6.0]
	assert nansum_axis_keepdims(values, 1)!.shape == [2, 1]
	assert nanprod_axis_keepdims(values, 0)!.shape == [1, 3]
	assert nanmin_axis_keepdims(values, -1)!.shape == [2, 1]
	assert nanmax_axis_keepdims(values, 0)!.shape == [1, 3]

	missing := vtl.from_array([math.nan(), 2.0, math.nan(), math.nan()], [2, 2])!
	sums := nansum_axis(missing, 1)!
	products := nanprod_axis(missing, 1)!
	mins := nanmin_axis(missing, 1)!
	maxs := nanmax_axis(missing, 1)!
	assert sums.to_array() == [2.0, 0.0]
	assert products.to_array() == [2.0, 1.0]
	assert mins.get([0]) == 2.0
	assert math.is_nan(mins.get([1]))
	assert maxs.get([0]) == 2.0
	assert math.is_nan(maxs.get([1]))

	empty_axis := vtl.empty[f64]([2, 0])
	assert nansum_axis(empty_axis, 1)!.to_array() == [0.0, 0.0]
	assert nanprod_axis(empty_axis, 1)!.to_array() == [1.0, 1.0]
	assert math.is_nan(nanmin_axis(empty_axis, 1)!.get([0]))
	assert math.is_nan(nanmax_axis(empty_axis, 1)!.get([1]))
}

fn test_nan_axis_reductions_reject_scalar_and_invalid_axis() ! {
	scalar := vtl.empty[f64]([], memory: .row_major)
	scalar.set([], 1.0)
	if _ := nansum_axis(scalar, 0) {
		assert false, 'axis reduction must reject scalar inputs'
	}
	values := vtl.from_2d([[1.0, 2.0]])!
	if _ := nanmax_axis(values, 2) {
		assert false, 'axis reduction must reject out-of-range axes'
	}
}

fn test_nan_multi_axis_reductions_and_shapes() ! {
	values := vtl.from_array([math.nan(), 2.0, 3.0, 4.0, 5.0, math.nan(), 7.0, 8.0], [
		2,
		2,
		2,
	])!
	assert nansum_axes(values, [0, -1], false)!.shape == [2]
	assert nansum_axes(values, [0, -1], false)!.to_array() == [7.0, 22.0]
	assert nanprod_axes(values, [0, 2], true)!.shape == [1, 2, 1]
	assert nanprod_axes(values, [0, 2], true)!.to_array() == [10.0, 672.0]
	assert nanmin_axes(values, [0, 2], false)!.to_array() == [2.0, 3.0]
	assert nanmax_axes(values, [0, 2], false)!.to_array() == [5.0, 8.0]
	assert nanmax_axes(values, [], false)!.to_array()[1] == 2.0
	if _ := nanmax_axes(values, [0, -3], false) {
		assert false, 'duplicate axes must return an error'
	}
	empty_axis := vtl.empty[f64]([2, 0])
	assert nansum_axes(empty_axis, [1], false)!.to_array() == [0.0, 0.0]
	assert nanprod_axes(empty_axis, [1], false)!.to_array() == [1.0, 1.0]
	assert math.is_nan(nanmin_axes(empty_axis, [1], false)!.get([0]))
	assert math.is_nan(nanmax_axes(empty_axis, [1], false)!.get([1]))
}

fn test_nan_sum_product_explicit_accumulator_types() {
	values := vtl.from_1d([f32(1e8), f32(1.0), f32(math.nan()), f32(-1e8)])!
	assert nansum_as[f32, f64](values) == 1.0
	products := vtl.from_1d([f32(1e20), f32(1e-20), f32(math.nan())])!
	assert math.abs(nanprod_as[f32, f64](products) - 1.0) < 1e-6
	assert nansum_as[f32, f64](vtl.from_1d([]f32{})!) == 0.0
	assert nanprod_as[f32, f64](vtl.from_1d([f32(math.nan())])!) == 1.0
}

fn test_nan_sum_product_explicit_accumulator_types_for_axis_reductions() ! {
	values := vtl.from_array([f32(1e8), f32(math.nan()), f32(1), f32(-1e8), f32(2), f32(math.nan()),
		f32(3), f32(4)], [2, 2, 2])!
	sums := nansum_axis_as[f32, f64](values, -1, false)!
	assert sums.shape == [2, 2]
	assert sums.to_array() == [1e8, -99999999.0, 2.0, 7.0]
	kept := nansum_axis_as[f32, f64](values, 1, true)!
	assert kept.shape == [2, 1, 2]
	assert kept.to_array() == [100000001.0, -1e8, 5.0, 4.0]

	products := nanprod_axis_as[f32, f64](values, 1, false)!
	assert products.to_array() == [1e8, -1e8, 6.0, 4.0]
}

fn test_nan_sum_product_explicit_accumulator_types_for_multi_axis_reductions() ! {
	values := vtl.from_array([f32(1e8), f32(math.nan()), f32(1), f32(-1e8), f32(2), f32(math.nan()),
		f32(3), f32(4)], [2, 2, 2])!
	sums := nansum_axes_as[f32, f64](values, [0, -1], false)!
	assert sums.shape == [2]
	assert sums.to_array() == [100000002.0, -99999992.0]
	kept := nansum_axes_as[f32, f64](values, [0, -1], true)!
	assert kept.shape == [1, 2, 1]
	assert kept.to_array() == sums.to_array()
	products := nanprod_axes_as[f32, f64](values, [0, 2], true)!
	assert products.shape == [1, 2, 1]
	assert products.to_array() == [200000000.0, -1200000000.0]

	unchanged := nansum_axes_as[f32, f64](values, [], false)!
	assert unchanged.shape == values.shape
	assert unchanged.get_nth(2) == 1.0
	assert math.is_nan(unchanged.get_nth(1))
	if _ := nansum_axes_as[f32, f64](values, [0, -3], false) {
		assert false, 'dtype-controlled NaN reduction must reject duplicate axes'
	}
}
